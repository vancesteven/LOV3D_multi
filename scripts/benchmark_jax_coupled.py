#!/usr/bin/env python
"""Benchmark: coupled JAX solver (lax.scan) vs. the NumPy reference solver.

Measures, per configuration (a lateral-variation model that activates N
coupled spherical-harmonic modes on an Nr-point radial grid):

  1. NumPy    : pylov3d.solver._get_solution_coupled(...)               (1 call)
  2. JAX cold : pylov3d.jax_coupled.jax_get_solution_coupled_scan(...)   (1st call
                in a fresh process: precompute + XLA compile + solve)
  3. JAX warm : median of >=3 subsequent calls in that same process
                (scan/aux caches are module-level, keyed on (n_s, Coup, Gg))

and, in separate single-purpose subprocesses, the peak resident set size
(RSS) of a NumPy-only run and a JAX-only run.

Subprocess isolation is required: JAX's persistent module-level caches and
NumPy's BLAS thread pools otherwise contaminate each other's timings if run
in the same interpreter.  This script re-invokes itself as a worker via
``sys.executable`` for every individual measurement.

Usage
-----
    venvLOV3Dconv/bin/python scripts/benchmark_jax_coupled.py           # full sweep
    venvLOV3Dconv/bin/python scripts/benchmark_jax_coupled.py --quick   # 2 smallest configs

Internal (do not call directly, but harmless to):
    ... --worker {numpy,jax} --config-index N          # single timing measurement
    ... --rss-wrapper {numpy,jax} --config-index N     # timing + isolated peak RSS
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_MARKER = "BENCH_RESULT_JSON:"

# `pylov3d` is not pip-installed in this venv; it is only importable when the
# repo root is on sys.path. Running this file directly puts scripts/ (not the
# repo root) at sys.path[0], so add the repo root explicitly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Timeout applied to each individual subprocess call (worker or rss-wrapper).
# Per the task spec: abort a single measurement past ~5 minutes and move on.
MEASUREMENT_TIMEOUT_S = 300

N_WARM = 3  # number of JAX warm-cache repeats (>= 3 required)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# All configs use the Io three-layer viscoelastic model from
# pylov3d/tests/test_jax_coupled_scan.py's `coupled_solution` fixture, varying
# the lateral shear-modulus perturbation's spherical-harmonic degree (n_lv)
# and the coupling perturbation_order to sweep the number of coupled modes N.
# N is data-dependent (mode activation is combinatorial in n_lv/order), so the
# "target" below is a planning label; the ACTUAL N is measured and reported.
CONFIGS = [
    dict(id="tiny",         target="N~4",     n_lv=2,  m_lv=0, order=2, Nrbase=20),
    dict(id="small",        target="N~10-15", n_lv=10, m_lv=0, order=2, Nrbase=20),
    dict(id="medium",       target="N~25-40", n_lv=12, m_lv=0, order=6, Nrbase=20),
    dict(id="large",        target="N~60-120", n_lv=15, m_lv=0, order=7, Nrbase=20),
    dict(id="large_denser", target="N~60-120, denser Nr", n_lv=15, m_lv=0, order=7, Nrbase=60),
]


def build_problem(cfg: dict):
    """Build (model, forcing, numerics, couplings, lateral) exactly as the
    JAX coupled-scan test fixture does, for the given config's lateral degree,
    perturbation order, and radial-grid density.
    """
    from pylov3d.couplings import get_couplings
    from pylov3d.grid import set_boundary_indices
    from pylov3d.rheology import get_rheology, process_lateral_variations
    from pylov3d.types import make_forcing, make_interior_model, make_numerics

    raw_model = make_interior_model(
        R0_km=[800.0, 1600.0, 1821.6],
        rho0=[5150.0, 3300.0, 3000.0],
        mu0=[0.0, 60e9, 65e9],
        Ks0=[200e9, 200e9, 200e9],
        eta0=[None, 1e19, None],
    )
    forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=3, method="combination", Nrbase=cfg["Nrbase"],
        perturbation_order=cfg["order"],
    )
    numerics, raw_model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(raw_model, forcing)
    model, lateral = process_lateral_variations(
        model, forcing, mu_variable={1: [(cfg["n_lv"], cfg["m_lv"], 0.1)]},
    )
    couplings = get_couplings(
        lateral.variations, forcing.n, forcing.m,
        perturbation_order=numerics.perturbation_order,
    )
    return model, forcing, numerics, couplings, lateral


# ---------------------------------------------------------------------------
# Worker mode: runs inside a fresh subprocess, prints one JSON line of results
# ---------------------------------------------------------------------------

def _worker_numpy(cfg: dict) -> dict:
    from pylov3d.solver import _get_solution_coupled

    model, forcing, numerics, couplings, lateral = build_problem(cfg)
    N = len(couplings.n_s)
    Nr = int(numerics.Nr)

    t0 = time.perf_counter()
    y, r, Y, aux = _get_solution_coupled(model, forcing, numerics, couplings, lateral)
    t1 = time.perf_counter()

    return {"backend": "numpy", "N": N, "Nr": Nr, "time_s": t1 - t0}


def _worker_jax(cfg: dict) -> dict:
    from pylov3d.jax_coupled import jax_get_solution_coupled_scan

    model, forcing, numerics, couplings, lateral = build_problem(cfg)
    N = len(couplings.n_s)
    Nr = int(numerics.Nr)

    t0 = time.perf_counter()
    y, r, Y, aux = jax_get_solution_coupled_scan(
        model, forcing, numerics, couplings, lateral,
    )
    t1 = time.perf_counter()
    cold_s = t1 - t0

    warm_times = []
    for _ in range(N_WARM):
        t0 = time.perf_counter()
        y, r, Y, aux = jax_get_solution_coupled_scan(
            model, forcing, numerics, couplings, lateral,
        )
        t1 = time.perf_counter()
        warm_times.append(t1 - t0)

    return {
        "backend": "jax", "N": N, "Nr": Nr,
        "cold_s": cold_s,
        "warm_times_s": warm_times,
        "warm_median_s": statistics.median(warm_times),
    }


def run_worker(backend: str, config_index: int) -> None:
    cfg = CONFIGS[config_index]
    if backend == "numpy":
        result = _worker_numpy(cfg)
    elif backend == "jax":
        result = _worker_jax(cfg)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    result["config_id"] = cfg["id"]
    print(RESULT_MARKER + json.dumps(result))


# ---------------------------------------------------------------------------
# RSS-wrapper mode: spawns exactly one worker child, then reads this
# process's own RUSAGE_CHILDREN (guaranteed to reflect only that one child,
# since this wrapper process itself was freshly spawned by the orchestrator
# and has had no other children reap before this point).
# ---------------------------------------------------------------------------

def run_rss_wrapper(backend: str, config_index: int) -> None:
    script = str(Path(__file__).resolve())
    proc = subprocess.run(
        [sys.executable, script, "--worker", backend, "--config-index", str(config_index)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    result = _extract_result(proc.stdout, backend, config_index)
    if result is None:
        result = {
            "backend": backend, "config_id": CONFIGS[config_index]["id"],
            "error": True,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-4000:],
        }
    # macOS ru_maxrss is in BYTES; Linux would be KiB (not relevant here,
    # but noted for portability).
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["peak_rss_bytes"] = ru.ru_maxrss
    print(RESULT_MARKER + json.dumps(result))


def _extract_result(stdout: str, backend: str, config_index: int) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    return None


# ---------------------------------------------------------------------------
# Orchestrator (default mode)
# ---------------------------------------------------------------------------

def _call_rss_wrapper(backend: str, config_index: int) -> dict:
    script = str(Path(__file__).resolve())
    label = f"{backend} config[{config_index}]={CONFIGS[config_index]['id']}"
    print(f"  -> running {label} ...", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, script, "--rss-wrapper", backend, "--config-index", str(config_index)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=MEASUREMENT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"     ABORTED (exceeded {MEASUREMENT_TIMEOUT_S}s)")
        return {
            "backend": backend, "config_id": CONFIGS[config_index]["id"],
            "aborted": True,
        }
    result = _extract_result(proc.stdout, backend, config_index)
    if result is None:
        print(f"     FAILED (no result JSON; returncode={proc.returncode})")
        print(proc.stderr[-4000:])
        return {
            "backend": backend, "config_id": CONFIGS[config_index]["id"],
            "error": True, "returncode": proc.returncode,
        }
    return result


def _fmt_s(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}s" if x >= 1 else f"{x * 1000:.1f}ms"


def _fmt_bytes(b) -> str:
    if b is None:
        return "n/a"
    return f"{b / 1e9:.2f} GB"


def run_benchmark(quick: bool) -> list[dict]:
    configs = CONFIGS[:2] if quick else CONFIGS
    rows = []
    for idx, cfg in enumerate(CONFIGS):
        if cfg not in configs:
            continue
        print(f"\n=== Config '{cfg['id']}' ({cfg['target']}): "
              f"n_lv={cfg['n_lv']}, order={cfg['order']}, Nrbase={cfg['Nrbase']} ===")

        numpy_res = _call_rss_wrapper("numpy", idx)
        jax_res = _call_rss_wrapper("jax", idx)

        N = numpy_res.get("N") or jax_res.get("N")
        Nr = numpy_res.get("Nr") or jax_res.get("Nr")

        row = {
            "config_id": cfg["id"],
            "target": cfg["target"],
            "n_lv": cfg["n_lv"],
            "order": cfg["order"],
            "Nrbase": cfg["Nrbase"],
            "N": N,
            "Nr": Nr,
            "numpy_time_s": numpy_res.get("time_s"),
            "numpy_aborted": numpy_res.get("aborted", False),
            "numpy_peak_rss_bytes": numpy_res.get("peak_rss_bytes"),
            "jax_cold_s": jax_res.get("cold_s"),
            "jax_warm_median_s": jax_res.get("warm_median_s"),
            "jax_warm_times_s": jax_res.get("warm_times_s"),
            "jax_aborted": jax_res.get("aborted", False),
            "jax_peak_rss_bytes": jax_res.get("peak_rss_bytes"),
        }
        if row["numpy_time_s"] and row["jax_warm_median_s"]:
            row["speedup_warm_vs_numpy"] = row["numpy_time_s"] / row["jax_warm_median_s"]
        else:
            row["speedup_warm_vs_numpy"] = None
        rows.append(row)

        print(f"  N={N}  Nr={Nr}")
        print(f"  NumPy:      {_fmt_s(row['numpy_time_s'])}"
              f"  (peak RSS {_fmt_bytes(row['numpy_peak_rss_bytes'])})"
              + ("  [ABORTED]" if row["numpy_aborted"] else ""))
        print(f"  JAX cold:   {_fmt_s(row['jax_cold_s'])}")
        print(f"  JAX warm:   {_fmt_s(row['jax_warm_median_s'])} (median of {N_WARM})"
              f"  (peak RSS {_fmt_bytes(row['jax_peak_rss_bytes'])})"
              + ("  [ABORTED]" if row["jax_aborted"] else ""))
        if row["speedup_warm_vs_numpy"]:
            print(f"  Speedup (warm JAX vs NumPy): {row['speedup_warm_vs_numpy']:.1f}x")

    return rows


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    header = (
        f"{'config':<14}{'N':>5}{'Nr':>6}{'NumPy':>11}{'JAX cold':>11}"
        f"{'JAX warm':>11}{'speedup':>9}{'NumPy RSS':>12}{'JAX RSS':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        speedup = f"{row['speedup_warm_vs_numpy']:.1f}x" if row["speedup_warm_vs_numpy"] else "n/a"
        print(
            f"{row['config_id']:<14}"
            f"{str(row['N']):>5}"
            f"{str(row['Nr']):>6}"
            f"{_fmt_s(row['numpy_time_s']):>11}"
            f"{_fmt_s(row['jax_cold_s']):>11}"
            f"{_fmt_s(row['jax_warm_median_s']):>11}"
            f"{speedup:>9}"
            f"{_fmt_bytes(row['numpy_peak_rss_bytes']):>12}"
            f"{_fmt_bytes(row['jax_peak_rss_bytes']):>11}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                         help="Only run the two smallest configs.")
    parser.add_argument("--worker", choices=["numpy", "jax"], default=None,
                         help=argparse.SUPPRESS)
    parser.add_argument("--rss-wrapper", choices=["numpy", "jax"], default=None,
                         help=argparse.SUPPRESS)
    parser.add_argument("--config-index", type=int, default=None,
                         help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args.worker, args.config_index)
        return
    if args.rss_wrapper is not None:
        run_rss_wrapper(args.rss_wrapper, args.config_index)
        return

    print("Coupled JAX vs NumPy benchmark")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Python:    {sys.executable}")
    print(f"Quick:     {args.quick}")

    rows = run_benchmark(quick=args.quick)
    print_table(rows)
    print("\nRaw results (JSON):")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
