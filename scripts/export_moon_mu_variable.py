# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Export the Moon Airy-crust lateral ``mu_variable`` to an .npz for MATLAB.

TASK-035 (original anchor, degree-1-removed field) / TASK-038 (re-anchor of
the shipped dichotomy field): the Moon lateral stage (``pylov3d.moon_lateral``,
TASK-031) needs an independent native-MATLAB anchor because TASK-034's
k2m-vs-GRAIL comparison depends on that spectrum. This mirrors
:func:`pylov3d.mars_lateral.export_mu_variable_lateral` (via
``scripts/export_mars_dwak_mu_variable.py``, the exact precedent for this
file) but for the Moon crust field:

  * ``dt`` from :func:`pylov3d.moon_lateral.crustal_thickness_variation`
    (LOLA relief above the GRAIL equipotential surface, Airy-compensated;
    C00 always removed, C20 removed by the shipped default
    ``include_c20=False``, and degree 1 RETAINED by the shipped default
    ``include_degree1=True`` — the nearside-farside dichotomy, PI decision
    2026-08-14; pass ``include_degree1=False`` for the pre-decision field
    that TASK-035 anchored);
  * complex ``mu_variable`` from
    :func:`pylov3d.moon_lateral.mu_variable_from_topography` (the same
    ``_dmu_ddt_coeff`` linearization + ``_real_sh_to_complex_mu_variable``
    conversion the Mars Airy/DWAK paths use, imported unmodified from
    :mod:`pylov3d.mars_lateral`).

Read-only use of the pylov3d modules -- nothing in the package is edited.
Same npz field layout / complex-SH convention as the Mars exports, so the
committed MATLAB reader (adapted here as ``read_moon_mu_variable_npz`` in
``scripts/moon_lateral_cross_check.m``) parses it unchanged.

Unlike Mars, there is no ``mu_scale`` free parameter here: the Moon
crust/mantle rigidities are fixed constants of the as-built ten-layer Weber
model (:data:`pylov3d.moon.LAYER_MU`), not a fitted/scaled quantity.

Run from the repo root:
    python scripts/export_moon_mu_variable.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pylov3d.moon import LAYER_MU
from pylov3d.moon_lateral import (
    CRUST_LAYER_INDEX,
    CRUST_THICKNESS_M,
    MANTLE_LAYER_INDEX,
    crustal_thickness_variation,
    mu_variable_from_topography,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO_ROOT / "data" / "moon" / "moon_mu_variable_lateral.npz"

_README = """\
Moon Airy-crust lateral rigidity export -- for the native-MATLAB lateral
anchor (scripts/moon_lateral_cross_check.m). Originally the TASK-035
deliverable (degree-1-removed field, 20 amplitudes); since the 2026-08-14
PI decision this carries the shipped dichotomy-retaining field
(include_degree1=True, 23 amplitudes -- re-anchored in TASK-038). The
include_degree1 key in this archive records which field this is; the
superseded 20-amplitude export is pinned in git history at 47b5377.

This is the committed lmax=4, shipped-default (include_c20=False) field from
pylov3d.moon_lateral.mu_variable_from_topography -- the same field the
regenerated Python lateral Love-number spectrum (docs/MOON_MODEL.md) was
computed from. It exists so MATLAB is handed the *exact* complex mu_variable
Python used, isolating the solver's sign convention from any
spherical-harmonic re-derivation.

Contents
--------
layer_idx, n, m, amp_real, amp_imag : crust-layer mu_variable entries
    (complex amp = amp_real + 1j*amp_imag). SAME complex-SH convention as the
    Mars exports (pylov3d/mars_lateral.py::_real_sh_to_complex_mu_variable,
    which moon_lateral imports and reuses unmodified). layer_idx is 0-based
    Python (9 = crust). MATLAB Interior_Model is 1-based, so the Moon crust
    is Interior_Model(10) and the sub-crust mantle shell is Interior_Model(9)
    (MATLAB layer index = python index + 1).
dt_n, dt_m, dt_val_m : the real 4pi-normalized Airy crustal-thickness SH
    coefficients [m] this was derived from (m >= 0: cosine C_nm; m < 0: sine
    S_n|m|).
mu_crust_pa, mu_mantle_pa [Pa], crust_thickness_m [m], crust_layer_index,
    mantle_layer_index, lmax, include_c20 : provenance / reproducibility.
    There is no mu_scale for the Moon: mu_crust_pa/mu_mantle_pa are fixed
    model constants (pylov3d.moon.LAYER_MU[9]/[8]), not a fitted parameter.

The Weber Moon is TEN layers (not four like Mars) and INCLUDES A FLUID OUTER
CORE (layer index 2, Vs=0, ocean=1) -- the part of scripts/
moon_lateral_cross_check.m with no Mars precedent; see that script's own
comments for the coupled-ocean boundary-condition path this exercises.

eta0 convention warning (same as the Mars exports): purely elastic model --
leave eta0 EMPTY ([]) on all ten MATLAB layers; eta0 = NaN NaN-poisons the
MATLAB solve (TASK-014 pt-1 gotcha).
"""


def export_moon_mu_variable(
    path: Path | str = DEFAULT_OUT,
    lmax: int = 4,
    include_c20: bool = False,
    include_degree1: bool = True,
) -> Path:
    """Write the Moon crust ``mu_variable`` + provenance to ``path`` (.npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dt = crustal_thickness_variation(
        lmax=lmax, include_c20=include_c20, include_degree1=include_degree1,
    )
    mu_variable = mu_variable_from_topography(
        lmax=lmax, include_c20=include_c20, include_degree1=include_degree1,
    )
    entries = mu_variable[CRUST_LAYER_INDEX]

    layer_idx = np.full(len(entries), CRUST_LAYER_INDEX, dtype=int)
    n_arr = np.array([e[0] for e in entries], dtype=int)
    m_arr = np.array([e[1] for e in entries], dtype=int)
    amp_arr = np.array([e[2] for e in entries], dtype=complex)

    dt_items = sorted(dt.items())
    dt_n = np.array([k[0] for k, _v in dt_items], dtype=int)
    dt_m = np.array([k[1] for k, _v in dt_items], dtype=int)
    dt_val_m = np.array([v for _k, v in dt_items], dtype=float)

    np.savez(
        path,
        layer_idx=layer_idx, n=n_arr, m=m_arr,
        amp_real=amp_arr.real, amp_imag=amp_arr.imag,
        dt_n=dt_n, dt_m=dt_m, dt_val_m=dt_val_m,
        mu_crust_pa=LAYER_MU[CRUST_LAYER_INDEX],
        mu_mantle_pa=LAYER_MU[MANTLE_LAYER_INDEX],
        crust_thickness_m=CRUST_THICKNESS_M,
        crust_layer_index=CRUST_LAYER_INDEX,
        mantle_layer_index=MANTLE_LAYER_INDEX,
        lmax=lmax,
        include_c20=include_c20,
        include_degree1=include_degree1,
        readme=_README,
    )
    return path


if __name__ == "__main__":
    out = export_moon_mu_variable()
    data = np.load(out, allow_pickle=True)
    amp = data["amp_real"] + 1j * data["amp_imag"]
    print(f"wrote {out}")
    print(f"  {len(amp)} crust mu_variable entries (layer_idx == {int(data['crust_layer_index'])})")
    print(f"  lmax = {int(data['lmax'])}, include_c20 = {bool(data['include_c20'])}")
    print("  (n, m)  amp = amp_real + i*amp_imag:")
    order = np.lexsort((data["m"], data["n"]))
    for i in order:
        print(f"    ({int(data['n'][i]):2d},{int(data['m'][i]):3d})  "
              f"{amp[i].real:+.6e} {amp[i].imag:+.6e}i")
    print(f"  max|amp| = {np.max(np.abs(amp)):.6e}")
    # the forcing-mode dt C20 is 0 by construction (removed by the shipped
    # default include_c20=False); report the largest dt coefficient instead.
    #
    # NOTE the distinction, which is load-bearing elsewhere in this project:
    # this is the largest SH *coefficient*, NOT the spatial maximum of the
    # synthesized field.  For the shipped dichotomy field the spatial max is
    # 32.616 km (largest coefficient: |dt(1,1)| = 6.834 km), and it is the
    # spatial max that sets the positivity margin max|dmu/mu_bar| = 0.9898
    # (the superseded degree-1-removed field's margin was 0.9902, quoted in
    # TASK-031b/036).  Use
    # crustal_thickness_diagnostics()['max_abs_dt_m'] for that number.
    idx_max_dt = int(np.argmax(np.abs(data["dt_val_m"])))
    print(f"  max|dt| SH coefficient = {abs(data['dt_val_m'][idx_max_dt]):.4f} m "
          f"at (n={int(data['dt_n'][idx_max_dt])}, m={int(data['dt_m'][idx_max_dt])})")
    print("    (spatial max of the synthesized field is larger — see note above)")
