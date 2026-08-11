# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Export the DWAK InSight-Moho crust ``mu_variable`` to an .npz for MATLAB.

TASK-029: the native-MATLAB anchor for the two first-order zonal coupling
channels (docs/tasks/TASK-029-matlab-first-order-channels.md) needs the exact
complex crust-layer ``mu_variable`` field the Python TASK-028 result was
computed from, so the MATLAB LOV3D solver can be handed *the identical field*
(rather than re-deriving the spherical-harmonic bookkeeping in MATLAB, where a
convention mismatch could masquerade as a sign error). This mirrors
:func:`pylov3d.mars_lateral.export_mu_variable_lateral` (the Airy-path export
that feeds ``scripts/mars_lateral_cross_check.m``) but for the DWAK field:

  * ``dt`` from :func:`pylov3d.mars_crust_models.moho_thickness_variation`
    (``R_topo - R_moho``, C00 removed, **C20 retained** -- the deliberate
    departure from the Airy path that activates the (2,0) channel);
  * complex ``mu_variable`` from
    :func:`pylov3d.mars_crust_models.mu_variable_from_dt` (the same
    ``_dmu_ddt_coeff`` linearization + ``_real_sh_to_complex_mu_variable``
    conversion the Airy path uses, imported unmodified).

Read-only use of the pylov3d modules -- nothing in the package is edited.
Same npz field layout / complex-SH convention as the Airy export, so the
committed MATLAB reader ``read_mars_mu_variable_npz`` parses it unchanged.

Run from the repo root:
    python scripts/export_mars_dwak_mu_variable.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pylov3d.mars import MARS, MARS_MU_SCALE
from pylov3d.mars_crust_models import moho_thickness_variation, mu_variable_from_dt
from pylov3d.mars_lateral import CRUST_LAYER_INDEX, LAYER_MU_CRUST, _mu_um_eff

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO_ROOT / "data" / "mars" / "mars_dwak_mu_variable.npz"

_README = """\
TASK-029 Mars DWAK (InSight-Moho) lateral rigidity export -- for the native-
MATLAB first-order-channel anchor (scripts/mars_first_order_channels.m).

This is the C20-RETAINING DWAK crustal field (TASK-028), not the Airy field
in mars_mu_variable_lateral.npz. It exists so MATLAB is handed the *exact*
complex mu_variable Python used, isolating the solver's sign convention from
any spherical-harmonic re-derivation.

Contents
--------
layer_idx, n, m, amp_real, amp_imag : crust-layer mu_variable entries
    (complex amp = amp_real + 1j*amp_imag), SAME complex-SH convention as the
    Airy export (pylov3d/mars_lateral.py::_real_sh_to_complex_mu_variable).
    layer_idx is 0-based (3 = crust). MATLAB layer index = python index + 1,
    so crust is Interior_Model(4).
dt_n, dt_m, dt_val_m : the real 4pi-normalized DWAK crustal-thickness SH
    coefficients [m] this was derived from (m >= 0: cosine C_nm; m < 0: sine
    S_n|m|). C00 removed, C20 RETAINED (unlike the Airy dt).
mu_crust_pa, mu_um_eff_pa [Pa], crust_thickness_m [m], mars_mu_scale [-],
    crust_layer_index, lmax, moho_model : provenance / reproducibility.

eta0 convention warning (same as the Airy export): purely elastic model --
leave eta0 EMPTY ([]) on all 4 MATLAB layers; eta0 = NaN NaN-poisons the
MATLAB solve (TASK-014 pt-1 gotcha).
"""


def export_dwak_mu_variable(
    path: Path | str = DEFAULT_OUT,
    model: str = "DWAK",
    lmax: int = 4,
    mu_scale: float | None = None,
) -> Path:
    """Write the DWAK crust ``mu_variable`` + provenance to ``path`` (.npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dt = moho_thickness_variation(model, lmax=lmax)
    mu_variable = mu_variable_from_dt(dt, mu_scale=mu_scale)
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
        mu_crust_pa=LAYER_MU_CRUST,
        mu_um_eff_pa=_mu_um_eff(mu_scale),
        crust_thickness_m=MARS["crust_thickness"],
        mars_mu_scale=(MARS_MU_SCALE if mu_scale is None else mu_scale),
        crust_layer_index=CRUST_LAYER_INDEX,
        lmax=lmax,
        moho_model=model,
        readme=_README,
    )
    return path


if __name__ == "__main__":
    out = export_dwak_mu_variable()
    data = np.load(out, allow_pickle=True)
    amp = data["amp_real"] + 1j * data["amp_imag"]
    print(f"wrote {out}")
    print(f"  {len(amp)} crust mu_variable entries (layer_idx == {int(data['crust_layer_index'])})")
    print(f"  moho_model = {str(data['moho_model'])}, lmax = {int(data['lmax'])}")
    print("  (n, m)  amp = amp_real + i*amp_imag:")
    order = np.lexsort((data["m"], data["n"]))
    for i in order:
        print(f"    ({int(data['n'][i]):2d},{int(data['m'][i]):3d})  "
              f"{amp[i].real:+.6e} {amp[i].imag:+.6e}i")
    # the two channels TASK-029 isolates, and the dt C20 that activates (2,0):
    dt_c20 = [v for nn, mm, v in zip(data["dt_n"], data["dt_m"], data["dt_val_m"])
              if nn == 2 and mm == 0]
    print(f"  dt C20 (retained) = {dt_c20[0]:+.4f} m" if dt_c20 else "  dt C20 absent!")
