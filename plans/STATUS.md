# Status — LOV3d-genai

Updated: 2026-09-30T03:30Z (claude-lov3d-genai)

Refresh the `Updated:` line and the affected sections in any session that
pushes commits, integrates artifacts, or changes a queue.

## Current focus

Bridge the Task-1 alteration state to real PlanetProfile Mars profiles: the
radial importer, mass-preserving reduction, and Love-number convergence gate
are in place and verified on a synthetic fixture; the next input is a real
PlanetProfile Mars radial artifact (none exists in-repo). Proposal-side
documentation in `external/SSS_2025_Mars` was synced to the code state on
2026-09-11 (see `PROPOSAL_ASSESSMENT_2026-09-11.md` there for open edits).

## In flight

| Item | Owner | Status | Artifact |
|---|---|---|---|
| Love-number convergence gate for radial reduction (`pylov3d/profile_convergence.py`) | claude-lov3d-genai | verified | `pylov3d/tests/test_profile_convergence.py` 10/10; `scripts/run_science_benchmarks.py` 174/174 (2026-08-28) |
| Liquid-core conversion in `reduced_shells_to_interior_model` | claude-lov3d-genai | verified | `test_profile_convergence.py::TestLiquidCoreConversion` |
| MATLAB TASK-046 anchors committed (`data/tests/io/*_anchor.mat`) | claude-lov3d-genai | verified | `scripts/io_compare_identical_coefficients_anchor.py` strict parity PASS, 1.3e-11 |
| Proposal docs sync (`external/SSS_2025_Mars` 51e5de5, a48d0b6, 9fe2ff1) | claude-lov3d-genai | implemented, unverified | no TeX toolchain to compile; static `\ref`/`\cite` check clean |
| Convergence gate on a real PlanetProfile Mars artifact | — | not implemented | needs artifact from planetprofile-genai lane |

Status must be one of `verified` / `implemented, unverified` / `not implemented`.

## Next: Berne et al. 2026 parity and extensions (proposed 2026-09-28)

Source: `external/SSS_2025_Mars/PROPOSAL_ASSESSMENT_2026-09-28_BERNE2026.md`
(capability-gap pass by a Claude general-purpose subagent, Opus 5.5; adjudicated
by claude-lov3d-genai). Berne 2026 used LOV3D (elastic) for Mars annual
degree-3 tidal tomography; the Zenodo MAP script (10.5281/zenodo.20823472) is
the natural parity reference. Priority order:

| # | Item | Effort | Status | Verification bar |
|---|---|---|---|---|
| B1 | Extended-Love-tensor API: loop forcing (2,m'), m'=-2..2, return K^{l'm'}_{lm} in real cos/sin basis | S | verified — `pylov3d/extended_love.py`; `test_extended_love.py` 14/14 fast + slow `test_mars_columns_match_matlab` (m'=0,1,2 vs native MATLAB, 1.2e-12 abs), re-run by manager 2026-09-28. m'<0 covered only by the Python-side conjugate identity | met 2026-09-28: `test_extended_love.py` 19/19 — zero-amplitude diag = 1D k2; columns m'=0,1,2 vs `mars_lateral_cross_check.mat` to 1e-10 (m'<0 via the tested conjugate identity, no direct MATLAB run); rotation-covariance CS-convention check |
| B2 | Berne 1D reference (ED Table 2) + mantle-l=1 MATLAB anchor | S | verified | ED Table 2 transcribed from paper image -> pylov3d/berne2026.py, k2=0.176316 within 1.2 sigma of 0.169+/-0.006 (`test_berne2026.py`); cross-solver lateral anchor exceeded the bar: pylov3d matches the shipped Zenodo LOV3D_open responses to 1.0e-12 worst rel over 365 modes x 5 forcings with degree-1..3 structure (`test_berne_zenodo_parity.py`, `scripts/berne_zenodo_convention_check.py`) |
| B3 | Ephemeris forcing f_p(t) + annual projection (Berne eq. 5-7) + polar-cap term (eq. 9-10) | M | not implemented | reproduce Zenodo script dC/dS (l=2,3) to <=1e-3 relative |
| B4 | Degree-3 forcing anchor (numpy path; JAX rejects n!=2) | S | verified — `pylov3d/tests/test_love_degree_n.py` 4/4; k2,k3,k4 vs analytic incompressible sphere, rel err 1.6e-9/6.8e-10/6.3e-10 | met 2026-09-28 (analytic half): `test_love_degree_n.py` k_n n=2,3,4 vs Love 1911 sphere to 1e-8; PyALMA3/MATLAB cross-check still open |
| B5 | Lateral-coefficient likelihood + sampler on dC/dS | M | not implemented | injected-truth recovery within 1 sigma; Berne ED Table 3 medians within their intervals |
| B6 | Large-amplitude convergence (perturbation_order 2-4 at 60-80% mu) | S | not implemented | successive dC3m changes < 1% of sigma; one MATLAB cross-check |
| B7 | Anelastic (Andrade/Burgers) lateral rheology at annual/semiannual periods | L | not implemented | 1D Andrade k_n vs PyALMA3 < 1e-4; Maxwell-limit Gate C parity |
| B8 | T, X_Fe, hydration, NAM water -> (mu, K, rho, Q) frequency-dependent mapping | M | not implemented | reproduce Berne dmu/dT = -0.204 GPa/K at 5.94e7 s and Fig. 3 contours to 5% |
| B9 | Joint thermal-vs-hydration discriminator (tidal A/B + GMM-3 + COM-COF + InSight Q + EM) | M | not implemented | synthetic-truth recovery; whitened cross-correlation < 0.95 |
| B10 | Lateral density / Moho relief in the tidal solve | L | not implemented | check Berne "<0.3%" claim against MATLAB/FE reference |

Next (B2, revised): Zenodo example is a 2-layer non-dimensional model on stock
LOV3D_open (not ED Table 2). Build a pylov3d parity anchor against its shipped
MATLAB `k2_responses.txt`, running both odd-m sine conventions to decide which
one the example used (see `coordination/open-questions.md`). Archive currently
only in /tmp; must be copied into `data/` (CC-BY-4.0) before a test can use it.

Follow-ups: (i) `mars_detectability.required_stokes_amplitude` applies
c_f/c_resp = 1/sqrt2 when m_forcing_solve != 0, which disagrees with K_real's
sqrt2|K| per standing component (shipped bounds use m_forcing_solve=0, so
unaffected) — not implemented, needs a targeted check; (ii) direct MATLAB
anchor for m'<0 forcing needs `scripts/mars_lateral_cross_check.m` rerun with
`forcing_orders = [-2 -1]`.

Open physics question (needs adjudication, not assumption): in the body-fixed
frame m'!=0 solar forcing oscillates (semi)diurnally; how much survives the
annual projection of eq. 7 must be checked by running the time series (B3).
Open-questions item 2 (sine convention) is RESOLVED 2026-09-28: the Zenodo
archive's shipped outputs use run_forward_shear.m's (-1)^m sine map, not the
canonical get_rheology.m one; solver parity is 1.0e-12. Whether the paper's
production pipeline shares that convention is a question for the authors.

Re-scope needed: `test_mars_detectability.py:226` ("no off-(2,0) mode
detectable") and TASK-043 (thermal-vs-crust 0.05 sigma) assume (2,0)-only
forcing, crust/L=2 templates, seismic-timescale beta, sigma=1.1e-11. Berne's
detection is l=1 **mantle** mu under the full solar tide with annual softening.
Not a contradiction, but the null result's scope must be stated explicitly.

## Preliminary: pointwise-connectivity higher-degree study (2026-09-29)

PRELIMINARY per Steve: Berne will provide updated calculations that supersede
these numbers; machinery is verified, numbers are for scoping only.
`pylov3d/mars_hydration_connectivity.py` applies Voigt/Hill/Reuss pointwise to
the crustal-thickness hydration field (Fig-3 geometry) and evaluates the full
coupled response via the extended Love tensor. 12-combo sweep (central/low x 3
laws x f_h {0.1, 0.5}, lmax_field=4, lmax_out=2, Nrbase=30, order 2), CSV at
`data/tests/mars/connectivity_higher_degree_preliminary.csv`:

- delta_k2_mean spans 6.3e-5 to 8.0e-4, consistent with the Fig-3 mean-only
  ranges; pointwise-vs-mean-only shifts are small (Jensen-convexity direction
  verified in tests).
- The degree-3 response to (2,0) forcing is CONVERGED in lateral truncation:
  lmax_out 2 -> 4 moves it only 1.6-3.4% (f_h=0.5 spot-check,
  `connectivity_higher_degree_lmax4_check.csv`). Robust range at f_h=0.5:
  1.8e-5 to 3.2e-5 across scenario x law.
- The k_2m splitting is NOT converged at lmax_out=2: it grows 10-40x at
  lmax_out=4 (to 1.4e-6..8.9e-6 at f_h=0.5) because the diagonal shift is
  second order and needs the degree-3/4 lateral content. The 02:30Z claim that
  the splitting m-ordering is connectivity-sensitive is WITHDRAWN — at
  lmax_out=4 every combo orders m=0 > m=1 > m=2; the Reuss m=2-largest
  ordering was a truncation artifact.
- 2026-09-30: full 45-combo curve sweep at lmax_out=2
  (`connectivity_higher_degree_curves.csv`) feeds the new degree-3 panel of
  proposal Figure 3 (pushed, 7dec481); l=3 numbers only, which are converged.
- Net: the l=3 channel (Berne's annual-tomography channel) remains the
  stronger lateral observable, by ~3-4x over the best k_2m splitting at
  lmax_out=4 (down from the spurious 30-60x at lmax_out=2). Quote splitting
  numbers only from lmax_out>=4 runs.

## Blockers

- Pushes go to `myfork`, **not** `origin` (`origin` is upstream `mroviranavarro/LOV3D_multi` and rejects with 403).
- No real PlanetProfile Mars radial artifact in-repo; planetprofile-genai uses its own `plans/CODEX-QUEUE.md` convention (not the shared inbox).
- Python env: reference venv `~/src/LOV3d_multi/venvLOV3Dconv-linux` is broken on this host; a working `.venv-linux` exists in this tree (git-ignored). `py3nj` must be copied from the reference venv (no dev headers to build it).
- No TeX toolchain on this host; proposal compile/page-count checks must run on Overleaf.
- `papers/berne2026tidal.pdf` is deliberately untracked (third-party copyright); `/papers/` is git-ignored.
