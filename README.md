# LOV3D

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=mroviranavarro/LOV3D_multi)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**LOV3D** is a package for computing the tidal response of viscoelastic self-gravitating bodies with lateral variations of interior properties. For a given interior structure and tidal load, the software solves the mass conservation, momentum conservation and Poisson equations and computes the tidal Love numbers. This is done in the spectral domain as detailed in [Rovira-Navarro et al. 2024](https://doi.org/10.3847/PSJ/ad381f).

The package exists in two implementations:
- **MATLAB** (`src/`): The original implementation supporting full 3D lateral variations
- **Python** (`pylov3d/`): A Python/NumPy port with a JAX (`lax.scan`, JIT) backend for the radial integration, validated against MATLAB, PyALMA3, and analytic references (Milestone 4 complete)

![logo](./docs/logo.png)

---

## pylov3d (Python)

### Status: Milestone 5 Complete

**Milestone 1 (1D spherically symmetric):** ✅ Complete  
Single-mode Love number computation for multi-layered bodies with viscoelastic interiors.

**Milestone 2 (3D lateral variations):** ✅ Complete  
Multi-mode coupling with lateral variations in rheology. Computes Love number spectra for bodies with laterally heterogeneous structure.

**Milestone 3 (MATLAB cross-validation):** ✅ Complete  
The Enceladus 2-layer benchmark validates against MATLAB LOV3D reference data: uniform k2 to <0.1% (0.0151953 vs 0.0151858), the amplitude sweep to <0.15%, and the full lateral Love spectrum (n=3/m=1, n=2/m=2, n=4/m=0, n=4/m=2) to <0.25%.

**Milestone 4 (independent benchmark + JAX port):** ✅ Complete  
- PyALMA3 cross-validation for identical 1D models (`test_benchmark_pyalma3.py`)
- PlanetProfile compatibility adapter (`compat.py`)
- JAX 1D propagator, then `jax.lax.scan` JIT radial integration (`jax_scan.py`)
- Coupled 8N×8N JAX scan port (`jax_coupled.py`): static-tensor einsum assembly, memoized JIT scan, full 4-tuple drop-in for the NumPy coupled solver including the auxiliary stress/strain recovery matrix (`jax_coupled_aux.py`). Matches the NumPy solver to ~1e-15 relative.
- Direct JAX↔MATLAB validation (`test_jax_matlab_validation.py`): the Enceladus lateral benchmark solved end-to-end by the JAX path matches published MATLAB Love-number spectra within the same per-order tolerances as the NumPy path, with no NumPy solver involved.
- Performance benchmark (`docs/BENCHMARK_jax_coupled.md`): on CPU the warm JAX path is 4.5–6.5× faster than NumPy for N=4–12 coupled modes and 3.7× at N=38, converging to parity near N≈100 where both are bound by dense 8N×8N linear solves.

**Milestone 5 (subsurface oceans):** ✅ Complete  
- 1D ocean path audited and repaired (it had never solved: a boundary-index convention bug made the 24×24 ocean BC system singular; the in-ocean propagator now uses MATLAB's Laplace-only formulation). Validated by the mu→0 fluid limit (4e-6 relative agreement) and by MATLAB: uniform Weber Moon k2 matches `k2_Q` to 2.2e-9.
- Coupled (multi-mode) ocean solver: 24N×24N boundary conditions (`bc_ocean_coupled.py`), in-ocean Laplace propagator (`ocean_coupled.py`), three-segment integration with identity restarts. Validated end-to-end against the published Weber Moon lateral-variation spectra (MATLAB/Qin, `data/tests/moon/`): order-1 modes to 2–5 ppm, forcing-mode deviation to ~0.3%.
- JAX coupled ocean scan (`jax_ocean_scan.py`): three-segment `lax.scan`, in-ocean segment matching NumPy to ~1e-19; full drop-in via the existing dispatch.
- Both solvers explicitly reject the degenerate configurations MATLAB handles incorrectly or not at all (ocean above the core, outermost ocean, zero-node layer above the ocean).

### Installation

```bash
cd pylov3d
pip install -e “.[test]”
```

Requires Python >= 3.10. Dependencies: NumPy >= 1.24, JAX >= 0.4.20.

### Quick Start

```python
from pylov3d import make_interior_model, make_forcing, make_numerics, get_love

# 4-layer Io model: core, deep mantle, asthenosphere, lithosphere
model = make_interior_model(
    R0_km=[965.0, 1591.6, 1791.6, 1821.6],
    rho0=[5150.0, 3244.0, 3244.0, 3244.0],
    mu0=[0.0, 6e10, 7.8e5, 6.5e10],
    eta0=[None, 1e20, 1e11, 1e23],
)

forcing = make_forcing(Td=153042.0, n=2, m=0, F=0.335)
numerics = make_numerics(n_layers=4, method=”combination”, Nrbase=100)

love, y_rad, model = get_love(model, forcing, numerics)
print(f”k2 = {complex(love.k[0]):.6e}”)
# k2 = 2.876674e-02 - 2.667945e-03j

# With lateral variations (5% shear modulus variation, n=2 m=0):
love_3d, y_rad, model = get_love(
    model, forcing, numerics,
    mu_variable={2: [(2, 0, 0.05)]}  # 5% variation in asthenosphere
)
print(f”Love number spectrum: {len(love_3d.k)} modes”)
# Multi-mode response due to lateral coupling
```

### Architecture

The Python version is a line-by-line algorithmic translation of the MATLAB source, preserving the same numerical methods, matrix formulations, and variable naming conventions. The package is organized into focused modules that mirror the MATLAB pipeline:

| Module | Lines | MATLAB Source | Purpose |
|--------|-------|---------------|---------|
| `types.py` | 213 | struct fields | NamedTuple data structures (InteriorModel, Forcing, NumericsConfig, etc.) |
| `constants.py` | 21 | scattered | Physical constants, array size limits |
| `grid.py` | 126 | `set_boundary_indices.m` | Radial grid setup (4 methods: variable, fixed, combination, manual) |
| `rheology.py` | 184 | `get_rheology.m` | Non-dimensionalization + Maxwell complex shear modulus |
| `propagator.py` | 349 | `get_solution.m` subfuncs | A-matrix builders (A1-A3, Aprop), gravity, Cash-Karp coefficients |
| `boundary_conditions.py` | 305 | `get_solution.m` BCs | BC assembly: 8x8 (no ocean) and 24x24 (with ocean) |
| `solver.py` | 306 | `get_solution.m` main | Cash-Karp RK5 integration of 8x8 fundamental matrix ODE |
| `love.py` | 142 | `get_Love.m` | Pipeline orchestrator + Love number extraction |
| `energy.py` | 620 | `get_energy.m` | Strain matrices (A14/A15), stress-strain, tidal dissipation (1D + coupled) |
| `energy_couplings.py` | 280 | `get_energy_couplings.m` | Energy coupling tensor (Wigner 9j products) |
| `couplings.py` | 340 | `get_couplings.m` | Mode coupling coefficients for lateral variations |
| `wigner.py` | 185 | external | Wigner 3j/6j/9j symbols via py3nj |
| `bodies.py` | 253 | `Select_Moon.m` | Planetary body catalog (Io, Europa, Enceladus, Titan, Ganymede) |
| `compat.py` | 139 | — | PlanetProfile compatibility adapter (PlanetStruct → InteriorModel) |
| `jax_propagator.py` | 407 | — | JAX 1D propagator (Python-loop increment) |
| `jax_scan.py` | 353 | — | JAX 1D `lax.scan` JIT radial integration |
| `jax_coupled.py` | 492 | — | JAX coupled 8N×8N assembly + memoized JIT scan + solve API |
| `jax_coupled_aux.py` | 90 | — | Auxiliary 3N×8N stress/strain recovery rows (jitted vmap) |
| `bc_ocean_coupled.py` | 125 | `get_solution.m` ocean BCs | 24N×24N coupled ocean boundary conditions |
| `ocean_coupled.py` | 42 | `get_solution.m` ocean Aprop | In-ocean Laplace-only coupled propagator |
| `jax_ocean_scan.py` | 310 | — | Three-segment JAX scan for ocean-bearing coupled models |
| **Total** | **~5,600** | **~4,900** | |

#### Key differences from MATLAB

1. **Data structures**: MATLAB uses nested structs with dynamic fields. Python uses typed NamedTuples with explicit fields, enforcing immutability and enabling static analysis.

2. **Normalization**: The MATLAB code normalizes in-place within the rheology module via struct field mutation. The Python version returns new InteriorModel instances with normalized fields, preserving the originals.

3. **Propagator matrices**: MATLAB builds A-matrices inside the integration loop. Python pre-caches material-dependent matrices (A1, A2) per layer and reconstructs only the radius-dependent propagator at each step, reducing redundant computation.

4. **Boundary conditions**: MATLAB assembles BCs via index arithmetic on large block matrices. Python uses explicit named slicing for clarity but preserves the identical block structure (8N x 8N no-ocean, 24N x 24N ocean).

5. **Complex arithmetic**: MATLAB handles complex numbers natively. Python uses `complex128` arrays throughout, with explicit `complex()` casts at extraction points.

6. **Ocean handling**: MATLAB uses a single code path with conditional blocks. Python separates ocean and no-ocean boundary conditions into distinct functions for clarity (`assemble_bc_no_ocean` vs `assemble_bc_ocean`).

7. **Module decomposition**: The ~1,900-line `get_solution.m` is split across `propagator.py`, `boundary_conditions.py`, and `solver.py`, making each component independently testable.

### Validation

674 tests across 45 test files (644 in the default fast lane, ~2 min; run the full set with `-m ""`), covering unit tests, physics tests, analytical cross-validation, lateral variation benchmarks, MATLAB reference cross-validation (both NumPy and JAX solver paths, with and without subsurface oceans), PyALMA3 cross-validation, and JAX↔NumPy equivalence.

#### Models tested

- **Uniform elastic sphere** (2-layer: tiny fluid core + elastic mantle)
  - h2 matches Kelvin/Love analytical formula within 3%: h2 = 5/(2(1 + 19mu/(2*rho*g*R)))
  - k2 matches the analytic gravity Love number k2 = 3h2/5 to <0.1% (0.038704), and is purely real (Im < 1e-10)
  - Zero tidal dissipation confirmed
  - Convergence with increasing radial resolution (Nrbase = 100, 200, 400)

- **Io 4-layer model** (core / deep mantle / asthenosphere / lithosphere)
  - k2 = 2.877e-02 - 2.668e-03j (magnitude O(0.01-1), dissipative)
  - Im(k2) < 0 (correct sign for dissipative body)
  - Resolution convergence verified (Nrbase = 50, 100, 200)
  - Nonzero tidal dissipation with finite energy profile
  - Pipeline roundtrip: `get_love` matches direct solver to machine precision (rel < 1e-12)

- **Viscoelastic limits**
  - High viscosity (eta = 1e30): approaches elastic behavior (|Im(k2)| < 1e-6)
  - Moderate viscosity (eta = 1e15): significant dissipation (Im(k2) < 0)
  - Viscosity sweep (eta = 1e10 to 1e25): resonance peak in |Im(k2)| confirmed

- **Energy consistency**
  - Elastic body: radial dissipation profile uniformly zero (atol < 1e-15)
  - Viscoelastic body: viscous layer dominates dissipation over elastic layers
  - Global dissipation formula scales correctly with -Im(k) and omega

#### Test breakdown

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test_propagator.py` | 41 | A-matrices, propagator, gravity, symmetries |
| `test_couplings.py` | 27 | Mode coupling coefficients, selection rules |
| `test_lateral_rheology.py` | 24 | Lateral variation processing, conjugate pairs |
| `test_energy.py` | 24 | Strain matrices, stress-strain, dissipation |
| `test_propagator_coupled.py` | 18 | Coupled multi-mode propagator matrices |
| `test_solver.py` | 16 | RK5 integration, BCs, ocean/no-ocean, convergence |
| `test_rheology.py` | 16 | Normalization, Maxwell rheology, elastic/viscous limits |
| `test_analytical.py` | 16 | Analytical formulas, cross-validation, limits |
| `test_types.py` | 13 | Data structures, factory functions, defaults |
| `test_solver_coupled.py` | 13 | Coupled 8N×8N solver, boundary conditions |
| `test_wigner.py` | 12 | Wigner 3j/6j symbols validation |
| `test_love.py` | 12 | Pipeline, Love number extraction, physics |
| `test_grid.py` | 11 | All 4 grid methods, boundary indices, edge cases |
| `test_lateral_e2e.py` | 10 | End-to-end lateral variation workflow |
| `test_bodies.py` | 10 | Body catalog, parameter ranges |
| `test_matlab_validation_ocean.py` | 13 | Weber Moon ocean+lateral vs MATLAB/Qin spectra |
| `test_jax_coupled_ocean.py` | 12 | JAX coupled ocean scan vs NumPy (segments, guards) |
| `test_solver_ocean.py` | 11 | 1D + coupled ocean path (fluid limit, structure) |
| `test_love_coupled.py` | 9 | Coupled multi-mode Love number extraction |
| `test_jax_propagator.py` | 9 | JAX 1D propagator + scan vs NumPy/analytic |
| `test_compat.py` | 9 | PlanetProfile compatibility adapter |
| `test_bc_ocean_coupled.py` | 8 | 24N×24N ocean BC assembler vs 1D reduction |
| `test_benchmark_pyalma3.py` | 8 | PyALMA3 cross-validation (1D models) |
| `test_matlab_validation.py` | 7 | MATLAB reference cross-validation (Enceladus, NumPy path) |
| `test_jax_coupled_scan.py` | 7 | Coupled JAX scan vs NumPy solver (Y, y_sol, Aprop_aux, K path) |
| `test_output_radial.py` | 5 | Radial profile output plots |
| `test_jax_coupled_build.py` | 5 | Coupled JAX 8N×8N assembly vs NumPy reference |
| `test_output_love.py` | 3 | Love number output plots |
| `test_output_convergence.py` | 3 | Grid convergence study plots |
| `test_jax_matlab_validation.py` | 3 | MATLAB reference cross-validation (Enceladus, JAX path) |
| `test_output_reference.py` | 2 | Reference output regression |
| **Total** | **377** | |

### Measured Performance (Milestone 4)

Full methodology and numbers in [`docs/BENCHMARK_jax_coupled.md`](./docs/BENCHMARK_jax_coupled.md) (Apple M4, CPU backend, `scripts/benchmark_jax_coupled.py` to reproduce). Coupled solve, warm JIT cache vs the NumPy reference:

| Coupled modes N | NumPy | JAX (warm) | Speedup |
|---:|---:|---:|---:|
| 4 | 124 ms | 27 ms | 4.5× |
| 12 | 778 ms | 119 ms | 6.5× |
| 38 | 4.97 s | 1.35 s | 3.7× |
| 101 | 33.2 s | 23.8 s | 1.4× |

- The JAX path is the right default for the realistic N≈5–20 regime and for parameter sweeps at fixed model structure (the compiled scan is memoized on the coupling structure; first call pays ~0.75–2 s of XLA compile at small/medium N).
- Near N≈100 both implementations are bound by the dense 8N×8N complex `linalg.solve` inside every Cash-Karp stage, so the speedup converges toward parity and JAX peak memory runs ~3× NumPy (4.4 vs 1.5 GB at N=101). NumPy remains competitive, and lighter, for one-shot very large N solves.
- GPU backends are untested so far; all figures are CPU.

### Future Development

- Ocean energy dissipation in the coupled path (the dissipation integral currently skips the ocean-ceiling node)
- Chunked `vmap` for the auxiliary-matrix build to cap large-N memory
- `jax.vmap` over forcings/frequencies for batched sweeps; GPU backend evaluation
- `jax.grad` through the pipeline for sensitivity analysis (dk/dmu, dk/deta) — gradients through the coupled builder already verified working
- Additional rheologies: Andrade, Burgers, extended Burgers
- ML surrogate models trained on pylov3d for real-time parameter exploration
- Spectral-to-geographic coordinate transforms for map visualization

---

## MATLAB Version

### Requirements

The code runs with MATLAB R2023a. 
Ghostscript is required if the user wants to store plots in pdf format.
The code uses the following third-party libraries: 

- [cmocean](https://github.com/chadagreene/cmocean): Thyng, Kristen, et al. “True Colors of Oceanography: Guidelines for Effective and Accurate Colormap Selection.” Oceanography, vol. 29, no. 3, The Oceanography Society, Sept. 2016, pp. 9–13, doi:10.5670/oceanog.2016.66.  
- [M_Map](www.eoas.ubc.ca/~rich/map.html): Pawlowicz, R., 2020. “M_Map: A mapping package for MATLAB”, version 1.4m, [Computer software], available online at www.eoas.ubc.ca/~rich/map.html.  
- [export_fig](https://github.com/altmany/export_fig/releases/tag/v3.40): Yair Altman (2023). export_fig (https://github.com/altmany/export_fig/releases/tag/v3.40), GitHub. Retrieved November 21, 2023.  
- [harmonicY](https://www.mathworks.com/matlabcentral/fileexchange/74069-wigner-3j-6j-9j): Javier Montalt Tordera (2023). Spherical Harmonics, GitHub. Retrieved November 21, 2023. 
- [Wigner 3j-6j-9j](https://www.mathworks.com/matlabcentral/fileexchange/74069-wigner-3j-6j-9j): Vladimir Sovkov (2023). Wigner 3j-6j-9j, MATLAB Central File Exchange. Retrieved October 4, 2023.  


### Usage

See the manual in `./docs` for information on inputs and outputs of the code.

The `tests/` directory contains several examples

- [**One layer, spherically-symmetric**](./tests/Test_One_Layer_Spherically_Symmetric.mlx): Compares LOV3D Love numbers against love numbers obtained analytically for a uniform spherically-symmetric body
- [**Multi-layered Spherically-symmetric**](./tests/Test_Io_Multi_Layer_Spherically_Symmetric.mlx): Multi-layered Io model based on [Steinke et al. 2020](https://doi.org/10.1016/j.icarus.2019.05.001), consisting of core, deep mantle, asthenosphere and lithosphere. The script obtains the Love numbers and compares them against results obtained with the spherically-symmetric code of  [Rovira-Navarro et al. 2022](https://doi.org/10.1029/2021JE007117). 
- [**Multi-layered Spherically-symmetric_Tidal_Heating**](./tests/Test_Io_Multi_Layer_Spherically_Symmetric_Tidal_Heating.mlx): Same as the previous test but:
	- (1) tidal heating is computed using  [get_energy.m](./src/get_energy.m)
	- (2) the geographical distribution of tidal heating is shown using the [plot_energy_map.m](./src/Plot_Tools/plot_energy_map.m) plotting function. 
	- (3) the y functions are computed and plotted using [get_map.m](../src/get_map.m) and [plot_map.m](./src/Plot_Tools/plot_map.m) 
- [**Multi-layered spherically-symmetric icy moon**](./tests/Test_Europa_Titan_Spherically_Symmetric.mlx): Multi-layered icy moon model. The script computes the Love numbers for a multi-layered Europa and Titan models based on [Beuthe et al. 2013](https://www.sciencedirect.com/science/article/pii/S0019103512004745?casa_token=xg0XfpmaHT4AAAAA:Qau6ppdURvhX_Vgm_NiDZVwEtERNnqcosVviHYGaLIHJLBugG7ZgBEnHNPG921Qc5SZAktQ6kw). 
- [**Enceladus with lateral variations**](./tests/Test_Enceladus_Two_Layers_Lateral_Variations.mlx): 3 layer Enceladus model consisting of a rigid core, ocean and ice-shell with lateral variations. Compares LOV3D Love numbers against love numbers obtained using the perturbation method of [Qin et al.](https://doi.org/10.1093/gji/ggu279) and the FEM model of [Berne et al.](https://doi.org/10.1029/2023GL106656). Reproduces Figure 2 of [Rovira et al. 2024](https://doi.org/10.48550/arXiv.2311.15710)
- [**Europa with lateral variations**](./tests/Test_Europa_Lateral_Variations.mlx): Europa model with lateral variations. The script computes the Love number spectra and the y functions. The script also uses the [plot_y.m](./src/Plot_Tools/plot_y.m) to plot the “y” functions (U,V,W,R,S,T, phi). 
- [**Consistency check tidal heating**](Consistency_test_Energy.m): This script can computes tidal heating using the Love numbers or the direct integration of the product of stress and strain rate and compares the results.  


### Structure 

- ` data/` contains data used in the code, including the coupling coefficients 
- ` docs/` documentation, including a user manual 
- ` licenses/` licenses of current software and some external routines used in the code 
- ` scripts/` some scripts that use the code 
- ` src/` source code 
- `tests/ ` contains several tests an examples 
- `pylov3d/` Python port (see above)


## Documentation 

The theory behind the method is detailed in [Rovira-Navarro et al. 2024](https://doi.org/10.3847/PSJ/ad381f). A user manual can be found in ` Docs/`


## Author (s)

This software have been developed by: 

- **Marc Rovira-Navarro** :  ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0002-9980-5065] Conceptualization, methodology and software  
- **Isamu Matsuyama**: ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0002-2917-8633] Conceptualization   
- **Allard Veenstra** software   
- **Steven Vance** Python port (pylov3d)


## License

The contents in the `docs/` directory together with all `png` files present in this repository are licensed under a **CC-BY 4.0** (see [CC-BY-4.0](licenses/CC-BY-4.0.txt) file). 

The source code, data files and example scripts are licensed under an **Apache License v2.0** (see [Apache-License-v2.0](licenses/Apache-License-v2.0.txt) file).

The following copyright notice is applicable to employees of Technische Universiteit Delft only (**Marc Rovira-Navarro** and **Allard Veenstra**):  

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program “LOV3D”. LOV3D is a  Matlab package for obtaining the tidal response of bodies with lateral variations of interior properties by the Author(s).  
Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.

&copy; 2023, M. Rovira-Navarro, I. Matsuyama, A. Veenstra

The code uses the following third party libraries:

Licenses and copyright statements for [cmocean](https://github.com/chadagreene/cmocean), [export_fig](https://github.com/altmany/export_fig/releases/tag/v3.40), [harmonicY](https://www.mathworks.com/matlabcentral/fileexchange/74069-wigner-3j-6j-9j) and [Wigner 3j-6j-9j]((https://www.mathworks.com/matlabcentral/fileexchange/74069-wigner-3j-6j-9j))  can be found in the [licenses](licenses/) folder.



## References

This software have been used in the following publications
- [Rovira-Navarro, M., Matsuyama, I., Dirkx, D., Berne, A., Calliess, D., Fayolle, S. 2025](https://doi.org/10.1029/2025GL114708) Prospects of Using Tidal Tomography to Constrain Ganymede's Interior, Geophysical Research Letters
- [Rovira-Navarro, M., Matsuyama, I., Berne, A. 2024](https://doi.org/10.3847/PSJ/ad381f). A Spectral Method to Compute the Tides of Laterally-Heterogeneous Bodies. Planetary Science Journal, 5
- Rovira-Navarro, M.,Matsuyama, I. & Berne, A., 2023 Revealing lateral structures in the interiors of planets and moons from tidal observations. AGU Fall Meeting Abstracts (2023).  
- [Rovira-Navarro, M. & Matsuyama, I. 2022](https://ui.adsabs.harvard.edu/abs/2022AGUFM.P45E2514R/abstract)., A Spectral Method to Study the Tides of Laterally Heterogenous Bodies.  AGU Fall Meeting Abstracts.  


## Cite this repository 

If you use this software please cite it as:

- [Rovira-Navarro, M., Matsuyama, I., Berne, A. 2024](https://doi.org/10.3847/PSJ/ad381f). A Spectral Method to Compute the Tides of Laterally-Heterogeneous Bodies. Planetary Science Journal, 5

## Would you like to contribute?

If you have any questions or queries or would like to contribute contact M. Rovira-Navarrro at m.roviranavarro@tudelft.nl

Future developments include: 
- Extend the code to other loadings (e.g., surface loads)
- Benchmark with FEM-viscoelastic code
- pylov3d post-Milestone-4 work: ocean layers in the coupled solver, batched frequency sweeps, GPU backends (see Future Development above)

Found a bug? Report an “Issue” in the issue's tab. 
