# Lateral bulk-modulus (`K`) validation

## Why this is a separate science task

The pylov3d conversion exposes `K_variable` in
`process_lateral_variations()`, and the coupled propagator accepts per-mode
`K_amp` coefficients. However, the current rheology processor sets those
coefficients to zero. Inspection of the MATLAB parent shows that this is not
simply a Python-port regression: the parent `get_rheology.m` also parses
`K_variable` but does not propagate non-zero bulk-modulus coefficients into
`rheology_variable(:,3)`.

A naive MATLAB/Python comparison would therefore risk validating a shared
zero-response bug. Lateral `K` must first be validated from the constitutive
physics, then checked against a repaired/reference MATLAB calculation.

## Constitutive normalization

For isotropic linear viscoelasticity,

\[
\sigma_{ij} = \lambda\,\epsilon_{kk}\delta_{ij} + 2\mu\epsilon_{ij},
\]

with bulk modulus

\[
K = \lambda + \frac{2}{3}\mu.
\]

The scalar (trace) stress contribution therefore carries

\[
3\lambda + 2\mu = 3K.
\]

The single-mode propagator uses this combination explicitly. In the coupled
propagator, the lateral scalar-rheology coefficient enters the corresponding
trace-stress row as `K_nm`. Consequently, the central normalization question
that must be settled before enabling `K_variable` is whether `K_nm` is defined
as the spherical-harmonic coefficient of `3K`, or whether a factor of three is
contained in the angular coupling coefficient convention.

The current source comments are internally ambiguous:

* `get_rheology.m` describes `K_variable(:,3)` as a fractional coefficient
  `K_l^m/K_0^0`;
* `get_solution.m` reads `rheology_variable(:,3)` directly as `K_nm` in the
  trace-stress term;
* the existing rheology code does not populate that column, so there is no
  working implementation from which to infer the convention empirically.

This factor must be derived from the tensor-harmonic equations or recovered
from the derivation in Rovira-Navarro et al. before a production implementation
is accepted.

## Required validation sequence

### 1. Constitutive unit test

Construct a minimal coupled-mode problem and test the `A1/A2` matrices directly.
For a prescribed lateral scalar modulus coefficient, compare the coupled
matrix element against the tensor-harmonic constitutive expression. This test
must resolve the factor-of-three convention independently of MATLAB.

Acceptance criterion: numerical equality at machine precision for selected
low-degree mode couplings.

### 2. Zero-amplitude recovery

With `K_variable=0`, the coupled propagator must reduce exactly to the current
uniform/lateral-mu result.

Acceptance criterion: Love numbers unchanged to floating-point precision.

### 3. Linear-amplitude scaling

For sufficiently small lateral `K` amplitudes, non-forcing Love-number modes
must scale linearly with the imposed amplitude and reverse sign when the
amplitude changes sign.

Acceptance criterion: first-order response ratios agree with amplitude ratios
to better than 1e-4 for amplitudes chosen inside the perturbative regime.

### 4. Independent finite-difference check

Use a spatially varying compressibility field in a direct or high-resolution
reference calculation, if available, or compare the coupled constitutive
operator against finite differences of the uniform operator projected onto
spherical harmonics.

This step is more important than parent-code parity because the MATLAB parent
currently shares the missing `K` path.

### 5. Repaired MATLAB parity

After the normalization is fixed, repair the minimal MATLAB path and generate
an archived reference for one simple elastic model, preferably the existing
Enceladus benchmark geometry with a single degree-2 zonal `K` perturbation.
Keep `mu_variable=0` so the physical source of coupling is unambiguous.

Archive:

* model parameters;
* lateral harmonic degree/order and amplitude;
* complete coupled `k`, `h`, `l` spectra;
* MATLAB commit/source hash;
* pylov3d commit hash;
* numerical resolution and perturbation order.

### 6. Add to the publication science matrix

Only after steps 1-5 pass should lateral `K` be promoted into
`run_science_benchmarks.py` as a claimed parent-code validation regime.

## Current scientific status

Lateral shear-modulus heterogeneity is strongly validated. Lateral bulk-modulus
heterogeneity is **not currently validated and should not be advertised as a
working scientific capability**. The API surface and coupled propagator contain
most of the intended machinery, but the rheology-to-coupling path is incomplete
in both pylov3d and the MATLAB parent.

That distinction is important for a methods paper: this is a known,
well-localized feature gap rather than evidence against the validated lateral
shear-modulus calculations.