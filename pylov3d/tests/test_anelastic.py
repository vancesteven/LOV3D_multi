# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.anelastic / pylov3d.anelastic_moon — TASK-025a
viscoelastic tidal response.

Three layers of coverage:

1. **Solver capability audit** — a live grep guarding the module
   docstring's claim that Andrade is not implemented anywhere in this
   Python port (nor the vendored MATLAB source).
2. **PyALMA3 validation** (skipped per-class, not module-wide, when
   ``alma`` is not installed — see ``requires_alma`` below) — Maxwell
   rheology on body-relevant (Mars/Moon) structures at their respective
   tidal forcing periods, quantified tolerance, extending
   ``test_benchmark_pyalma3.py``'s toy-body benchmark to the as-built
   models. Andrade is validated only where a comparison is possible at
   all (the elastic limit, common to both codes); the fully-Andrade
   Moon Q-consistency calculation itself is an external-tool estimate,
   not a pylov3d-vs-PyALMA3 agreement check (pylov3d has no Andrade
   path) — see module docstring in ``pylov3d/anelastic.py``.
3. **Headline science regressions** — the Mars forward sweep and the
   Moon Maxwell/Andrade gap-closing + Q-consistency numbers, pinned so
   future changes to either solver are caught.

Heavier tests (bisections, multi-eta sweeps) are marked ``slow``
(excluded from the default fast lane, see ``pyproject.toml``).

Note on ``alma`` (PyALMA3) availability: a module-scope
``pytest.importorskip("alma")`` was used in an earlier draft of this
file, which -- because pytest treats a ``Skipped`` exception raised
during module execution/collection as skipping the *entire* module --
silently skipped every test in this file on a checkout without the
optional ``alma`` extra installed, including
``TestSolverCapabilityAudit`` (the live grep guard against a future
Andrade implementation landing unnoticed) and ``TestImpliedQ`` (pure
Python, no ``alma`` dependency at all). Fixed by using a
``try/except ImportError`` at import time (which does not abort
collection) plus a per-class ``pytestmark = requires_alma`` applied
only to the classes/tests that actually call into PyALMA3.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from pylov3d import anelastic as A
from pylov3d import anelastic_moon as AM
from pylov3d import mars as mars_mod
from pylov3d import moon as moon_mod

try:
    import alma  # noqa: F401 -- presence check only; nothing here calls it directly
    _HAS_ALMA = True
except ImportError:
    alma = None
    _HAS_ALMA = False

#: Apply as ``pytestmark`` on any class (or as a decorator on any single
#: test) that calls into PyALMA3, so it is skipped -- not collection-
#: aborted -- when the optional ``alma`` extra is not installed. See the
#: module docstring note above for why this replaced a module-scope
#: ``pytest.importorskip("alma")``.
requires_alma = pytest.mark.skipif(not _HAS_ALMA, reason="PyALMA3 ('alma') is not installed")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 1. Solver capability audit
# ---------------------------------------------------------------------------

class TestSolverCapabilityAudit:
    """Guards the module docstring's capability claim with a live check,
    so a future Andrade port (or an accidental regression) is caught
    rather than silently going stale in prose."""

    # Files already known (as of TASK-025a) to mention "andrade" only in
    # prose -- either this module's own capability audit/docstrings, or
    # pre-existing "future work" caveats in pylov3d.mars / pylov3d.moon /
    # pylov3d.moon_mc that this task's docs sections now partially answer.
    # Verified by direct reading (not just this grep) that none of these
    # contain an actual Andrade rheology *implementation*.
    _ANDRADE_PROSE_ONLY_FILES = {
        "anelastic.py", "anelastic_moon.py", "test_anelastic.py",
        "mars.py", "moon.py", "moon_mc.py",
    }

    def test_no_andrade_in_python_port(self):
        result = subprocess.run(
            ["grep", "-rln", "-i", "--include=*.py", "andrade", str(_REPO_ROOT / "pylov3d")],
            capture_output=True, text=True,
        )
        hits = [
            line for line in result.stdout.splitlines()
            if Path(line).name not in self._ANDRADE_PROSE_ONLY_FILES
        ]
        assert hits == [], (
            f"Unexpected 'andrade' references outside the known prose-only "
            f"files {sorted(self._ANDRADE_PROSE_ONLY_FILES)} -- either a new "
            f"Andrade implementation landed (update the module docstring "
            f"capability claim) or a new file needs auditing: {hits}"
        )

    def test_no_andrade_in_vendored_matlab(self):
        src_dir = _REPO_ROOT / "src"
        if not src_dir.is_dir():
            pytest.skip("vendored MATLAB src/ not present in this checkout")
        result = subprocess.run(
            ["grep", "-rl", "-i", "andrade", str(src_dir)],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "", (
            "Vendored MATLAB source now references Andrade -- the "
            "'neither codebase implements it' claim in "
            "pylov3d/anelastic.py's module docstring is stale."
        )

    def test_maxwell_is_implemented(self):
        # compute_complex_rheology exists and produces a non-zero-imaginary
        # muC for a viscoelastic layer (sanity anchor for the audit above).
        from pylov3d.rheology import get_rheology
        from pylov3d.types import make_forcing, make_interior_model

        model = make_interior_model(
            R0_km=[10.0, 100.0], rho0=[3000.0, 3000.0],
            mu0=[0.0, 1e10], eta0=[None, 1e15],
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
        m = get_rheology(model, forcing)
        assert complex(m.muC[1]).imag != 0.0


# ---------------------------------------------------------------------------
# 2. implied_Q
# ---------------------------------------------------------------------------

class TestImpliedQ:

    def test_purely_real_is_infinite(self):
        assert A.implied_Q(complex(0.169, 0.0)) == math.inf

    def test_dissipative_sign_gives_positive_Q(self):
        # Im(k2) < 0 is the project's dissipative-tide convention
        # (test_benchmark_pyalma3.py); Q must come out positive.
        Q = A.implied_Q(complex(0.024, -0.001))
        assert Q == pytest.approx(24.0)
        assert Q > 0

    def test_matches_hand_formula(self):
        k2 = complex(0.169, -0.0078)
        assert A.implied_Q(k2) == pytest.approx(-k2.real / k2.imag)


# ---------------------------------------------------------------------------
# 3. Native pylov3d Maxwell sanity (no PyALMA3 needed)
# ---------------------------------------------------------------------------

class TestNativeMaxwellSanity:
    """At very large mantle viscosity, the Maxwell-viscoelastic model
    must recover the purely-elastic as-built reference values -- these
    are the two numbers already pinned elsewhere in the repo
    (MARS_MU_SCALE's k2=0.169, WEBER_K2_UNIFORM), so this doubles as a
    consistency check between ``pylov3d.anelastic`` and
    ``pylov3d.mars`` / ``pylov3d.moon``."""

    def test_mars_high_eta_recovers_elastic_k2(self):
        k2 = A.mars_maxwell_k2(1e25)
        assert k2.real == pytest.approx(mars_mod.MARS["k2"], abs=1e-6)
        assert abs(k2.imag) < 1e-6

    def test_moon_high_eta_recovers_elastic_k2(self):
        k2 = AM.moon_maxwell_k2(1e25, Nrbase=30)
        assert k2.real == pytest.approx(moon_mod.WEBER_K2_UNIFORM, abs=1e-6)
        assert abs(k2.imag) < 1e-6

    def test_mars_low_eta_increases_dissipation(self):
        # Near the Maxwell peak (omega*tau_M ~ 1), Im(k2) should be
        # substantially non-zero and negative (dissipative).
        k2 = A.mars_maxwell_k2(1e16)
        assert k2.imag < 0
        assert abs(k2.imag) > 1e-3


# ---------------------------------------------------------------------------
# 4. PyALMA3 Maxwell validation — body-relevant structures
# ---------------------------------------------------------------------------

class TestMarsMaxwellPyALMA3Validation:
    """Full as-built 4-layer Mars model (no simplification needed --
    Mars's core is already a simple fluid layer at the center), Maxwell
    mantle, Konopliv-relevant solar-semidiurnal forcing period. Ks
    driven to the incompressible limit on the pylov3d side to match
    PyALMA3's incompressible propagator (see anelastic.py module
    docstring, "second finding"); this isolates the Maxwell
    complex-modulus mechanics pylov3d shares with the elastic path
    (already validated compressibly to ~1e-12 against native MATLAB,
    TASK-014/020).

    Tolerance: 1e-3 relative (matches the pre-existing toy-body
    benchmark's own stated tolerance in test_benchmark_pyalma3.py).
    Measured agreement across the etas below is closer to ~4e-5.
    """

    pytestmark = requires_alma

    @staticmethod
    def _pylov3d_k2(eta):
        from pylov3d.love import get_love
        from pylov3d.types import make_forcing, make_numerics

        model = A.mars_maxwell_incompressible_model()
        model = A._with_eta(model, {1: eta, 2: eta})
        forcing = make_forcing(Td=mars_mod.MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(model, forcing, numerics)
        return complex(love.k[0])

    @pytest.mark.parametrize("eta", [1e15, 1e16, 1e17])
    def test_maxwell_agrees_with_alma(self, eta):
        k2_py = self._pylov3d_k2(eta)
        k2_alma = A.mars_alma_k2(eta, rheology="maxwell")
        assert k2_py.real == pytest.approx(k2_alma.real, rel=1e-3)
        assert k2_py.imag == pytest.approx(k2_alma.imag, rel=1e-3)
        assert k2_py.imag < 0 and k2_alma.imag < 0


class TestMoonMaxwellPyALMA3Validation:
    """Simplified fluid-core + Maxwell-mantle + elastic-crust Moon-relevant
    body (see ``moon_simplified_body`` docstring for why the real
    10-layer structure, which has an internal ocean, cannot be given to
    PyALMA3 directly). Draconic-month forcing period. Incompressible
    limit on both sides, same rationale as the Mars case above.
    """

    pytestmark = requires_alma

    @pytest.mark.parametrize("eta", [1e15, 1e17, 1e19])
    def test_maxwell_agrees_with_alma(self, eta):
        k2_py = AM.moon_simplified_maxwell_k2(eta, incompressible=True)
        k2_alma = AM.moon_alma_k2(eta, rheology="maxwell")
        assert k2_py.real == pytest.approx(k2_alma.real, rel=1e-3)
        assert k2_py.imag == pytest.approx(k2_alma.imag, rel=1e-3)

    @pytest.mark.slow
    def test_maxwell_gap_closing_via_alma_matches_headline_row(self):
        # The Moon headline table's "Maxwell (simplified body, PyALMA3,
        # cross-check)" row was, in an earlier draft, actually pylov3d's
        # own moon_simplified_maxwell_k2() result mislabeled as PyALMA3
        # (the digits matched pylov3d's, not PyALMA3's, and no test
        # called moon_alma_k2 for this fit at all) -- see
        # docs/MOON_MODEL.md, "Anelasticity (TASK-025a)". This test
        # actually calls moon_alma_k2 (PyALMA3) to bisect the same
        # fractional-gap target on the simplified body, so the "fully
        # independent code path" claim in that section is demonstrated,
        # not merely asserted.
        elastic_k2_simplified = AM.moon_simplified_maxwell_k2(1e30).real
        gap_ratio = moon_mod.MOON["k2"] / moon_mod.WEBER_K2_UNIFORM
        target_simplified = elastic_k2_simplified * gap_ratio

        eta_alma, _ = A._bisect_log_decreasing(
            lambda e: AM.moon_alma_k2(e, rheology="maxwell").real,
            1e10, 1e20, target_simplified, tol=1e-6,
        )
        k2_alma = AM.moon_alma_k2(eta_alma, rheology="maxwell")
        Q_alma = A.implied_Q(k2_alma)

        assert k2_alma.imag < 0
        assert 0 < Q_alma < 5.0
        # Same qualitative headline as the pylov3d-only cross-check in
        # TestSimplifiedBodyCrossCheck: catastrophic (Q < 5) dissipation,
        # not a tight numerical pin (PyALMA3's own bisection/ndigits
        # settings are not swept for convergence here).
        eta_real, _, Q_real = AM.fit_moon_maxwell_gap(Nrbase=30)
        assert eta_alma == pytest.approx(eta_real, rel=0.05)
        assert Q_alma == pytest.approx(Q_real, rel=0.1)


# ---------------------------------------------------------------------------
# 5. Andrade — PyALMA3-only sanity (no pylov3d comparison possible)
# ---------------------------------------------------------------------------

class TestAndradeExternalSanity:
    """pylov3d implements no Andrade rheology (see module docstring), so
    no pylov3d-vs-PyALMA3 agreement check is possible for Andrade --
    per the task spec, this validates what PyALMA3's Andrade path does
    on its own terms, anchored where possible to independently-validated
    numbers (the elastic limit)."""

    pytestmark = requires_alma

    def test_andrade_elastic_limit_matches_pylov3d_elastic(self):
        # As eta -> infinity, Andrade's complex modulus -> the real
        # elastic mu (see alma/__init__.py complex_rigidity, code 6:
        # both the 1/(eta*s) and the eta-dependent Andrade term vanish).
        # This ties the Andrade path to pylov3d's own (validated, in the
        # SAME incompressible limit -- see TestMarsMaxwellPyALMA3Validation)
        # elastic Mars k2 through PyALMA3's shared elastic branch. (NOT
        # pylov3d's ordinary compressible elastic k2=0.169 -- PyALMA3 is
        # incompressible throughout, see anelastic.py module docstring,
        # "second finding"; comparing against the compressible value here
        # would just re-measure the compressibility gap, not validate
        # Andrade.)
        #
        # alpha=0.3 is not an arbitrary choice here: the Andrade->elastic
        # convergence as eta->infinity is alpha-dependent (residual decays
        # as (s*eta/mu)^(-alpha), see moon_simplified_elastic_k2_alma's
        # docstring in anelastic_moon.py for the same effect on the Moon
        # side) and is slower at smaller alpha. Verified directly: at
        # eta=1e30 this test's rel=1e-3 assertion passes at alpha=0.3
        # (measured rel. diff ~2e-5) but FAILS at alpha=0.15 (measured
        # rel. diff ~3.9e-3, roughly 4x the tolerance) -- so this test's
        # pass does not, by itself, establish PyALMA3's Andrade branch
        # converges to the elastic limit uniformly across alpha.
        from pylov3d.love import get_love
        from pylov3d.types import make_forcing, make_numerics

        k2_andrade_huge_eta = A.mars_alma_k2(1e30, rheology="andrade", alpha=0.3)

        incompr_model = A.mars_maxwell_incompressible_model()  # eta0 all None => elastic
        forcing = make_forcing(Td=mars_mod.MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(incompr_model, forcing, numerics)
        k2_pylov3d_incompr_elastic = complex(love.k[0])

        assert k2_andrade_huge_eta.real == pytest.approx(
            k2_pylov3d_incompr_elastic.real, rel=1e-3
        )
        assert abs(k2_andrade_huge_eta.imag) < 1e-5

    def test_andrade_dissipative_sign_convention(self):
        k2 = A.mars_alma_k2(1e20, rheology="andrade", alpha=0.3)
        assert k2.imag < 0

    def test_andrade_Q_decreases_with_decreasing_eta(self):
        # More relaxed (lower viscosity) -> more dissipation -> lower Q.
        k2_stiff = A.mars_alma_k2(1e22, rheology="andrade", alpha=0.25)
        k2_soft = A.mars_alma_k2(1e19, rheology="andrade", alpha=0.25)
        assert A.implied_Q(k2_stiff) > A.implied_Q(k2_soft)

    def test_andrade_alpha_changes_response_at_fixed_eta(self):
        # No monotonicity claim asserted here (Q vs. alpha at FIXED eta
        # is not monotonic in one simple direction -- verified empirically
        # while building this test; the physically meaningful, monotonic
        # statement is instead about GAP-CLOSING eta decreasing as alpha
        # increases, exercised by the slow parametrized sweep in
        # TestMoonHeadline). This is a cheap sanity check only: alpha
        # must actually change the complex response at fixed eta (i.e.
        # PyALMA3's Andrade branch is not silently ignoring the
        # parameter).
        eta = 5e20
        k2_lo = A.mars_alma_k2(eta, rheology="andrade", alpha=0.2)
        k2_hi = A.mars_alma_k2(eta, rheology="andrade", alpha=0.35)
        assert k2_lo != k2_hi
        assert abs(k2_lo.imag - k2_hi.imag) > 1e-6


# ---------------------------------------------------------------------------
# 6. Headline science (slow: bisections / multi-eval sweeps)
# ---------------------------------------------------------------------------

class TestMarsForwardSweep:
    """Forward (non-fitting) complex k2 across the literature Andrade
    mantle-viscosity range (Bagheri et al. 2019, reporting Castillo-Rogez
    & Banerdt 2012: 1e19-1e22 Pa s), Maxwell rheology, at Mars's
    solar-semidiurnal forcing period. Pinned regression: at this range,
    Maxwell's Im(k2) is negligible (Q > 1e4) -- the Maxwell relaxation
    time at these viscosities is many orders of magnitude longer than
    the ~12.3 hr forcing period, so a *literature-consistent* Andrade
    viscosity predicts essentially no Maxwell dissipation at all. This
    is the same well-known Maxwell-vs-Andrade mismatch quantified for
    the Moon in TestMoonHeadline below, on Mars's own numbers -- see
    docs/MARS_MODEL.md "Anelasticity (TASK-025a)".
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("eta", [1e19, 1e20, 1e21, 1e22])
    def test_literature_eta_gives_near_elastic_maxwell_response(self, eta):
        k2 = A.mars_maxwell_k2(eta)
        assert k2.real == pytest.approx(mars_mod.MARS["k2"], rel=1e-3)
        assert A.implied_Q(k2) > 1e4

    @pytest.mark.slow
    def test_near_peak_dissipation_eta_gives_low_Q(self):
        # tau_M ~ Td when eta ~ mu*Td; mu_mantle ~ 1e11 Pa,
        # Td = MARS_FORCING_TD ~ 4.4e4 s => eta ~ 4-5e15 Pa s.
        k2 = A.mars_maxwell_k2(5e15)
        assert A.implied_Q(k2) < 15


class TestMoonHeadline:
    """The TASK-025a headline: what mantle viscosity (Maxwell, on the
    real 10-layer structure) or Andrade viscosity+alpha (on the
    simplified body, via PyALMA3) closes the Moon's documented 4.6%
    elastic k2 gap, and whether the implied Q is consistent with
    Williams & Boggs (2015) Q = 38 +/- 4 (:data:`AM.MOON_MONTHLY_Q`,
    :data:`AM.MOON_MONTHLY_Q_SIGMA`). See docs/MOON_MODEL.md,
    "Anelasticity (TASK-025a)" for the full numbers and discussion.
    """

    @pytest.mark.slow
    def test_maxwell_gap_closing_Q_is_grossly_inconsistent(self):
        # Regression-pinned: Maxwell alone cannot simultaneously match
        # the observed k2 gap and a Q anywhere near 38 -- it requires
        # sitting almost exactly at the Maxwell dissipation peak, which
        # forces catastrophic (Q < 5) dissipation. This is the
        # documented, literature-consistent inadequacy of Maxwell for
        # planetary tidal response (e.g. Nimmo et al. 2012).
        eta, k2, Q = AM.fit_moon_maxwell_gap(Nrbase=30)
        assert k2.real == pytest.approx(moon_mod.MOON["k2"], abs=1e-4)
        # Pin the physical (dissipative) sign convention explicitly: a
        # sign-inverted (energy-generating) complex modulus would give
        # Q = -0.79, which the old bare `Q < 5.0` check below could not
        # distinguish from the physical Q = +0.79 (a reviewer-caught gap
        # -- see docs/MOON_MODEL.md, "Anelasticity (TASK-025a)").
        assert k2.imag < 0
        assert 0 < Q < 5.0
        n_sigma = abs(Q - AM.MOON_MONTHLY_Q) / AM.MOON_MONTHLY_Q_SIGMA
        assert n_sigma > 5.0

    @requires_alma
    @pytest.mark.slow
    @pytest.mark.parametrize("alpha,q_lo,q_hi", [(0.3, 30.0, 60.0), (0.35, 25.0, 50.0)])
    def test_andrade_gap_closing_Q_in_literature_alpha_range(self, alpha, q_lo, q_hi):
        # Andrade alpha in the commonly-adopted range
        # (A.ANDRADE_ALPHA_COMMON_RANGE-adjacent; 0.35 sits just outside
        # it but is the value MOON_MODEL.md's existing literature review
        # already highlights) gives an implied Q broadly consistent with
        # Williams & Boggs (2015) 38 +/- 4 -- unlike the Maxwell case
        # above. Bounds are generous (not a tight pin) since PyALMA3's
        # bisection tolerance and Nrbase are not swept for convergence
        # here; the qualitative contrast with Maxwell is the point.
        eta, k2, Q = AM.fit_moon_andrade_gap(alpha)
        assert q_lo < Q < q_hi

    @requires_alma
    @pytest.mark.slow
    def test_andrade_alpha_0p35_within_a_few_sigma_of_measured_Q(self):
        eta, k2, Q = AM.fit_moon_andrade_gap(0.35)
        n_sigma = abs(Q - AM.MOON_MONTHLY_Q) / AM.MOON_MONTHLY_Q_SIGMA
        assert n_sigma < 3.0


class TestSimplifiedBodyCrossCheck:
    """The simplified Moon body (used for the PyALMA3 comparisons above,
    since PyALMA3 cannot represent the real internal ocean) should give
    essentially the same Maxwell gap-closing viscosity and implied Q as
    the real 10-layer structure -- confirming the simplification does
    not distort the rheological physics being tested."""

    @pytest.mark.slow
    def test_simplified_and_real_body_maxwell_gap_agree(self):
        eta_real, k2_real, Q_real = AM.fit_moon_maxwell_gap(Nrbase=30)

        # Same fractional gap on the simplified body (its own elastic
        # baseline differs slightly from the real structure's).
        elastic_k2_simplified = AM.moon_simplified_maxwell_k2(1e30).real
        gap_ratio = moon_mod.MOON["k2"] / moon_mod.WEBER_K2_UNIFORM
        target_simplified = elastic_k2_simplified * gap_ratio

        eta_simp, _ = A._bisect_log_decreasing(
            lambda e: AM.moon_simplified_maxwell_k2(e).real,
            1e10, 1e20, target_simplified, tol=1e-6,
        )
        k2_simp = AM.moon_simplified_maxwell_k2(eta_simp)
        Q_simp = A.implied_Q(k2_simp)

        # Pin the dissipative sign on both sides (see the same-purpose
        # assertions in TestMoonHeadline above -- a sign-inverted complex
        # modulus would otherwise still pass the bare rel-tolerance
        # comparisons below, since Q_real and Q_simp would invert
        # together).
        assert k2_real.imag < 0 and k2_simp.imag < 0
        assert 0 < Q_real < 5.0 and 0 < Q_simp < 5.0
        assert eta_simp == pytest.approx(eta_real, rel=0.05)
        assert Q_simp == pytest.approx(Q_real, rel=0.1)
