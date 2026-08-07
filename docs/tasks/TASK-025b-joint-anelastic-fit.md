# TASK-025b (Machine B): joint (rigidity, viscosity, alpha) Moon fit

## Why this task exists

TASK-025a established, as a forward consistency check, that Andrade rheology
at literature-plausible parameters can close the Moon's 4.6% elastic k2 gap
while simultaneously reproducing the measured monthly Q = 38 ± 4, and that
Maxwell cannot (it implies Q ≈ 0.79, ~9σ away).

Its adversarial review then established the limit of that result, and this
ticket exists to remove it: **every row of the 025a headline table holds
mantle rigidity at its as-built value and attributes the whole k2 gap to
anelastic softening — and that attribution is not unique.** A purely elastic
`mu_scale` in [0.955, 0.96] closes the same gap with zero anelasticity, and
the shipped TASK-019 posterior (`mu_scale` median 0.965, k2 = 0.02419)
already absorbs it into rigidity. Rigidity and anelasticity are degenerate
against k2 alone.

The question this task answers: **does adding Q as a second observable break
that degeneracy?** Q is the quantity anelasticity controls and rigidity does
not, so in principle it should. Nobody has run it.

## Owner and routing

**Machine B.** Compute-heavy, token-light: the deliverable is a converged
posterior plus a short numeric summary, not a code design. Everything needed
is committed. Claim it in the ledger the usual way (edit Owner, commit
`coord: claim TASK-025b`, push immediately) before starting.

## Prerequisites

TASK-025a must be committed first (it is at the user's commit gate as of this
writing — check the ledger row before claiming). Entry points:

- `pylov3d/anelastic_moon.py` — Moon anelastic forwards. Every function takes
  plain scalars and returns a complex Love number, which is the call shape a
  log-likelihood needs (025a was designed for this).
- `pylov3d/moon_mc.py` — the existing 4-parameter elastic Moon MC
  (`core_rho_scale`, `mu_scale`, `R_fluid_core`, `mantle_rho_scale`) and its
  constraint set.
- `scripts/moon_pocomc.py` — the sampler driver you wrote for TASK-019; this
  task is its anelastic successor.
- `docs/MOON_MODEL.md`, "Anelasticity (TASK-025a)" — read the whole section,
  especially "A fixed-mantle-rigidity caveat" and the literature parameter
  ranges for priors.

## What to build

Extend the Moon Monte Carlo to sample rheology jointly with structure:

1. **Free parameters:** the existing four, plus mantle viscosity `eta`
   (log-uniform) and Andrade `alpha` (uniform). Use the prior ranges recorded
   in 025a's "Literature parameter ranges" subsection — do not invent ranges;
   if one is missing, say so and stop rather than guessing.
2. **Observables:** the existing four constraints, plus **Q = 38 ± 4 at the
   draconic month (27.212 d)** as a fifth Gaussian constraint. The forcing
   period matters and was itself a review correction — use the constant in
   `pylov3d.anelastic_moon`, not a hardcoded number.
3. **The Andrade problem.** `pylov3d` implements Maxwell only; 025a's Andrade
   numbers come from PyALMA3 called as an external reference, on a
   *simplified* body (PyALMA3 supports a fluid layer only at the centre and
   is incompressible, so it cannot represent the Weber Moon's internal
   ocean). Decide and **document** which of these you run, with the
   consequence stated:
   - **(a) Maxwell-only joint fit** through native `pylov3d` on the real
     10-layer body. Cheapest and fully validated, but 025a already shows
     Maxwell cannot reach Q ≈ 38 — so the expected outcome is that the Q
     constraint is unsatisfiable and the posterior is driven to the prior
     edge. That is a legitimate, publishable negative result and a good
     smoke test, but it is not the science question.
   - **(b) Andrade joint fit** through PyALMA3 on the simplified body. Answers
     the actual question, but every posterior sample inherits the simplified
     structure and incompressibility. Quantify the structure penalty first
     using 025a's own control (Maxwell on real vs simplified moved Q by
     0.19%) and state whether that carries over to Andrade.
   - **(c) Both**, (a) as validation and (b) as the science run. Preferred if
     the compute budget allows.
4. **Convergence to the TASK-019 standard:** `n_active >= 64`,
   `n_effective >= 128`, dynamic termination, Kish ESS reported. Archive the
   chain as `docs/figures/proposal/moon_anelastic_chain.npz` mirroring
   `moon_posterior_chain.npz`, plus a pairplot.

## The question to answer explicitly

Report, in numbers:

- **Is the rigidity/anelasticity degeneracy broken?** Give the marginal on
  `mu_scale` with and without the Q constraint, and the `mu_scale`-`eta` (and
  `mu_scale`-`alpha`) correlation coefficients. If they remain strongly
  anti-correlated, the degeneracy survives and Q did not break it — say so
  plainly; that is the finding.
- **Where does `mu_scale` land** relative to the elastic TASK-019 median of
  0.965? If the joint fit pulls it toward 1.0, that is the quantitative
  statement that the elastic fit was absorbing anelasticity into rigidity —
  the thing 025a could only assert qualitatively.
- **What alpha does the data prefer**, and is it inside the 0.2-0.3 range
  Efroimsky (2012) gives for silicates? 025a's forward check sat at 0.30-0.35
  by construction (one gap-closing point per alpha); this is the first time
  alpha is actually *inferred* rather than assumed.

## Constraints

- Do not modify `pylov3d/anelastic.py` or `pylov3d/anelastic_moon.py` beyond
  what a driver needs — if you find a bug in them, report it rather than
  patching around it.
- Standard project citation rule applies to anything you write into the docs:
  no verbatim quotation unless you retrieved the source in that session;
  paraphrase or delete otherwise. A prior round of this task family shipped a
  fabricated quote, which is why the rule is explicit.
- Prose standard: do not use the words "genuine" or "honest"; state facts and
  provenance directly.
- Python suite must stay green (`venvLOV3Dconv/bin/python -m pytest
  pylov3d/tests/ -q`).

## Done criteria

Chain + pairplot committed; a "Joint anelastic fit (TASK-025b)" section
appended to `docs/MOON_MODEL.md` answering the three questions above with
numbers; ledger updated to DONE with the headline result; awaiting VERIFIED
by a different driver.
