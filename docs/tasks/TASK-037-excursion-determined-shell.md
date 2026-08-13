# TASK-037: the reference shell is not free — it is set by the excursion

## What TASK-036b established, and the one thing it left open

B's T-sweep answered the question TASK-036 asked and answered it in the
negative: **the spectrum does not converge in `T`.** `|Delta k20|` scales as
`1/T`, so widening the shell chases the amplitude toward zero rather than
revealing a stable limit. Fix (A) is therefore not the resolution-only change
the design note hoped for — it moves published numbers by `T_old/T_new`.

Two observations sharpen that, and together they close the question B left
open ("the reference shell choice is a physics argument, not a numerical
one").

**First, the `1/T` scaling was algebraically inevitable.** Since
`delta_mu/mu_bar = dt (mu_c - mu_m) / (T mu_c)` and the forcing-mode shift is
linear in the perturbation to first order, `Delta k20 ∝ 1/T` identically. The
sweep confirmed it, but no sweep could have found otherwise. That is worth
recording so nobody re-runs it expecting a different answer.

**Second, and this is the point: `T` is not free.** The shell exists to
contain the crust-mantle boundary wherever it sits. A shell that does not
span the boundary's full excursion mis-assigns material; one much wider than
the excursion dilutes the anomaly over inert thickness. That bounds `T` from
below at roughly the excursion range, and the natural choice is the smallest
shell that contains it.

Take `T = 2 max|dt|`. Then

```
|delta_mu / mu_bar|  =  dt K / T  =  K / 2       (K = |mu_c - mu_m| / mu_c)
```

**identically, at every truncation, with no free parameter.** For the Moon
`K/2 = 0.607`; for Mars `0.626`. Both comfortably inside the positivity
bound, and the bound is now a property of the *material contrast alone* —
checkable before any solve, and satisfied for any `K < 2`.

Note what this says about TASK-036b's own result: B's `T = 85` km at
`lmax = 6` gives `|delta_mu/mu_bar| = 0.6069`, and `2 max|dt|` at `lmax = 6`
is `84.99` km. **B's converged configuration was the excursion-determined
shell**, reached by choosing a round number. That is encouraging — the rule
picks out the configuration that already worked — but it was not arrived at
by the rule, so it has not been tested as one.

## Correction from the committed T-sweep artifact (A, 2026-08-13)

The premise above is weaker than first stated, and the ticket is kept
because of it rather than in spite of it.

`|delta_mu/mu_bar|` is exactly `1/T` — verified to four decimals. But
`|Delta k20|` is **not**: it follows `T^-1.338`, because it mixes a
first-order term (`~1/T`) with a second-order one (`~1/T^2`). Fitting
`a/T + b/T^2` reproduces all four sweep rungs to better than 0.2%.

So fixing `T = 2 max|dt|` fixes the *perturbation* at `K/2` but does
**not** fix the *response*. The excursion rule removes the arbitrary
parameter from the perturbation amplitude; whether it yields a
convergent `Delta k20` is genuinely open, and the test below is
therefore a real test rather than a formality.

The off-forcing modes *are* pure first order (`T^-0.996`), so for those
the rule does pin the amplitude. If the ladder converges for the
off-modes but not the forcing mode, that is a meaningful and reportable
asymmetry — and it would sharpen the existing conclusion that the
spatial information lives in the off-forcing spectrum.

## What to test

The rule ties `T` to the field, so `T` now varies with truncation
(`65.3 / 76.0 / 85.0` km at `lmax = 4 / 5 / 6` for the Moon). The open
question is whether `Delta k20` converges in `lmax` **under the rule**, which
is a different ladder from B's fixed-`T` one.

1. Run the Moon ladder with `T = 2 max|dt|(lmax)` recomputed at each rung.
   Report `Delta k20` and the tracked off-modes, and whether the sequence
   settles. **This is the deciding test**: if it converges, the lateral
   amplitude has a well-defined value with no free parameter; if it does not,
   the shell-averaging formulation genuinely cannot deliver one and that
   should be stated as a limit of the method rather than worked around.
2. Do the same for Mars, where `K/2 = 0.626`.
3. Sanity-check the rule's premise rather than assuming it: confirm that
   `2 max|dt|` really does span the boundary excursion for these fields.
   `max|dt|` is a maximum of the absolute value, so if `dt` is strongly
   asymmetric the true range is `dt_max - dt_min`, which may differ. **Use
   the actual range if it differs materially, and say which you used.**

## Caveats to carry

- Widening `T` changes the background 1-D model, so the reference fit must be
  re-checked at each `T`, not assumed. TASK-036b already flagged this.
- The rule is a modelling argument, not a derivation from the tidal equations.
  It says the shell should contain what it is averaging over. That is
  defensible and it removes an arbitrary parameter, but it should be
  presented as a stated convention with its justification, not as the unique
  correct treatment.
- Nothing here rescues the *absolute* amplitude if the answer is that it
  depends on the averaging scheme. It may simply be that a fixed-shell Voigt
  average cannot define a lateral amplitude, in which case the honest output
  is that limitation, clearly stated.

## Constraints

- Driver scripts only; change no shipped default in this task. Migration
  remains a separate ticket, because it moves published numbers.
- Prose standard: never "genuine" or "honest".
