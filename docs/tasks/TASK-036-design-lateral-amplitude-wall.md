# TASK-036 (design + A): the lateral-amplitude wall is geometric, not a solver limit

## The diagnosis, which reframes the problem

Two independent results ran into the same bound:

- **TASK-028 (Mars):** the InSight-derived crustal fields reach
  `max|dmu/mu_bar| = 0.9689` against a hard bound of 1.0, so the density /
  calibration axis could not be explored.
- **TASK-031b (Moon):** `max|dmu/mu_bar|` is 0.9902 at `lmax=4` and crosses
  unity at `lmax=5`, so angular convergence cannot be demonstrated at all —
  the field goes non-physical before it converges.

Both were described as a limit of "the linearized rigidity mapping". That
description is misleading, and the correction matters because it changes
what would fix it.

**The map is not an approximation of the solver.** For a model shell of fixed
thickness `T` whose material is crust where the real crust reaches, and
mantle below it,

```
delta_mu / mu_bar  =  dt * (mu_c - mu_m) / (T * mu_c)
```

is the *exact* volume-fraction (Voigt) average of the two materials inside
that shell. It is linear in `dt` because volume fraction is linear in `dt`,
not because anything was expanded to first order. Raising the solver's
`perturbation_order` therefore cannot help: the solver is not where the
approximation lives.

**What actually breaks is the shell geometry.** The expression exceeds unity
when the thickness excursion is large enough, relative to the shell, that
the mixing rule would demand a negative fraction of one material. Measured
for the Moon (`CRUST_THICKNESS_M` = 40 km, contrast
`|mu_c - mu_m| / mu_c` = 1.2139):

| | max\|dt\| | max\|dmu/mu_bar\| |
|---|---:|---:|
| lmax=4 | 32.63 km | 0.9902 |
| lmax=5 | 37.99 km | 1.1531 |
| lmax=6 | 42.50 km | 1.2897 |

Unity is reached at `|dt|` = 32.95 km. Note that this is **less than the
40 km shell**: the shell is not even full. The binding factor is the
rigidity *contrast* of 1.2139, which amplifies a thickness fraction of 0.82
into a rigidity fraction of 0.99.

That is the whole wall, on both bodies, and it is a modelling choice rather
than a physical or numerical necessity.

## Two candidate fixes, both cheap to test

**(A) Thicken the reference shell.** The bound scales as `1/T`. Bringing the
Moon's `lmax=6` field to a comfortable 0.8 margin needs `T >= 64.5 km`
against the present 40 km. This is a pure model-resolution change — the
crust/mantle boundary still sits where it sits; the shell that brackets it
is simply wider. Cost: the contrast is smeared over a larger radial range,
so the *background* tidal response shifts and the 1-D fit must be checked
(and possibly re-fit) rather than assumed. **Test: does the Love spectrum
converge as `T` grows at fixed total contrast?** If it does, the present
40 km shell is simply too thin to carry this field and the answer is to
widen it.

**(B) Change the mixing rule.** Voigt (arithmetic, iso-strain) is the
stiffest of the standard bounds and is what makes the expression go
negative. Reuss (harmonic, iso-stress) stays positive for every fraction in
[0, 1] by construction. For shear in a layered shell under tidal flexure
neither bound is exactly correct, so this is a genuine physical modelling
choice and not a trick to dodge the bound — which is precisely why it needs
arguing rather than adopting. **Do not ship (B) without stating which
loading geometry justifies it**, and note it changes results at *every*
amplitude, not only near the bound, so it would perturb already-published
numbers.

Preference: try **(A) first**, because it changes resolution rather than
physics and is falsifiable by a convergence test. Treat (B) as the fallback
if (A) shows the spectrum is still moving when `T` gets large enough to
matter.

## Why this is worth doing

It is currently the binding constraint on the whole lateral-amplitude
programme, on both bodies:

- The Moon lateral result (TASK-031) is stuck at "highest cutoff the
  linearization admits", not a converged number.
- The Mars crustal-model sensitivity (TASK-028) could not explore the
  density axis, which is the dominant remaining uncertainty there — and the
  reason given, `|dmu/mu_bar| = 0.9689`, is this same bound.

Both would be unblocked by the same change, if (A) works.

## Scope for a first pass

1. Reproduce the bound analytically from `_dmu_ddt_coeff` and confirm the
   table above, so the diagnosis is pinned by something other than this note.
2. For the Moon, sweep `T` (say 40, 55, 70, 85 km) at fixed crust/mantle
   contrast, and report the `lmax=4` Love spectrum at each. **The question is
   whether the spectrum converges in `T`.** If it does not, (A) is not a fix
   and this needs the harder conversation.
3. Only if (2) converges: rerun the TASK-031b ladder at the chosen `T` and
   report whether angular convergence is now reachable.
4. Do the same diagnostic arithmetic for Mars and state what `T` would be
   needed there.

Do not change any shipped default as part of this task. The deliverable is a
diagnosis and a recommendation, not a migration.

## Constraints

- Driver scripts and a design note; do not change `moon_lateral.py`,
  `mars_lateral.py`, or any solver module in this pass.
- If (A) works, the follow-on migration is a separate ticket, because it
  moves published numbers.
- Prose standard: never "genuine" or "honest".
