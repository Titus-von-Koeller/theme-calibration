# Reproduction record: the five failures from the 2026-09-06 sitting

The gate this branch is measured against. Written before any fix, so the fix has something
to be checked against rather than a memory of what was red.

## Why a record file rather than a script

The failures exist only against response rows that are UNCOMMITTED in Titus's main checkout;
committing them is his decision, not this branch's. So no script in this repository can
reproduce them from tracked state, and a script that silently passes when the rows are absent
would be worse than none. What is reproducible is the procedure plus the exact inputs, which
is what this file pins: the snapshot digests make the observation checkable later even after
his working tree has moved on.

## Procedure

The five measurement files named in `theme/paths.py` were copied out of the main checkout into
a scratch directory and read through the `THEME_DATA` seam. Nothing was written to
`~/src/theme-calibration`; the derived `observer-fit.json` and the `observer-logp-*` cache pair
were deliberately NOT copied, so the observer refits from the copied rows.

```
THEME_DATA=<scratch>/data OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  pixi run --manifest-path <this worktree>/pixi.toml \
  python -m pytest -o addopts="" -q tests/test_conspicuity.py tests/test_realization.py
```

## Inputs, digested at the moment of the snapshot

Snapshot taken 2026-09-08 15:05:47 CEST. The main checkout's logs had been modified at 15:02,
i.e. minutes earlier — the rows are live, which is precisely why this branch reads a copy.

| file | md5 |
| --- | --- |
| aesthetics-responses.jsonl | db96e7bab35fcbd2392c86becd12861e |
| applied-themes.jsonl | 2f9fb0a379e7c8ae19a7ec044fca4e6d |
| calibration-responses.jsonl | 00cc6e90ac02c4acc3c6905f10250141 |
| lived-responses.jsonl | d41d8cd98f00b204e9800998ecf8427e (empty) |
| measured-theme.json | copied unchanged |

## Result: 5 failed, 20 passed, 15.42s

Exactly the five named in the brief, and nothing else in these two files.

```
FAILED tests/test_conspicuity.py::test_a_chromatic_step_counts_for_fewer_jnd_where_the_ellipse_is_weak[day]
FAILED tests/test_conspicuity.py::test_a_chromatic_step_counts_for_fewer_jnd_where_the_ellipse_is_weak[night]
FAILED tests/test_conspicuity.py::test_the_other_matches_owe_the_meaning_roles_multiple[day]
FAILED tests/test_conspicuity.py::test_the_other_matches_owe_the_meaning_roles_multiple[night]
FAILED tests/test_realization.py::TestSeparationFloorRegime::test_the_constant_is_in_force_while_no_trial_varied_size
```

## The state of the fit under those inputs

Measured through the same seam, so these numbers explain the five lines above.

```
ELLIPSE  phi=-0.0756  w1=0.4241  w2=1.0350  lightness_gain=0.1972
size_is_identified() -> True
gamma marginal   values [0.0, 0.35, 0.7, 1.05, 1.4]
                 p      [0.0211, 0.9779, 0.00103, 2.83e-08, 1.92e-13]
                 spread max-min = 0.9779   (the function's bar is 0.03)
gamma_mean = 0.34298
separation_floor day   6.5619 dE  = 1.9893 x DE_MIN (3.2985)
separation_floor night 5.5837 dE  = 1.9893 x DE_MIN (2.8068)
```

## Two causes, not one — and not the 4+1 the brief assumed

**Cause A, the ellipse refit (2 failures).** `w2 = 1.0350` is above 1 by 3.5%, so a chromatic
step along that axis now counts for marginally MORE observer steps than a lightness step of the
same dE. `test_a_chromatic_step_counts_for_fewer_jnd_where_the_ellipse_is_weak` asserts
`red_green <= lightness + 1e-9` from a docstring premise that "the fitted ellipse has w1 and w2
below 1". That premise is a property of a live fit, asserted to nine decimal places.

**Cause B, the regime switch (3 failures).** The other three all follow from
`size_is_identified()` turning True. `other_baseline_jnd` IS
`separation_floor(polarity)[0] / DE_MIN[polarity]`, so the two
`test_the_other_matches_owe_the_meaning_roles_multiple` failures and the
`TestSeparationFloorRegime` failure are one cause wearing three names — which is exactly what
that test's own docstring predicted: "the point of routing it through separation_floor is that
both switch regime together."

## What the switch actually says, which is the good news

The new glyph-size trials MEASURED the size exponent, and it came out where the constant had
been standing in for it. 97.8% of gamma's posterior mass sits on the grid point 0.35;
`IMPLIED_EXPONENT` was 0.35, and the fitted floor is 1.9893x DE_MIN against the retired
constant's 2.0x — a difference of 0.5%. So the transition was continuous, precisely as
`test_the_floor_switches_to_the_fitted_exponent_once_size_is_measured` said it would be. The
constant was not wrong; it is now measured instead of assumed.

Precision, stated honestly: the gamma grid is `[0, 0.35, 0.7, 1.05, 1.4]` — five points spaced
0.35 apart, and 0.35 is itself a grid point. The exponent is identified well enough to exclude
0.7 and above, not well enough to claim more than two digits.

## Reported, not changed (see the queue item)

`size_is_identified()` reaches the posterior through a private attribute. Its documented path,
`fit.summary()["marginals"]["gamma"]`, returns None here because `summary()["marginals"]` is an
EMPTY dict; the value is found only by the `or` fallback to `fit._p["marginals"]["gamma"]`.
`theme/` is outside this branch's partition, so this is reported rather than touched.
