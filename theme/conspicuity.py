"""Conspicuity: how loud a find highlight is, in the observer's own steps, and the
baseline every highlight owes the page.

A highlight is found by visual SEARCH, not by side-by-side discrimination. The observer
model measures discrimination -- can two patches be told apart, adjacent, at 104 px -- and
that is the right unit to state a search floor in, but the floor itself sits many steps
above one: a target has to win against the whole page at a glance (Duncan & Humphreys'
target-distractor similarity; Nagy & Sanchez on colour differences large enough for
parallel search). The instrument had let a highlight through at 1.5 discrimination steps
from the page, so every few trials showed a tint nobody could call a highlight, and the
duels already said so: with the fainter side under 3 steps the louder highlight won 9 of 9
day duels and 4 of 4 at night. That is not a preference axis to keep exploring; it is a
region of the space that is not a highlight at all.

Two decisions live here, and only here, so the hunt arm and the theme space cannot drift
apart on either:

1. **The metric.** Distance from the page in the observer's steps, along the direction the
   fill actually took. CAM16-UCS is close to uniform for the average eye, but the fitted
   observer is not average: the confusion-axis ellipse (phi, w1, w2) says a red-green step
   needs about 1.6x the dE of a lightness step before it is seen, and the threshold grows
   with page lightness (gL). Raw dE would score a red fill and a blue fill of equal dE as
   equally findable; his own thresholds say they are not. The hue stays entirely free -- a
   floor stated in steps forces no hue -- which is what "not boxed into the classical signal
   colours" needs and what a hue list could never give.
2. **The baseline.** The current match owes the page `CURRENT_BASELINE_JND` steps; the other
   matches owe it the same multiple the meaning-carrying roles owe each other, through
   `thresholds.separation_floor`, so the two regimes (constant now, fitted size exponent
   later) switch together. Both are constants and are said to be: `find_time_knee` fits
   where the hunt data stops rewarding loudness, so the constant can be replaced by a
   measurement once that knee identifies, and not before.
"""

from typing import NamedTuple

import numpy as np

from .color import hex_to_rgb, rgb_to_ucs
from .observer import REFERENCE_GROUNDS
from .thresholds import DE_MIN, VISION_FIT, separation_floor

#: How many of the observer's discrimination steps the CURRENT match must sit from the
#: page, whatever direction the fill took.
#:
#: 4, from the logged hunts. Over the first 33 usable trials, salience correlated with log
#: find time at -0.43 (day) and -0.37 (night), and a median split gave 3489 ms against
#: 2066 ms by day, 2897 against 2225 by night: a faint highlight costs over a second, which
#: measures patience rather than the theme. 4x excluded roughly the slowest quarter of what
#: had been shown. The same number now floors the whole space rather than only the timed
#: hunt, because the duels rejected fainter fills outright (module docstring).
#:
#: A constant standing in for `find_time_knee`, which fits the same quantity from the hunt
#: log; promote it to the fitted value when the knee identifies, and not before -- a flat
#: fit's midpoint is worse than a judged constant.
CURRENT_BASELINE_JND = 4.0

#: Fewer hunts than this and there is no knee to fit.
MIN_HUNTS_FOR_KNEE = 12


class Ellipse(NamedTuple):
    """The observer's discrimination ellipse in a'b', and how threshold moves with page
    lightness. The isotropic ellipse is the fallback when no vision log exists."""

    phi: float  # confusion-axis angle, radians
    w1: float  # weight along phi (d = de * sqrt(w) along that axis)
    w2: float  # weight across phi
    lightness_gain: float  # gL: log-threshold per unit of J'/100


ISOTROPIC = Ellipse(phi=0.0, w1=1.0, w2=1.0, lightness_gain=0.0)


def _fitted_ellipse(fit=VISION_FIT):
    if fit is None:
        return ISOTROPIC
    return Ellipse(
        phi=float(np.deg2rad(fit.phi_deg_mean)),
        w1=float(fit.w1_mean),
        w2=float(fit.w2_mean),
        lightness_gain=float(fit.gL_mean),
    )


ELLIPSE = _fitted_ellipse()

#: J'/100 of the page each polarity's DE_MIN is stated on; the lightness scaling starts here.
REFERENCE_J = {
    polarity: float(rgb_to_ucs(hex_to_rgb([ground]))[0, 0] / 100.0) for polarity, ground in REFERENCE_GROUNDS.items()
}


def observer_jnd(delta_ucs, ground_j, polarity, ellipse=None):
    """A CAM16-UCS difference as a count of the observer's discrimination steps.

    `delta_ucs` is (..., 3) rows of (dJ', da', db'); `ground_j` is the page's J'/100,
    scalar or one per row. The weighted length is the observer model's own `d` (the
    quantity its psychometric function is fit on), and one step is the polarity's DE_MIN
    scaled to this page's lightness by exp(gL * (J' - J'_reference)) -- posterior means
    used pointwise, the same monotone-faithful stand-in `observer.discriminability` makes.
    """
    ellipse = ellipse or ELLIPSE
    delta = np.asarray(delta_ucs, dtype=float)
    along = np.cos(ellipse.phi) * delta[..., 1] + np.sin(ellipse.phi) * delta[..., 2]
    across = -np.sin(ellipse.phi) * delta[..., 1] + np.cos(ellipse.phi) * delta[..., 2]
    weighted = np.sqrt(delta[..., 0] ** 2 + ellipse.w1 * along**2 + ellipse.w2 * across**2)
    step = DE_MIN[polarity] * np.exp(
        ellipse.lightness_gain * (np.asarray(ground_j, dtype=float) - REFERENCE_J[polarity])
    )
    return weighted / step


def other_baseline_jnd(polarity):
    """What the OTHER matches owe the page: the meaning roles' separation multiple, so a
    quiet match is still a readable mark at glyph size and switches regime with them."""
    return separation_floor(polarity)[0] / DE_MIN[polarity]


def highlight_baseline(polarity):
    """(current steps, other steps, how they were arrived at) -- the readout's form."""
    other = other_baseline_jnd(polarity)
    return (
        CURRENT_BASELINE_JND,
        other,
        f"current match {CURRENT_BASELINE_JND:g} observer steps from the page (the measured search "
        f"floor, a constant until the hunt knee identifies); other matches {other:.2g} steps, "
        f"{separation_floor(polarity)[1]}",
    )


def baselines_hold(current_jnd, other_jnd, polarity):
    """Does a highlight pair clear what it owes the page?

    Written as "require the good case": a NaN from an out-of-range channel compares False
    with everything, so `not (x < floor)` would pass it and `x >= floor` refuses it.
    """
    return bool(current_jnd >= CURRENT_BASELINE_JND) and bool(other_jnd >= other_baseline_jnd(polarity))


def conspicuity_of(theme, polarity):
    """(current steps, other steps) from a theme's own hexes -- what a reader with only the
    logged theme can recompute, and what the tests compare the published field against."""
    ucs = rgb_to_ucs(hex_to_rgb([theme["ground"], theme["find_current"], theme["find_other"]]))
    steps = observer_jnd(ucs[1:] - ucs[0], ucs[0, 0] / 100.0, polarity)
    return float(steps[0]), float(steps[1])


class FindTimeKnee(NamedTuple):
    """Where loudness stops buying find time, fitted from the hunts.

    A hinge on log time: flat above the knee, rising below it at `slope_per_step`. `gain`
    is the share of log-time variance the hinge explains over a flat line; near zero means
    the data has not located a knee and the constant should stand.
    """

    knee_jnd: float
    slope_per_step: float  # log-time per step BELOW the knee; negative = fainter is slower
    gain: float
    n: int


def find_time_knee(steps, rt_ms):
    """Fit the hinge over the observed range, or None below MIN_HUNTS_FOR_KNEE hunts.

    Grid search over the knee with a closed-form line at each candidate: with a few dozen
    hunts the likelihood in the knee is far from quadratic, so a grid is both cheaper and
    more honest than a local optimiser. The knee is confined to the interior so a fit
    cannot report an edge it has no data beyond.
    """
    steps = np.asarray(steps, dtype=float)
    log_rt = np.log(np.asarray(rt_ms, dtype=float))
    if len(steps) < MIN_HUNTS_FOR_KNEE:
        return None
    flat_sse = float(((log_rt - log_rt.mean()) ** 2).sum())
    low, high = np.percentile(steps, [15, 85])
    best = None
    for knee in np.linspace(low, high, 41):
        shortfall = np.maximum(0.0, knee - steps)
        design = np.column_stack([np.ones_like(steps), shortfall])
        coefficients, *_ = np.linalg.lstsq(design, log_rt, rcond=None)
        sse = float(((log_rt - design @ coefficients) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(knee), float(coefficients[1]))
    sse, knee, slope = best
    return FindTimeKnee(knee_jnd=knee, slope_per_step=-slope, gain=1.0 - sse / flat_sse, n=len(steps))
