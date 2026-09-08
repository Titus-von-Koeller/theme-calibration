"""The conspicuity metric and the highlight baseline.

The metric has to be the observer's own: one step along lightness at the reference page IS
one DE_MIN, a chromatic step costs more dE where the fitted ellipse is weak, and a lighter
page raises the step. The baseline has to fail closed. The knee fit has to recover a knee
that was planted -- the recovery-test shape the rest of the suite uses.
"""

import inspect

import numpy as np
import pytest

from theme import conspicuity, thresholds
from theme.conspicuity import (
    CURRENT_BASELINE_JND,
    ISOTROPIC,
    Ellipse,
    baselines_hold,
    find_time_knee,
    observer_jnd,
    other_baseline_jnd,
)
from theme.thresholds import DE_MIN

polarities = ["day", "night"]


@pytest.mark.parametrize("polarity", polarities)
def test_one_lightness_step_at_the_reference_page_is_exactly_the_threshold(polarity):
    delta = np.array([DE_MIN[polarity], 0.0, 0.0])
    assert observer_jnd(delta, conspicuity.REFERENCE_J[polarity], polarity) == pytest.approx(1.0)


@pytest.mark.parametrize("polarity", polarities)
def test_a_chromatic_step_counts_for_fewer_jnd_where_the_ellipse_is_weak(polarity):
    """A chromatic step costs what the FITTED ellipse says it costs, derived here from
    phi/w1/w2 rather than asserted as an inequality.

    This used to assert `chromatic <= lightness` from a docstring premise that both weights
    sit below 1. That is a property of one fit, not of the metric: the 2026-09-06 sitting
    refit w2 to 1.0350, so a b' step legitimately became marginally MORE costly than a
    lightness step and the assertion failed at 1e-9 (see five-failures-reproduction.md).
    The closed form below cannot go red when a sitting moves the fit, and it is a stronger
    check than the inequality was -- it pins the rotation into the confusion axis AND both
    weightings, where `<=` only pinned their sign relative to 1.

    The weak-axis claim in the name survives as the ordering assertion: wherever the two
    weights differ, the axis with the smaller weight is the cheaper one. The `<= 1` regime
    itself is still pinned, deterministically and fit-independently, by
    `test_a_weak_ellipse_makes_every_chromatic_step_cheaper`.
    """
    reference = conspicuity.REFERENCE_J[polarity]
    ellipse = conspicuity.ELLIPSE
    lightness = observer_jnd(np.array([5.0, 0.0, 0.0]), reference, polarity)
    red_green = observer_jnd(np.array([0.0, 5.0, 0.0]), reference, polarity)
    blue_yellow = observer_jnd(np.array([0.0, 0.0, 5.0]), reference, polarity)

    # A pure a' step lands cos(phi) along the confusion axis and -sin(phi) across it, so
    # its weighted length is sqrt(w1 cos^2 + w2 sin^2); a pure b' step swaps the two.
    cos2 = np.cos(ellipse.phi) ** 2
    sin2 = np.sin(ellipse.phi) ** 2
    assert red_green == pytest.approx(lightness * np.sqrt(ellipse.w1 * cos2 + ellipse.w2 * sin2))
    assert blue_yellow == pytest.approx(lightness * np.sqrt(ellipse.w1 * sin2 + ellipse.w2 * cos2))

    weak, strong = sorted([ellipse.w1, ellipse.w2])
    if strong > weak:
        assert min(red_green, blue_yellow) < max(red_green, blue_yellow)


@pytest.mark.parametrize("polarity", polarities)
def test_a_weak_ellipse_makes_every_chromatic_step_cheaper(polarity):
    """The claim the test above used to make, kept but pinned to a FROZEN ellipse.

    Worth keeping fit-independently: "both weights below 1 means no chromatic step is
    dearer than a lightness step" is a real property of the metric, and it is exactly the
    kind of statement that must not be read off live data, because then a sitting decides
    whether the suite is green.
    """
    weak = Ellipse(phi=0.3, w1=0.4, w2=0.9, lightness_gain=0.0)
    reference = conspicuity.REFERENCE_J[polarity]
    lightness = observer_jnd(np.array([5.0, 0.0, 0.0]), reference, polarity, ellipse=weak)
    for chromatic in (np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 5.0])):
        assert observer_jnd(chromatic, reference, polarity, ellipse=weak) <= lightness + 1e-9


def test_the_isotropic_ellipse_is_plain_de_over_the_threshold():
    delta = np.array([[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]])
    steps = observer_jnd(delta, conspicuity.REFERENCE_J["day"], "day", ellipse=ISOTROPIC)
    assert steps == pytest.approx(np.array([3.0, 5.0]) / DE_MIN["day"])


def test_a_lighter_page_raises_the_step_when_the_fit_says_so():
    ellipse = Ellipse(phi=0.0, w1=1.0, w2=1.0, lightness_gain=0.5)
    delta = np.array([4.0, 0.0, 0.0])
    reference = conspicuity.REFERENCE_J["day"]
    darker = observer_jnd(delta, reference - 0.1, "day", ellipse=ellipse)
    lighter = observer_jnd(delta, reference + 0.1, "day", ellipse=ellipse)
    assert lighter < observer_jnd(delta, reference, "day", ellipse=ellipse) < darker


def test_observer_jnd_is_batched_over_rows_and_grounds():
    delta = np.tile(np.array([3.0, 1.0, -1.0]), (4, 1))
    grounds = np.linspace(0.8, 0.95, 4)
    batched = observer_jnd(delta, grounds, "day")
    single = [observer_jnd(row, g, "day") for row, g in zip(delta, grounds, strict=True)]
    assert batched == pytest.approx(single)


@pytest.mark.parametrize("polarity", polarities)
def test_the_other_matches_owe_the_meaning_roles_multiple(polarity):
    """Whatever regime `separation_floor` is in, this is its multiple -- that IS the point
    of routing it through there, so both switch together.

    This used to assert the 2x constant outright. The constant was only ever the
    unidentified regime's value, so when the 2026-09-06 sitting identified the size
    exponent the assertion failed while the code was doing exactly what it promised. The
    test now asserts the routing plus whichever regime is in force, so it holds on both
    sides of the switch. The two branches are not symmetric in what they prove: the
    constant branch pins a number, the fitted branch pins the closed form the exponent
    implies -- which is the only one of the two a refit can move.
    """
    multiple = other_baseline_jnd(polarity)
    floor, why = thresholds.separation_floor(polarity)

    # The routing claim: this reads separation_floor rather than keeping its own constant.
    assert multiple == pytest.approx(floor / DE_MIN[polarity])

    # Take the read size from the signature so this cannot drift from the function's own
    # default the way a second literal 14.0 would.
    read_size_px = inspect.signature(thresholds.separation_floor).parameters["size_px"].default

    if thresholds.size_is_identified():
        exponent = thresholds.VISION_FIT.gamma_mean
        assert multiple == pytest.approx((thresholds.REFERENCE_SIZE_PX / read_size_px) ** exponent)
        assert "retired" in why, "the fitted regime must say the constant is no longer applied"
    else:
        assert multiple == pytest.approx(thresholds.FIXED_SCALE_FACTOR)
        assert "constant" in why, "the constant regime must say out loud that it is not measured"


@pytest.mark.parametrize("polarity", polarities)
def test_baselines_require_the_good_case(polarity):
    other = other_baseline_jnd(polarity)
    assert baselines_hold(CURRENT_BASELINE_JND, other, polarity)
    assert not baselines_hold(CURRENT_BASELINE_JND - 1e-9, other, polarity)
    assert not baselines_hold(CURRENT_BASELINE_JND, other - 1e-9, polarity)
    assert not baselines_hold(float("nan"), other, polarity), "a NaN highlight must be refused, not passed"
    assert not baselines_hold(CURRENT_BASELINE_JND, float("nan"), polarity)


def test_find_time_knee_recovers_a_planted_knee():
    rng = np.random.default_rng(7)
    steps = rng.uniform(1.5, 12.0, 80)
    planted_knee, slope = 5.0, 0.35
    log_rt = np.log(2000.0) + slope * np.maximum(0.0, planted_knee - steps) + rng.normal(0, 0.15, 80)
    knee = find_time_knee(steps, np.exp(log_rt))
    assert knee.n == 80
    assert abs(knee.knee_jnd - planted_knee) < 0.8
    assert knee.slope_per_step == pytest.approx(-slope, abs=0.1)
    assert knee.gain > 0.4


def test_find_time_knee_reports_no_gain_on_flat_data_and_none_on_too_few():
    rng = np.random.default_rng(3)
    steps = rng.uniform(2.0, 12.0, 60)
    flat = find_time_knee(steps, np.exp(np.log(2000.0) + rng.normal(0, 0.2, 60)))
    assert flat.gain < 0.15, "a hinge on noise must not claim to have found a knee"
    assert find_time_knee(steps[:5], np.full(5, 2000.0)) is None
