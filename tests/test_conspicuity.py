"""The conspicuity metric and the highlight baseline.

The metric has to be the observer's own: one step along lightness at the reference page IS
one DE_MIN, a chromatic step costs more dE where the fitted ellipse is weak, and a lighter
page raises the step. The baseline has to fail closed. The knee fit has to recover a knee
that was planted -- the recovery-test shape the rest of the suite uses.
"""

import numpy as np
import pytest

from theme import conspicuity
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
    """The fitted ellipse has w1 and w2 below 1 (or equal to it in the isotropic fallback),
    so the same dE along a' or b' is never seen as MORE than along lightness."""
    reference = conspicuity.REFERENCE_J[polarity]
    lightness = observer_jnd(np.array([5.0, 0.0, 0.0]), reference, polarity)
    red_green = observer_jnd(np.array([0.0, 5.0, 0.0]), reference, polarity)
    blue_yellow = observer_jnd(np.array([0.0, 0.0, 5.0]), reference, polarity)
    assert red_green <= lightness + 1e-9
    assert blue_yellow <= lightness + 1e-9
    weak, strong = sorted([conspicuity.ELLIPSE.w1, conspicuity.ELLIPSE.w2])
    if strong > weak:
        assert min(red_green, blue_yellow) < max(red_green, blue_yellow)


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
    """Until the size exponent is identified that multiple is the 2x constant; the point of
    routing it through separation_floor is that both switch regime together."""
    assert other_baseline_jnd(polarity) == pytest.approx(2.0)


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
