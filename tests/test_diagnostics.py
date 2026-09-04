"""The readouts that say whether the answer is settled -- and whether to believe them.

Three of them, and each has its own way of lying.

`factor_effect` asks whether the preferred theme depends on some logged property of how it
was SHOWN. At these counts, giving the utility two extra parameters clears a fixed
threshold by chance in roughly one run in five, so a permutation null is not decoration and
the test is calibrated in BOTH directions: quiet when nothing is there, awake when
something is.

The guard that matters most, though, is the one that counts the duels which reached the
test. An earlier version of this file encoded the winner as "a"/"b" instead of 0/1, the
filter dropped every row, the function returned its not-enough-data p of 1.0, and "stays
quiet under a true null" passed vacuously -- a green test measuring nothing at all.

`axis_consensus` says WHAT the surviving themes disagree about, which is the difference
between "four distinct themes" and "the ground is decided, the accent hue is not".

`progress_report` says whether another sitting is worth clicking, by comparing the verdict
now against the verdict as it stood `back` duels ago on the SAME candidate set.
"""

import numpy as np
import pytest

# The Bradley-Terry generators live with the model they exercise; a second copy of them
# here would drift from that one, and a drifted test is worse than none.
from test_preference_model import clone_and_spread_candidates, duel_log

SURFACES = ("editor", "panel", "notebook")


def surface_duels(tilt, n=96, seed=0):
    """Duels whose utility is tilted on axis 0 by which surface they were shown on.

    `tilt` of 0 is a true null -- one utility, three surfaces -- and a positive tilt moves
    the optimum in opposite directions on the editor and the panel.
    """
    r = np.random.default_rng(seed)
    w = np.zeros(9)
    w[[0, 4, 6]] = [1.5, 1.0, -1.0]
    rows = []
    for i in range(n):
        a, b = r.random(9), r.random(9)
        s = SURFACES[i % 3]
        z = (a - b) @ w + tilt * (1.0 if s == "editor" else -1.0 if s == "panel" else 0.0) * (a - b)[0]
        rows.append(
            {
                "mode": "duel",
                "polarity": "day",
                "surface": s,
                "paused": False,
                "theta_a": list(map(float, a)),
                "theta_b": list(map(float, b)),
                # 0 = theme_a won, matching duels_from's encoding. Against "a"/"b" this
                # test passed vacuously: the filter dropped every row, the function
                # returned its not-enough-data p of 1.0, and "stays quiet" read as a pass.
                "choice": 0 if r.random() < 1.0 / (1.0 + np.exp(-z)) else 1,
            }
        )
    return rows


@pytest.fixture(scope="module")
def factor_calibration(search_model):
    """`factor_effect` on the surface factor: the row count, six nulls and six planted tilts.

    Computed in this order deliberately. `factor_effect` memoises on (key, polarity, data)
    and NOT on nperm, so the seed-0 null below is served the 20-permutation answer cached
    by the row-count call above it (p = 0.95, where a fresh 120-permutation fit gives
    0.88). Both are nowhere near the 0.10 threshold, so the calibration reads the same
    either way -- but the order is pinned here rather than left to whichever order pytest
    happens to call the tests in. `surface_effect` is this same call with key="surface".
    """
    n_seen = search_model.factor_effect(surface_duels(0.0, seed=0), "day", "surface", nperm=20)[0]
    p_null = [search_model.factor_effect(surface_duels(0.0, seed=s), "day", "surface", nperm=120)[2] for s in range(6)]
    p_real = [search_model.factor_effect(surface_duels(2.5, seed=s), "day", "surface", nperm=120)[2] for s in range(6)]
    return {"n_seen": n_seen, "p_null": p_null, "p_real": p_real}


def test_the_factor_test_sees_the_duels_it_is_given(factor_calibration):
    """The non-vacuity guard: a quiet answer only means something if the rows arrived.

    Every other assertion about this function is worthless without it, which is why it is
    a test of its own rather than a clause in one.
    """
    assert factor_calibration["n_seen"] == 96, f"{factor_calibration['n_seen']} of 96 synthetic duels reached the test"


def test_the_factor_test_stays_quiet_under_a_true_null(factor_calibration):
    """Six nulls, at most one of them allowed to fire below p = 0.10.

    Measured p-values: [0.95, 0.65, 0.9, 0.83, 0.94, 0.62]. A fixed information-criterion
    threshold on the same data fires about 1 in 5, which is exactly why the null here is a
    permutation of the surface labels rather than a table lookup.
    """
    fired = sum(p < 0.10 for p in factor_calibration["p_null"])
    assert fired <= 1, (
        f"{fired} of 6 fired under a true null; p-values {[round(p, 2) for p in factor_calibration['p_null']]}"
    )


def test_the_factor_test_finds_a_planted_surface_effect(factor_calibration):
    """And it has to have power: at least 4 of 6 runs with a 2.5-logit tilt planted.

    Measured p-values: [0.35, 0.0, 0.0, 0.0, 0.0, 0.0]. Note what 2.5 logits means -- at
    48 duels a tilt of 1 logit was detected 1 run in 12, so a quiet verdict from the real
    log reads as "not visible here", never as "settled".
    """
    fired = sum(p < 0.10 for p in factor_calibration["p_real"])
    assert fired >= 4, (
        f"only {fired} of 6 fired with a planted 2.5-logit tilt; "
        f"p-values {[round(p, 2) for p in factor_calibration['p_real']]}"
    )


@pytest.fixture(scope="module")
def consensus(search_model):
    """P(best) mass concentrated on one value of axis 0 and spread over axis 5.

    240 themes, weighted by a narrow Gaussian on axis 0 alone: axis 0 is settled near 0.8
    and axis 5 is untouched, which is the distinction the readout exists to draw.
    """
    rng = np.random.default_rng(5)
    th = rng.random((240, 9))
    w = np.exp(-((th[:, 0] - 0.8) ** 2) / (2 * 0.05**2))
    w = w / w.sum()
    return {c[0]: c for c in search_model.axis_consensus({"p_best": w}, th)}


def test_a_settled_axis_reads_narrow_and_an_untouched_one_reads_wide(consensus):
    """Spread is reported relative to the 0.289 of a uniform axis, so the numbers compare.

    Measured: settled axis 0.16, untouched axis 1.07. The gap the assertion demands is
    wide (below 0.4 against above 0.8) because a readout that only just separated a
    concentrated axis from an untouched one could not be trusted on a real log.
    """
    settled, untouched = consensus[0][1], consensus[5][1]
    assert settled < 0.4 < 0.8 < untouched, (
        f"spread relative to uniform: settled axis {settled:.2f}, untouched axis {untouched:.2f}"
    )


def test_the_settled_axis_reports_where_it_settled(consensus):
    """Narrow is only half the answer; the readout also has to say WHERE, or it cannot tell
    him which value his clicks have chosen. Measured 0.794 against a planted 0.80."""
    mean = consensus[0][2]
    assert abs(mean - 0.8) < 0.05, f"posterior-weighted mean of the settled axis {mean:.3f} against a planted 0.80"


def test_progress_report_compares_two_fits_and_reports_movement(search_model):
    """Two fits, 40 duels apart, over one fixed candidate set.

    The candidate set is held fixed on purpose: refitting against freshly bred candidates
    would make the comparison about which themes happened to be bred rather than about the
    evidence. What comes back has to be a real readout -- 300 duels counted, and at least
    one group in the credible set -- because "None, needs more duels" is a legitimate
    answer from this function and would otherwise pass silently.
    """
    responses = duel_log(search_model, 300, seed=8)
    clones, spread = clone_and_spread_candidates()
    prog = search_model.progress_report(responses, "day", spread + clones, back=40)
    assert prog is not None, "None, i.e. needs more duels -- but 300 were supplied"
    assert prog["duels"] == 300, f"{prog['duels']} duels counted of 300 supplied"
    assert prog["set_now"] >= 1, (
        f"leader {100 * prog['lead_then']:.0f}% -> {100 * prog['lead_now']:.0f}%, "
        f"set {prog['set_then']} -> {prog['set_now']} over {prog['back']} duels"
    )
