"""What the trial schedule must guarantee.

The recorder rebuilds a trial from the log rather than trusting what the page sent back.
That only works if trial generation is a pure function of the trial number and the log --
so determinism here is not a nicety, it is what makes every archived response describe the
stimulus that was actually shown.
"""

import random

import numpy as np
import pytest

from theme import responses, schedule
from theme.schedule import NIGHT_SHARE, block_polarity, duel_surface, run_info, schedule_mode, trial_for
from theme.space import POOL
from theme.stimulus import READING_PX, SURFACES

BLOCK = 24  # trials per polarity block
DUELS_PER_BLOCK = 16


def duels(day, night):
    """A log of answered duels, `day` of one polarity and `night` of the other."""
    return [{"mode": "duel", "polarity": "day", "choice": 0}] * day + [
        {"mode": "duel", "polarity": "night", "choice": 0}
    ] * night


#: Plenty of duels, balanced, so the bootstrap is long over and the polarity steering is
#: at its target: what the arm schedule looks like in steady state.
STEADY = duels(400, 600)


@pytest.fixture(scope="module")
def answered():
    """The real log, read only. Its length and composition drive the schedule."""
    return responses.DEFAULT_LOG.read()


def test_generating_the_same_trial_twice_gives_the_same_trial(answered):
    """The property the whole recording scheme rests on."""
    n = len(answered)
    first, second = trial_for(n, answered), trial_for(n, answered)
    assert first["theta_a"] == second["theta_a"]
    assert first.get("theta_b") == second.get("theta_b")
    assert (first["mode"], first["polarity"], first["snippet"], first["code_px"]) == (
        second["mode"],
        second["polarity"],
        second["snippet"],
        second["code_px"],
    )


def test_a_duel_varies_the_theme_and_nothing_else(answered):
    """Both halves render the SAME page, so the only difference is the theme under test."""
    n = next(i for i in range(len(answered), len(answered) + BLOCK) if trial_for(i, answered)["mode"] == "duel")
    trial = trial_for(n, answered)
    assert trial["theta_a"] != trial["theta_b"], "a duel against itself measures nothing"
    assert "snippet" in trial and isinstance(trial["snippet"], int)


@pytest.mark.parametrize("slot,expected", [(0, "duel"), (15, "duel"), (16, "comprehension"), (20, "search")])
def test_a_run_batches_trials_of_one_kind(slot, expected):
    """Sixteen duels, then four probes, then four hunts: one instruction serves a whole run,
    so no click is spent re-reading it and the task never switches mid-stride."""
    _polarity, arm = schedule_mode(slot, STEADY)
    assert arm == expected


def test_polarity_alternates_by_block_until_the_log_can_steer():
    """Blocked rather than interleaved, so the observer's adaptation state is part of the
    measurement instead of noise in it -- a light page judged with dark-adapted eyes is a
    different stimulus."""
    assert schedule_mode(0, [])[0] != schedule_mode(BLOCK, [])[0]
    assert schedule_mode(0, [])[0] == schedule_mode(2 * BLOCK, [])[0]


def test_a_block_runs_night_while_night_is_short_of_its_share():
    """Titus's call of 2026-09-05: night's fit predicted its own second half at 44.6%, so
    the blocks steer night duels first rather than alternating. The steering is decided at
    the block boundary from the duels before it, and every trial in the block agrees."""
    short_of_night = duels(111, 104)
    polarities = {block_polarity(n, short_of_night) for n in range(len(short_of_night), len(short_of_night) + BLOCK)}
    assert polarities == {"night"}


def test_a_block_runs_day_once_night_holds_its_share():
    at_target = duels(400, 608)  # 1008 rows: a whole number of blocks, so the boundary is the log's end
    assert block_polarity(len(at_target), at_target) == "day"
    assert 0.5 < NIGHT_SHARE < 1.0, "a share outside (0.5, 1) is not 'weight night', it is a different rule"


def test_a_block_does_not_change_polarity_because_a_duel_landed_inside_it():
    """The boundary decides; rows arriving inside the block cannot flip it, or the run
    gate's instruction would name one polarity and the stimulus another."""
    history = duels(100, 100)
    start = len(history) - len(history) % BLOCK
    first = block_polarity(start, history)
    for extra in range(1, BLOCK):
        grown = history + duels(0, extra) if first == "night" else history + duels(extra, 0)
        assert block_polarity(start + extra, grown) == first


def test_surfaces_are_sampled_equally_often():
    """NOT `n % 3`. The block is 24 trials of which 16 are duels, and 3 divides 24, so a
    modular rotation never de-phases: one surface would take 6 of every 16 duels and the
    others 5, forever. The log showed exactly that lock before it was fixed."""
    duels = [duel_surface(n, 200) for n in range(BLOCK * 12) if n % BLOCK < DUELS_PER_BLOCK]
    counts = {surface: duels.count(surface) for surface in SURFACES}
    assert max(counts.values()) - min(counts.values()) <= 1, f"unbalanced over 12 blocks: {counts}"


def test_the_first_duel_of_a_run_is_not_always_the_same_surface():
    """Position within a run carries its own effects -- freshest eyes, and the largest
    adaptation step from whatever was on screen before -- so it must not be confounded
    with which surface is being shown."""
    first_of_run = {duel_surface(BLOCK * block, 200) for block in range(12)}
    assert len(first_of_run) == len(SURFACES), f"slot 0 only ever shows {sorted(first_of_run)}"


def test_each_surface_is_shown_at_the_size_it_is_read_at(answered):
    """Preference measured at 12-13px was being applied to reading at 14 and 16. Contrast
    sensitivity falls with glyph scale, so that is not a free assumption in a colour
    experiment."""
    for n in range(len(answered), len(answered) + BLOCK):
        trial = trial_for(n, answered)
        if trial["mode"] == "duel":
            assert trial["code_px"] == READING_PX[trial["surface"]]


def test_run_info_agrees_with_the_arm_schedule():
    """Two functions describe the same schedule; they must not drift apart."""
    for n in range(3 * BLOCK):
        polarity, arm, position, run_length = run_info(n, STEADY)
        assert (polarity, arm) == schedule_mode(n, STEADY)
        assert 0 <= position < run_length


def test_the_memo_never_serves_a_trial_built_from_a_different_log(answered):
    """A memo may only ever return the same computation it would have performed.

    Keyed by trial number alone it did not: on the real log, trial 40 asked for with the log
    and asked for with an empty history returned the same object -- a comprehension probe on
    the editor, when the second caller's own answer is a duel on the panel. Reachable in one
    pytest process by test order, since the suite drives the app with a scratch log while
    other tests read the real one, and reachable in the recorder as a row describing a theme
    that was never on screen.

    So this asserts both halves: the same history always gives the same trial, and a
    different history gives a different one rather than whatever was cached first.
    """
    n = 40
    if len(answered) < n:
        pytest.skip(f"needs {n} recorded responses, log has {len(answered)}")

    with_history = trial_for(n, answered)
    schedule.TRIAL_MEMO.clear()
    recomputed = trial_for(n, answered)
    assert with_history == recomputed, "the same history must give the same trial"

    with_none = trial_for(n, [])
    assert (with_none["mode"], with_none["theta_a"]) != (with_history["mode"], with_history["theta_a"]), (
        "an empty history must not be answered from the cached full-history trial"
    )


def test_a_find_hunt_survives_a_page_whose_highlight_cannot_be_realized():
    """The sweep replaces the find axes with a fixed lattice, which need not contain the
    values that made the champion's own theme realizable -- so every variant can be refused
    at once, leaving nothing to choose between. That raised on an empty list, which on the
    click path is a 500 and a stalled sitting. The base below is such a page."""
    base = np.array([0.995, 0.316, 0.183, 0.88, 0.812, 0.668, 0.958, 0.926, 0.748])
    assert schedule._conspicuous_grid(schedule._find_axis_grid(base), "day") == [], (
        "this base no longer refuses every variant -- find another, or drop this test"
    )
    trial = schedule._search_trial(0, [], "day", None, POOL["day"], random.Random(0), np.random.default_rng(0))
    assert trial["mode"] == "search"
    assert trial["theme_a"] is not None, "a hunt with no theme is a blank page"
    assert trial["code_px"] == READING_PX["editor"]


class TestShelfDuels:
    """Once a plateau forms, the duels have to attack the plateau.

    The information-gain challenger maximises information about who wins THIS duel, which
    any uncertain pairing anywhere in the space supplies. The verdict needs information
    about which theme is BEST, and once several themes share the probability of being best,
    that uncertainty lives entirely in comparisons between them.

    Measured before this existed: of the next sixteen day duels, zero put two shelf members
    together and the median utility gap between arms was 0.59 -- the instrument was
    separating themes it could already tell apart. After: eight of sixteen, and the median
    gap halved to 0.28.
    """

    def test_a_plateau_draws_duels_between_its_own_members(self, answered):
        from theme import preference, space
        from theme.schedule import TRIAL_MEMO

        fit = preference.fitted(answered)
        TRIAL_MEMO.clear()
        kinds = [
            trial["kind"]
            for n in range(len(answered), len(answered) + 48)
            if (trial := trial_for(n, answered))["mode"] == "duel"
        ]
        assert "shelf" in kinds, (
            f"a plateau exists on at least one polarity but no upcoming duel compares two of "
            f"its members; trial kinds were {sorted(set(kinds))}"
        )
        assert space and fit  # the fixture is only meaningful with a real fit behind it

    def test_a_shelf_duel_is_a_closer_contest_than_the_field(self, answered):
        """The point is not that shelf duels happen, it is that they are harder."""
        from theme import preference
        from theme.schedule import TRIAL_MEMO

        fit = preference.fitted(answered)
        TRIAL_MEMO.clear()
        gaps = {"shelf": [], "eig": []}
        for n in range(len(answered), len(answered) + 64):
            trial = trial_for(n, answered)
            if trial["mode"] != "duel" or trial["kind"] not in gaps:
                continue
            mean = preference.mean_utility_at(fit, [trial["theta_a"], trial["theta_b"]], trial["polarity"])
            gaps[trial["kind"]].append(abs(float(mean[0] - mean[1])))
        if not gaps["shelf"] or not gaps["eig"]:
            pytest.skip("this log produced only one kind of duel over the sampled window")
        shelf_median = sorted(gaps["shelf"])[len(gaps["shelf"]) // 2]
        field_median = sorted(gaps["eig"])[len(gaps["eig"]) // 2]
        assert shelf_median <= field_median, (
            f"shelf duels should be the closer contests: median utility gap {shelf_median:.2f} "
            f"against {field_median:.2f} for the field"
        )

    def test_a_single_clear_leader_draws_no_shelf_duels(self, monkeypatch, answered):
        """No plateau, no shelf duels. A lone leader needs checking against the space rather
        than against itself, or the search confines itself to what it already believes."""
        from theme import schedule

        monkeypatch.setattr(schedule, "best_set", lambda *a, **k: {"verdict": "single", "credible": [0, 1, 2]})
        assert schedule._shelf_indices(object(), "day", [None]) == []
