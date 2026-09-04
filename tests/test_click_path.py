"""The path a click takes: request in, record on disk, next trial out.

These are the tests the old notebook version could not have: every piece of it worked in
isolation while the trial still vanished from the screen. A suite that only checks the
model's arithmetic cannot see that, so the flow is checked as a flow.
"""

import pytest
from conftest import correct_choice, report


def test_a_fresh_log_starts_at_the_first_trial(client):
    assert client.get("/api/status").json() == {"responses": 0, "duels": 0}


def test_a_trial_carries_everything_the_page_needs(client):
    trial = client.get("/api/trial/0").json()
    # If any of these is missing the page renders something blank or unclickable, which is
    # precisely the failure mode this project was restructured to remove.
    for field in ("n", "mode", "polarity", "is_duel", "chip", "prompt_html", "cards", "page_bg", "gate"):
        assert field in trial, f"a trial payload without {field!r} cannot be displayed"
    assert trial["cards"], "a trial with no cards is a blank screen"
    assert all(card["html"] and card["ground"] for card in trial["cards"])


def test_answering_records_one_row_and_returns_the_next_trial(client, scratch_log):
    trial = client.get("/api/trial/0").json()
    out = client.post("/api/response", json=report(0, correct_choice(0, []))).json()
    assert out["ok"] is True
    assert len(scratch_log.read()) == 1, "the answer must reach the log"
    assert out["next"]["n"] == 1, "the next trial comes back in the same round trip"
    assert out["next"]["cards"], "and it is displayable"
    assert out["next"] != trial


def test_twenty_consecutive_trials_stay_answerable(client, scratch_log):
    """The regression test for the blank screen, at the level where it can be automated.

    Twenty is past the first run boundary (sixteen duels, then four probes), so it crosses
    a gate and an arm change -- which is where the notebook version lost the stage.
    """
    for n in range(20):
        out = client.post("/api/response", json=report(n, correct_choice(n, scratch_log.read()))).json()
        assert out["ok"] is True, f"trial {n} was refused: {out}"
        assert out["next"]["cards"], f"trial {n + 1} came back with nothing to show"
    rows = scratch_log.read()
    assert len(rows) == 20
    assert [r["n"] for r in rows] == list(range(20)), "trial numbers must be dense and ordered"


def test_a_stale_answer_is_refused_rather_than_misrecorded(client, scratch_log):
    """A double click, or a page left open across a restart, must not write to another row.

    The trial is recomputed from the log at record time, so an answer for a trial that is
    no longer next cannot be attributed to whatever is next instead. It is dropped, and the
    page is handed the trial it should actually be showing.
    """
    client.post("/api/response", json=report(0, correct_choice(0, [])))
    out = client.post("/api/response", json=report(0, correct_choice(0, []))).json()
    assert out["ok"] is False
    assert out["reason"] == "stale"
    assert len(scratch_log.read()) == 1, "the stale answer must not have been recorded"
    assert out["next"]["n"] == 1, "and the page is told where it actually is"


def test_the_recorded_row_describes_the_stimulus_that_was_shown(client, scratch_log):
    """A row that does not carry its own conditions cannot be re-analysed later.

    This log has already been re-read under three successive models; each time, what made
    that possible was that every row said what was on screen.
    """
    client.get("/api/trial/0")
    client.post("/api/response", json=report(0, correct_choice(0, [])))
    row = scratch_log.read()[0]
    for field in ("theta_a", "theme_a", "polarity", "surface", "code_px", "page_bg", "rt_ms", "paused"):
        assert field in row, f"a response without {field!r} is not re-analysable"
    assert row["rt_ms"] == pytest.approx(1500.0)
    assert row["paused"] is False


def test_a_paused_trial_is_flagged_so_its_clock_is_not_believed(client, scratch_log):
    client.post("/api/response", json=report(0, correct_choice(0, []), pauses=1))
    assert scratch_log.read()[0]["paused"] is True


def test_the_page_and_its_assets_are_served(client):
    assert client.get("/").status_code == 200
    for asset in ("/static/app.js", "/static/app.css"):
        assert client.get(asset).status_code == 200, f"{asset} missing means an unstyled or dead page"
