"""The path a click takes: request in, record on disk, next trial out.

These are the tests the old notebook version could not have: every piece of it worked in
isolation while the trial still vanished from the screen. A suite that only checks the
model's arithmetic cannot see that, so the flow is checked as a flow.

Three things here are checked at the level of the served page rather than the model,
because that is the only level at which they are true or false: the instrument's own chrome
has to be legible on the surround the trial paints, a duel has to vary the theme and
nothing else, and the clock must not start before the run is begun.
"""

import re
from pathlib import Path

import pytest
from conftest import correct_choice, report

from theme import schedule
from theme.color import hex_to_rgb, wcag

APP_JS = Path(__file__).resolve().parents[1] / "theme" / "static" / "app.js"

#: WCAG AA for body text. The chrome carries the instruction and the begin button, so it is
#: body text; the themes under test are held to this and the frame around them cannot be
#: held to less.
CHROME_FLOOR = 4.5


def chrome_contrast(trial: dict) -> float:
    """How far the instrument's own ink stands off the surround this trial paints."""
    ink = hex_to_rgb([trial["chrome_ink"]])
    ground = hex_to_rgb([trial["page_bg"]])
    return float(wcag(ink, ground)[0])


def text_of(html: str) -> str:
    """The characters a reader sees, with every tag and therefore every colour removed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", html)).strip()


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
    a gate and an arm change -- which is where the notebook version lost the stage. The
    chrome is checked on every one of them, so the single-card arms are covered too: those
    paint the page with the theme under test rather than with the neutral surround, which is
    the case where a legible frame is not automatic.
    """
    for n in range(20):
        out = client.post("/api/response", json=report(n, correct_choice(n, scratch_log.read()))).json()
        assert out["ok"] is True, f"trial {n} was refused: {out}"
        following = out["next"]
        assert following["cards"], f"trial {n + 1} came back with nothing to show"
        ratio = chrome_contrast(following)
        assert ratio >= CHROME_FLOOR, (
            f"trial {following['n']} ({following['polarity']}, {following['mode']}) draws its chrome at "
            f"{ratio:.2f}:1 on {following['page_bg']} -- an invisible instruction and an invisible button"
        )
    rows = scratch_log.read()
    assert len(rows) == 20
    assert [r["n"] for r in rows] == list(range(20)), "trial numbers must be dense and ordered"


@pytest.mark.parametrize("n,expected_polarity", [(0, "day"), (24, "night")])
def test_the_chrome_is_legible_on_both_polarities(client, n, expected_polarity):
    """A screenshot once caught the chrome rendering at 1.1:1 on a light ground, which no
    test asserted and every test passed through. The surround flips with polarity, so an
    ink that is right for one is wrong for the other and only checking both catches it."""
    trial = client.get(f"/api/trial/{n}").json()
    assert trial["polarity"] == expected_polarity
    ratio = chrome_contrast(trial)
    assert ratio >= CHROME_FLOOR, (
        f"{expected_polarity} chrome {trial['chrome_ink']} on {trial['page_bg']} is {ratio:.2f}:1"
    )


def test_a_duel_shows_two_themes_on_one_page(client):
    """A duel varies the theme and nothing else. Two cards, two different grounds, and the
    same characters in both -- if the pages differed, the answer would be about the code."""
    trial = client.get("/api/trial/0").json()
    assert trial["is_duel"] and len(trial["cards"]) == 2
    left, right = trial["cards"]
    assert left["ground"] != right["ground"], "a duel against the same theme measures nothing"
    assert text_of(left["html"]) == text_of(right["html"]), (
        "the two halves of a duel must render the same page, or the choice is about the code"
    )
    assert text_of(left["html"]), "a card with no text is a blank half-screen"
    assert left["html"] != right["html"], "same text, different colours -- that is the stimulus"


def test_the_first_trial_is_gated_so_no_clock_runs_on_an_empty_room(client):
    """Opening the tab and walking away must not start a clock. The run gate covers the
    stimulus until begin is pressed, and the page reveals -- and only then baselines the
    clock -- on that press."""
    trial = client.get("/api/trial/0").json()
    assert trial["gate"] is True, "the first trial of a run must arrive gated"
    assert trial["gate_text"], "a gate with no instruction is a dead end"


def test_the_clock_is_baselined_only_by_a_reveal():
    """The one invariant behind every reaction time, checked in the source because that is
    where it lives: nothing but reveal() may set the clock's baseline.

    A `t_render` set while the stimulus was still covered would silently fold the time
    spent reading an instruction, or sitting in another tab, into the measurement -- and
    every row would still look like a perfectly ordinary reaction time.
    """
    source = APP_JS.read_text()
    assignments = [line.strip() for line in source.splitlines() if re.match(r"^\s*(let\s+)?t0\s*=", line)]
    assert assignments, "app.js no longer has a clock baseline called t0 -- update this test"
    for line in assignments:
        assert "-1" in line or "performance.now()" in line, f"unexpected clock baseline: {line}"
    now_assignments = [line for line in assignments if "performance.now()" in line]
    assert len(now_assignments) == 1, (
        f"the clock must be baselined in exactly one place (reveal); found {len(now_assignments)}"
    )
    reveal = source.split("function reveal()", 1)[1].split("\nfunction ", 1)[0]
    assert now_assignments[0] in reveal, "the clock's baseline must be set inside reveal()"
    assert "t0 < 0" in source, "answering with no baseline must be refused, not timed from -1"


def test_a_refused_answer_is_shown_to_whoever_gave_it():
    """A dropped answer that the page swallows is worse than an error: the trial advances,
    the response is gone, and nothing on screen says so. That cost a sitting once."""
    source = APP_JS.read_text()
    assert "out.ok === false" in source, "the page must inspect the server's refusal"
    stale_branch = source.split("out.ok === false", 1)[1].split("\n  }", 1)[0]
    assert "interrupt(" in stale_branch or "cover(" in stale_branch, (
        "a refused answer must put something on screen, not advance in silence"
    )


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


def test_a_row_describes_the_stimulus_that_was_painted_across_a_restart(client, scratch_log):
    """The recorded row must describe the page that was on screen, with nothing cached in
    the server standing between the two.

    The sequence matters, because the way this broke needs all of it. Answer enough trials
    that the model is fitted. Ask for a trial that is not answerable yet -- a look-ahead,
    which is what the page used to do to warm the server and what any request for a future
    trial still does; that trial can only be built from a truncated log, since the row it
    should have been fitted on has not been written. Answer the current trial, and the
    server hands back the next one. Then drop every in-process memo -- a restart, with the
    tab still open -- and answer what is on screen.

    The grounds the row says were compared have to be the grounds the page painted. They
    were not, before the trial memo was keyed on the log as well as the trial number: the
    look-ahead's truncated-history trial was served back under the same key as the trial
    built from the full log, so the row described a theme that was never on screen.
    """
    for n in range(5):
        client.post("/api/response", json=report(n, correct_choice(n, scratch_log.read())))

    client.get("/api/trial/6")  # a look-ahead, while row 5 is still unanswered
    out = client.post("/api/response", json=report(5, correct_choice(5, scratch_log.read()))).json()
    assert out["ok"] is True
    shown = out["next"]
    assert shown["n"] == 6 and shown["is_duel"]
    painted = sorted(card["ground"] for card in shown["cards"])

    schedule.TRIAL_MEMO.clear()
    out = client.post("/api/response", json=report(6, correct_choice(6, scratch_log.read()))).json()
    assert out["ok"] is True

    row = scratch_log.read()[6]
    recorded = sorted([row["theme_a"]["ground"], row["theme_b"]["ground"]])
    assert painted == recorded, (
        f"the row names themes that were never on screen: painted {painted}, recorded {recorded}"
    )
    assert row["page_bg"] == shown["page_bg"], "the row's surround must be the one painted"


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


def test_a_log_under_a_directory_that_does_not_exist_yet_still_works(tmp_path):
    """A fresh checkout has no data/ directory. Reading must be an empty log rather than an
    error, and the first append must create the directory -- otherwise the first sitting on
    a new machine fails at the first click."""
    from theme import responses

    log = responses.ResponseLog(tmp_path / "not" / "created" / "responses.jsonl")
    assert log.read() == [], "a missing log is an empty log, not an error"
    log.append({"n": 0, "mode": "duel"})
    assert log.read() == [{"n": 0, "mode": "duel"}]


def test_the_page_and_its_assets_are_served(client):
    assert client.get("/").status_code == 200
    for asset in ("/static/app.js", "/static/app.css"):
        assert client.get(asset).status_code == 200, f"{asset} missing means an unstyled or dead page"
