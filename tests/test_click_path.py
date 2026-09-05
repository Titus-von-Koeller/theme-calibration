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

from theme import schedule, vision
from theme.color import hex_to_rgb, wcag
from theme.server import CHROME_INK, app, get_log, get_posterior, get_vision_log

#: The first colour trial of the first block, and the trial after the colour run: a run that
#: reaches the second is a run that crossed the arm.
FIRST_COLOUR = schedule.DUELS_PER_BLOCK + 2 * schedule.PROBES_PER_BLOCK
PAST_THE_COLOUR_RUN = schedule.BLOCK + 1

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


def answer_next(client, scratch_log, scratch_vision_log, scratch_posterior, *, choice=None, **how) -> dict:
    """Answer whatever trial is next, correctly unless a choice is given, the way the page
    would: the trial rebuilt from the logs, the vision numbering echoed back."""
    answered, vision_answered = scratch_log.read(), scratch_vision_log.read()
    n = len(answered)
    if choice is None:
        choice = correct_choice(n, answered, vision_answered, scratch_posterior)
    return client.post("/api/response", json=report(n, choice, vision_n=len(vision_answered), **how)).json()


def seeded_vision_log(scratch_vision_log, scratch_posterior, rows: int) -> None:
    """A vision log as a sitting in the app would have left it: `rows` answered trials, all
    correct, from the app's generator."""
    history = []
    for m in range(rows):
        stimulus = vision.trial_for(
            m, history, sizes=vision.APP_SIZES, generator=vision.APP_GENERATOR, posterior=scratch_posterior
        )
        row = vision.build_entry(m, stimulus, stimulus["odd_position"], f"2026-09-05T10:{m:02d}:00+00:00")
        scratch_vision_log.append(row)
        history.append(row)


def test_a_fresh_log_starts_at_the_first_trial(client):
    assert client.get("/api/status").json() == {"responses": 0, "duels": 0}


def test_the_warm_up_fits_the_logs_the_app_is_wired_to(
    monkeypatch, scratch_log, scratch_vision_log, scratch_posterior
):
    """The lifespan warm-up must go through the same seams as the endpoints. Read from the
    module defaults it fitted the REAL aesthetics log at every test-client start, and would
    have run the real observer posterior -- sidecar write included -- from inside the suite
    the moment the real log stood at a colour slot."""
    from fastapi.testclient import TestClient

    from theme import server

    warmed = []
    monkeypatch.setattr(server, "payload", lambda *args: warmed.append(args))
    app.dependency_overrides[get_log] = lambda: scratch_log
    app.dependency_overrides[get_vision_log] = lambda: scratch_vision_log
    app.dependency_overrides[get_posterior] = lambda: scratch_posterior
    try:
        with TestClient(app):
            pass
    finally:
        app.dependency_overrides.clear()
    assert len(warmed) == 1, "the warm-up fits exactly the next trial"
    n, answered, vision_answered, posterior = warmed[0]
    assert (n, answered, vision_answered) == (0, [], []), "the scratch logs, not the real ones"
    assert posterior is scratch_posterior


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


def test_a_run_through_the_colour_arm(client, scratch_log, scratch_vision_log, scratch_posterior):
    """Thirty-three consecutive trials from an empty log: sixteen duels, four probes, four
    hunts, eight colour trials, and the first duel of the next block. The colour arm is the
    one that writes to a second log and paints the page with a ground of its own, so every
    served trial is checked for a legible frame and both logs are checked for what they
    gained. One colour trial is answered wrong on purpose, and one is answered after a
    notebook sitting appended to the vision log, which must be refused."""
    served = {}
    for n in range(PAST_THE_COLOUR_RUN):
        if n == FIRST_COLOUR:
            # The vision log is shared with the notebook. A row lands there between the
            # page being shown and the answer arriving; the answer echoes the numbering it
            # was shown and is refused, and nothing is written to either log.
            foreign = vision.trial_for(0, [], posterior=scratch_posterior)
            scratch_vision_log.append(vision.build_entry(0, foreign, 0, "2026-09-05T09:00:00+00:00"))
            stale = client.post("/api/response", json=report(n, 0, vision_n=0)).json()
            assert stale == {"ok": False, "reason": "stale", "next": stale["next"]}
            assert stale["next"]["n"] == n and stale["next"]["vision_n"] == 1
            assert len(scratch_log.read()) == n and len(scratch_vision_log.read()) == 1
        wrong = n == FIRST_COLOUR + 1
        choice = (correct_choice(n, scratch_log.read(), scratch_vision_log.read(), scratch_posterior) + 1) % 4
        out = answer_next(client, scratch_log, scratch_vision_log, scratch_posterior, choice=choice if wrong else None)
        assert out["ok"] is True, f"trial {n} was refused: {out}"
        following = out["next"]
        served[following["n"]] = following
        assert following["cards"], f"trial {n + 1} came back with nothing to show"
        ratio = chrome_contrast(following)
        assert ratio >= CHROME_FLOOR, (
            f"trial {following['n']} ({following['polarity']}, {following['mode']}) draws its chrome at "
            f"{ratio:.2f}:1 on {following['page_bg']}"
        )

    colour_trials = [served[n] for n in range(FIRST_COLOUR, schedule.BLOCK)]
    assert [t["mode"] for t in colour_trials] == ["discrimination"] * schedule.DISCRIMINATION_PER_BLOCK
    assert served[schedule.BLOCK]["mode"] == "duel", "the block after the colour run starts over"
    assert colour_trials[0]["gate"] is True and not any(t["gate"] for t in colour_trials[1:]), (
        "one instruction serves the whole colour run"
    )
    for trial in colour_trials:
        assert trial["chip"].startswith("colour") and "1 2 3 4" in trial["keys"]
        (card,) = trial["cards"]
        assert card["ground"] == trial["page_bg"], "the stimulus paints the whole page with its ground"
        assert len(re.findall(r'data-slot="(\d)"', card["html"])) == 4, "four squares, one slot each"
        assert card["html"].count(card["ground"]) == 0, "no square may be the ground colour"

    rows = scratch_log.read()
    assert [r["n"] for r in rows] == list(range(PAST_THE_COLOUR_RUN)), "trial numbers must be dense and ordered"
    colour_rows = [r for r in rows if r["mode"] == "discrimination"]
    assert [r["vision_n"] for r in colour_rows] == list(range(1, 9)), "each points at the vision row it made"
    assert all("snippet" not in r and "snippet_fresh" not in r for r in colour_rows), "no page was shown"
    assert [r["correct"] for r in colour_rows] == [True, False] + [True] * 6

    vision_rows = scratch_vision_log.read()
    assert [r["n"] for r in vision_rows] == list(range(9)), "one series across both surfaces"
    assert "rt_ms" not in vision_rows[0] and vision_rows[0]["generator"] == vision.NOTEBOOK_GENERATOR
    app_rows = vision_rows[1:]
    assert len(app_rows) == schedule.DISCRIMINATION_PER_BLOCK
    for row in app_rows:
        assert row["rt_ms"] == pytest.approx(1500.0) and row["paused"] is False
        assert row["surface"] == "app" and row["input_method"] == "mouse"
        assert row["generator"] == vision.APP_GENERATOR and row["size_px"] in (16, 10)
        assert row["ground_hex"] == served[FIRST_COLOUR + row["n"] - 1]["page_bg"], (
            "the row's ground was the one painted"
        )
    assert [r["correct"] for r in app_rows] == [True, False] + [True] * 6


def test_a_dark_ground_inside_a_day_block_gets_the_light_ink(
    client, scratch_log, scratch_vision_log, scratch_posterior
):
    """The colour arm paints the page with a ground from its own family, which is dark for
    four of the seven grounds, inside a block whose chrome ink was chosen for a light page.
    Sixteen vision trials already answered put the app's first colour run on the night
    ground; the block is still a day block."""
    seeded_vision_log(scratch_vision_log, scratch_posterior, vision.BLOCK)
    for _ in range(FIRST_COLOUR):
        answer_next(client, scratch_log, scratch_vision_log, scratch_posterior)
    trial = client.get(f"/api/trial/{FIRST_COLOUR}").json()
    assert (trial["mode"], trial["polarity"], trial["vision_n"]) == ("discrimination", "day", vision.BLOCK)
    assert trial["page_bg"] == dict(vision.GROUND_LIST)["night"], "the second ground of the family is the dark one"
    assert trial["chrome_ink"] == CHROME_INK["night"], "the ink follows the surround, not the block's polarity"
    assert chrome_contrast(trial) >= CHROME_FLOOR
    out = answer_next(client, scratch_log, scratch_vision_log, scratch_posterior)
    assert out["ok"] and scratch_vision_log.read()[-1]["ground_hex"] == trial["page_bg"]


@pytest.mark.parametrize("ground", [hex_ for _label, hex_ in vision.GROUND_LIST])
def test_the_chrome_is_legible_on_every_ground_the_colour_arm_can_paint(ground):
    from theme.server import chrome_ink_for

    ratio = float(wcag(hex_to_rgb([chrome_ink_for(ground)]), hex_to_rgb([ground]))[0])
    assert ratio >= CHROME_FLOOR, f"{chrome_ink_for(ground)} on {ground} is {ratio:.2f}:1"


@pytest.mark.parametrize("n,expected_polarity", [(0, "day"), (schedule.BLOCK, "night")])
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


def test_the_page_echoes_the_vision_numbering_and_answers_the_slots_with_four_keys():
    """Two things only the page can do for the colour arm, checked in the source because
    that is where they live: echo `vision_n` so the recorder can refuse an answer whose
    vision log moved on, and answer the four slots with four keys equidistant from the hand
    -- at glyph scale a square is a near-unclickable target, and a per-slot difference in
    motor cost is exactly what a guess drifts toward."""
    source = APP_JS.read_text()
    body = source.split("const body = {", 1)[1].split("};", 1)[0]
    assert "vision_n: trial.vision_n" in body, "the answer must carry the vision numbering it was shown"
    assert '["1", "2", "3", "4"].includes(k)' in source, "four keys, one per slot"
    assert "answer(parseInt(k, 10) - 1)" in source, "key 1 is slot 0"


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
