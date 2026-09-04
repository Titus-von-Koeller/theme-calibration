#!/usr/bin/env python
"""The trial instrument, as a web app that owns its own page.

Why this exists. The trials used to live in a marimo anywidget, and marimo tears the
widget down and rebuilds it on every answer. A timed psychophysics task cannot work that
way: the DOM it measures against has to outlive the measurement. The notebook version
needed three separate workarounds for not owning its mount -- reparenting to <body> to
escape marimo's stacking context, a page-owned persistent stage so loading placeholders
did not flash between trials, and a render-generation guard so a stale render could not
clobber a live one -- and still lost a race that left an empty full-screen stage over the
page (reproduced on the versions both before and after the styling change, so it was never
about styling).

Here the page is built once and never rebuilt. Trials arrive as JSON and only the contents
change. There is no teardown, no reparenting, no z-index war, and no skew token on the path
a click takes.

The model is untouched: this imports exactly the same `theme` package the analysis notebook
imports, so the search, the floors and the log format are the ones already calibrated.

    pixi run serve
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import responses, stimulus, trialspec
from .schedule import run_info, trial_for

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="theme trials")


# What the chip in the corner calls each arm: short, so the eye takes it in one hit.
ARM_LABEL = {"duel": "duel", "comprehension": "spot", "search": "find"}


def page_for(trial: dict) -> dict:
    """The code page a trial is shown on, with its width and length preferences."""
    return stimulus.snippet_for(
        trial["snippet"],
        trial.get("snippet_width"),
        trial.get("target_kind"),
        trial.get("snippet_lines"),
    )


def payload(n: int, answered: list[dict]) -> dict:
    """Everything the page needs to show trial n, and nothing it does not.

    The HTML for each candidate page is rendered here rather than in the browser because
    the tokenizer, the role colours and the floors all live in Python; the page's job is
    to display it, time it, and report a click.
    """
    trial = trial_for(n, answered)
    polarity, _arm, position, run_length = run_info(n, len(answered))
    page = page_for(trial)
    is_duel = trial["mode"] == "duel"
    prompt, cards, current_match = trialspec.stimulus_for(n, trial, page)
    return {
        "n": n,
        "mode": trial["mode"],
        "polarity": polarity,
        "is_duel": is_duel,
        "chip": f"{ARM_LABEL[trial['mode']]} · {polarity} page",
        "prompt_html": prompt,
        "cards": cards,
        "find_current": current_match,
        # A duel keeps the polarity's neutral surround, since painting the page with either
        # candidate's ground would advantage it; a single-card trial paints the page with
        # the theme under test, which is what a theme owning the screen actually looks like.
        "page_bg": (responses.DUEL_SURROUND[polarity] if is_duel else trial["theme_a"]["ground"]),
        "keys": "← →  or click" if is_duel else "space pauses",
        "progress": f"{position + 1} of {run_length}",
        # A gate only at the START of a run: one instruction serves the whole run, and no
        # click is spent re-reading it mid-stride.
        "gate": position == 0,
        "gate_text": trialspec.gate_text_for(trial["mode"], polarity, run_length),
    }


@app.get("/api/trial/{n}")
def api_trial(n: int) -> dict:
    return payload(n, responses.read_responses())


class Answer(BaseModel):
    n: int
    choice: int
    t_render: float
    t_click: float
    pauses: int = 0
    input_method: str = "mouse"


@app.post("/api/response")
def api_response(a: Answer) -> dict:
    """Append one answer, then hand back the next trial in the same round trip.

    The guard is the same one the notebook had and matters for the same reason: the trial
    is recomputed from the LOG at record time, never read from whatever the page happened
    to be holding, so a stale page cannot mis-record. A click for a trial that is no longer
    next is dropped rather than written to the wrong row.
    """
    answered = responses.read_responses()
    if a.n != len(answered):
        return {"ok": False, "reason": "stale", "next": payload(len(answered), answered)}
    trial = trial_for(a.n, answered)
    entry = responses.build_entry(a.n, trial, page_for(trial), a.model_dump())
    responses.append_response(entry)
    answered = [*answered, entry]
    return {"ok": True, "next": payload(len(answered), answered)}


@app.get("/api/status")
def api_status() -> dict:
    rows = responses.read_responses()
    duels = sum(1 for r in rows if r.get("mode") == "duel")
    return {"responses": len(rows), "duels": duels}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
