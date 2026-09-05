#!/usr/bin/env python
"""The trial instrument, as a web app that owns its own page.

Why this exists. The trials used to live in a notebook widget, and that notebook tears the
widget down and rebuilds it on every answer. A timed psychophysics task cannot work that
way: the DOM it measures against has to outlive the measurement. That version needed three
separate workarounds for not owning its mount -- reparenting to <body> to escape the host's
stacking context, a page-owned persistent stage so loading placeholders did not flash
between trials, and a render-generation guard so a stale render could not clobber a live
one -- and still lost a race that left an empty full-screen stage over the page (reproduced
on the versions both before and after the styling change, so it was never about styling).

Here the page is built once and never rebuilt. Trials arrive as JSON and only the contents
change. There is no teardown, no reparenting, no z-index war, and no skew token on the path
a click takes.

The model is untouched: this imports exactly the same `theme` package the analysis notebook
imports, so the search, the floors and the log format are the ones already calibrated.

    pixi run serve
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import paths, responses, stimulus, trialspec, vision
from .color import hex_to_rgb, wcag
from .schedule import run_info, trial_for

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def warm_the_model(_app: FastAPI):
    """Fit the model before the first click rather than during it.

    A fresh process pays about 3.3 s for the first trial: the reaction-time exponent is
    chosen by five-fold cross-validation over four candidate values, which is twenty
    Laplace fits, and it is memoised only once it has run. Paying that at boot costs
    nobody anything -- the server starts while the tab is still being opened -- whereas
    paying it on the first answer puts three seconds inside a measurement.
    """
    try:
        answered = responses.DEFAULT_LOG.read()
        payload(len(answered), answered, DEFAULT_VISION_LOG.read(), DEFAULT_POSTERIOR)
    except Exception as exc:  # a warm-up must never stop the app from serving
        print(f"warm-up skipped, so the first trial will pay for the fit: {exc!r}")
    yield


app = FastAPI(title="theme trials", lifespan=warm_the_model)


#: Ink for the instrument's own chrome, per polarity. Deliberately not part of any theme:
#: the chrome is furniture, and a theme under test must never colour the frame it is judged
#: in. Both clear 7:1 on their polarity's surround.
CHROME_INK = {"day": "#3a3634", "night": "#cfcac6"}

# What the chip in the corner calls each arm: short, so the eye takes it in one hit.
ARM_LABEL = {"duel": "duel", "comprehension": "spot", "search": "find", "discrimination": "colour"}

#: The vision log the fourth arm appends to, and the posterior kept warm beside it.
DEFAULT_VISION_LOG = responses.ResponseLog(paths.VISION_LOG)
DEFAULT_POSTERIOR = vision.POSTERIOR


def chrome_ink_for(page_bg: str) -> str:
    """The chrome ink that contrasts with THIS surround.

    A discrimination trial paints the page with its own ground, which may be dark inside a
    day block or light inside a night one, so the ink follows the surround rather than the
    polarity.
    """
    ground = hex_to_rgb([page_bg])
    return max(CHROME_INK.values(), key=lambda ink: float(wcag(hex_to_rgb([ink]), ground)[0]))


def page_for(trial: dict) -> dict | None:
    """The code page a trial is shown on, with its width and length preferences. None for
    an arm that shows no code."""
    if trial["mode"] == "discrimination":
        return None
    return stimulus.snippet_for(
        trial["snippet"],
        trial.get("snippet_width"),
        trial.get("target_kind"),
        trial.get("snippet_lines"),
    )


def _stage(n: int, trial: dict, page: dict) -> dict:
    """The stimulus itself: the instruction, and the card HTML to put on screen.

    Rendered here rather than in the browser because the tokenizer, the role colours and
    the floors all live in Python; the page's job is to display it, time it, and report a
    click.
    """
    prompt, cards, current_match = trialspec.stimulus_for(n, trial, page)
    return {
        "mode": trial["mode"],
        "is_duel": trial["mode"] == "duel",
        "prompt_html": prompt,
        "cards": cards,
        "find_current": current_match,
    }


def _chrome(trial: dict, polarity: str, position: int, run_length: int) -> dict:
    """The instrument's own furniture around the stimulus."""
    page_bg = responses.surround_for(trial, polarity)
    chip = (
        f"colour · {trial['vision']['size_px']} px · {trial['vision']['ground']}"
        if trial["mode"] == "discrimination"
        else f"{ARM_LABEL[trial['mode']]} · {polarity} page"
    )
    keys = {"duel": "← →  or click", "discrimination": "1 2 3 4  or click"}.get(trial["mode"], "space pauses")
    return {
        "polarity": polarity,
        "chip": chip,
        # The same rule the recorded row uses, from the same place, so the row cannot
        # describe a surround other than the one painted.
        "page_bg": page_bg,
        # The instrument's OWN chrome -- prompt, chip, progress, the gate -- has to contrast
        # with whatever surround this trial paints, and the surround flips with polarity and,
        # on the colour arm, with the ground under test. A stylesheet cannot know that, so it
        # is sent per trial. Getting it wrong is not cosmetic: a light chrome on a light day
        # page is an invisible instruction and an invisible begin button.
        "chrome_ink": chrome_ink_for(page_bg),
        "keys": keys,
        "progress": f"{position + 1} of {run_length}",
        # A gate only at the START of a run: one instruction serves the whole run, and no
        # click is spent re-reading it mid-stride.
        "gate": position == 0,
        "gate_text": trialspec.gate_text_for(trial["mode"], polarity, run_length),
    }


def payload(n: int, answered: list[dict], vision_answered: list[dict], posterior=None) -> dict:
    """Everything the page needs to show trial n, and nothing it does not.

    The keys here are a contract with static/app.js and are read nowhere else.
    """
    trial = trial_for(n, answered, vision_answered, posterior)
    # The log BEFORE trial n, the same prefix trial_for reads, so the run gate and the
    # progress readout agree with the arm and the polarity the trial was actually built at.
    polarity, _arm, position, run_length = run_info(n, answered[:n])
    page = page_for(trial)
    return {
        "n": n,
        **_stage(n, trial, page),
        **_chrome(trial, polarity, position, run_length),
    }


def get_log() -> responses.ResponseLog:
    """The response log this app writes to.

    A FastAPI dependency rather than a module import, so a test can point the whole app at
    a temporary file with `app.dependency_overrides[get_log]`. Without that seam the only
    honest end-to-end test would append to a year of real measurements.
    """
    return responses.DEFAULT_LOG


def get_vision_log() -> responses.ResponseLog:
    """The colour-discrimination log the fourth arm appends to; the same seam as get_log."""
    return DEFAULT_VISION_LOG


def get_posterior() -> vision.Posterior:
    """The observer posterior the fourth arm generates from. A test points it at a scratch
    directory, so its sidecar never lands beside the real log."""
    return DEFAULT_POSTERIOR


#: Annotated rather than a `Depends()` default: a call in a default argument is evaluated
#: once at import and is the usual source of surprising shared state.
LogDep = Annotated[responses.ResponseLog, Depends(get_log)]
VisionLogDep = Annotated[responses.ResponseLog, Depends(get_vision_log)]
PosteriorDep = Annotated[vision.Posterior, Depends(get_posterior)]


@app.get("/api/trial/{n}")
def api_trial(n: int, log: LogDep, vision_log: VisionLogDep, posterior: PosteriorDep) -> dict:
    return payload(n, log.read(), vision_log.read(), posterior)


class Answer(BaseModel):
    n: int
    choice: int
    t_render: float
    t_click: float
    pauses: int = 0
    input_method: str = "mouse"


@app.post("/api/response")
def api_response(a: Answer, log: LogDep, vision_log: VisionLogDep, posterior: PosteriorDep) -> dict:
    """Append one answer, then hand back the next trial in the same round trip.

    The guard matters for the reason it always did: the trial is recomputed from the LOG at
    record time, never read from whatever the page happened to be holding, so a stale page
    cannot mis-record. A click for a trial that is no longer next is dropped rather than
    written to the wrong row, and the page is handed the trial it should be showing.

    What makes that safe is that `trial_for` is a pure function of the trial number and the
    rows before it, so the trial rebuilt here is the same one the page was shown -- with no
    dependence on anything cached in this process, and therefore none on whether the server
    has restarted since the page loaded.
    """
    answered = log.read()
    vision_answered = vision_log.read()
    if a.n != len(answered):
        return {"ok": False, "reason": "stale", "next": payload(len(answered), answered, vision_answered, posterior)}
    trial = trial_for(a.n, answered, vision_answered, posterior)
    entry = responses.build_entry(a.n, trial, page_for(trial), a.model_dump())
    if trial["mode"] == "discrimination":
        # The measurement goes to the vision log, which is the observer's one input; the
        # app's own row is bookkeeping that keeps trial numbers dense and points at it.
        vision_row = responses.vision_entry(trial, a.model_dump(), entry["ts"])
        vision_log.append(vision_row)
        vision_answered = [*vision_answered, vision_row]
    log.append(entry)
    answered = [*answered, entry]
    return {"ok": True, "next": payload(len(answered), answered, vision_answered, posterior)}


@app.get("/api/status")
def api_status(log: LogDep) -> dict:
    rows = log.read()
    duels = sum(1 for row in rows if row.get("mode") == "duel")
    return {"responses": len(rows), "duels": duels}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
