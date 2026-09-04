"""Shared fixtures.

Two ideas run through this suite.

The first is that a statistical instrument has to be tested by RECOVERY: plant a truth,
run the machinery, and check the truth comes back. Every number the instrument prints
looks like a measurement, so "it ran without raising" proves nothing about whether it
measures anything.

The second is that the parts which decide what Titus SEES have to be tested through the
real path -- request in, record on disk, next trial out -- because the failure that
prompted this project's restructuring was invisible to every unit test that existed: each
piece worked, and the trial still vanished from the screen.
"""

import numpy as np
import pytest

from theme import responses
from theme.server import app, get_log

#: A flat pool of feasible points, so the search and the likelihood are exercised without
#: the colour layer's floors deciding which candidates exist. The floors have their own
#: constraints and their own tests; mixing them in here would make a failure ambiguous
#: between "the search is wrong" and "no candidate was legible".
POOL_THETA = np.random.default_rng(0xA55).random((512, 9))


@pytest.fixture(scope="session")
def search_model():
    """The real model, with the colour layer stubbed out.

    Session-scoped and mutating module globals is deliberate: `theme.model` reads POOL and
    realize at call time, and every test in this suite wants the same stub. Restoring them
    afterwards keeps the process honest for anything that runs later in the same session.
    """
    from theme import model

    original = (model.POOL, model.realize, model.prior_mean)
    model.POOL = {"day": [(t, {"ok": True}) for t in POOL_THETA], "night": []}
    model.realize = lambda theta, polarity: {"ok": True}
    model.prior_mean = lambda theta, polarity: 0.0
    yield model
    model.POOL, model.realize, model.prior_mean = original


@pytest.fixture
def scratch_log(tmp_path) -> responses.ResponseLog:
    """An empty response log in a temporary directory.

    Never the real one. A suite that wrote to `data/aesthetics-responses.jsonl` would
    append junk to a year of measurements, and a suite that only ever read it could not
    test recording at all.
    """
    return responses.ResponseLog(tmp_path / "responses.jsonl")


@pytest.fixture
def client(scratch_log):
    """A TestClient whose app writes to the scratch log."""
    from fastapi.testclient import TestClient

    app.dependency_overrides[get_log] = lambda: scratch_log
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def correct_choice(trial_number: int, answered: list[dict]) -> int:
    """Which token id answers trial `trial_number` correctly.

    Derived the same way the recorder derives it -- the per-trial RNG seeded from the
    trial number alone -- which is exactly the property that lets a response be recorded
    from the log rather than from whatever the page was holding. A test that recomputed it
    some other way would be testing its own arithmetic.
    """
    from theme import trialspec
    from theme.schedule import trial_for
    from theme.server import page_for

    trial = trial_for(trial_number, answered)
    page = page_for(trial)
    rng = trialspec.rng_for(trial_number)
    if trial["mode"] == "duel":
        return 0
    if trial["mode"] == "comprehension":
        return int(rng.choice(page["fn_ids"]))
    return int(rng.choice(page["ident_ids"]))


def report(trial_number: int, choice: int, *, ms: float = 1500.0, pauses: int = 0) -> dict:
    """What the page posts back for one answered trial."""
    return {
        "n": trial_number,
        "choice": choice,
        "t_render": 1000.0,
        "t_click": 1000.0 + ms,
        "pauses": pauses,
        "input_method": "mouse",
    }
