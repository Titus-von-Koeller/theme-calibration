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

from theme import responses, vision
from theme import server as theme_server
from theme.server import app, get_log, get_posterior, get_vision_log

#: A flat pool of feasible points, so the search and the likelihood are exercised without
#: the colour layer's floors deciding which candidates exist. The floors have their own
#: constraints and their own tests; mixing them in here would make a failure ambiguous
#: between "the search is wrong" and "no candidate was legible".
POOL_THETA = np.random.default_rng(0xA55).random((512, 9))


@pytest.fixture(scope="session")
def search_model():
    """The real model, with the colour layer stubbed out.

    The stub replaces `realize_many`, which is the seam the search actually calls -- it
    realizes candidates in batches, because colour-science costs the same for one colour as
    for sixty. Stubbing the older per-theme `realize` here silently did nothing once that
    changed, and the suite caught it as a fixture error rather than as a wrong number,
    which is the good outcome.

    Session-scoped and mutating module globals deliberately: `theme.model` reads these at
    call time and every test in this suite wants the same stub. Restoring them afterwards
    keeps the process honest for anything running later in the same session.
    """
    from theme import model

    original = (model.POOL, model.realize_many, model.prior_mean)
    model.POOL = {"day": [(t, {"ok": True}) for t in POOL_THETA], "night": []}
    model.realize_many = lambda thetas, polarity: [{"ok": True} for _ in np.atleast_2d(thetas)]
    model.prior_mean = lambda theta, polarity: 0.0
    yield model
    model.POOL, model.realize_many, model.prior_mean = original


@pytest.fixture(scope="session", autouse=True)
def no_test_reaches_the_real_posterior(tmp_path_factory):
    """Every process-wide default posterior points at a scratch directory for the whole
    suite.

    The observer posterior caches its grid beside the real vision log. A test that builds
    a colour trial without naming a posterior -- the schedule tests do, walking a block
    ahead of the real log -- falls back to the module default, and the first run of this
    suite in a worktree rewrote the tracked sidecar metadata to "n": 0 that way. The guard
    in `Posterior` has been fixed; this makes the property structural as well, so no test
    can write beside the measurements whatever it forgets to pass.
    """
    scratch = vision.Posterior(sidecar_dir=tmp_path_factory.mktemp("default-posterior"))
    originals = (vision.POSTERIOR, theme_server.DEFAULT_POSTERIOR)
    vision.POSTERIOR = theme_server.DEFAULT_POSTERIOR = scratch
    yield
    vision.POSTERIOR, theme_server.DEFAULT_POSTERIOR = originals


@pytest.fixture
def scratch_log(tmp_path) -> responses.ResponseLog:
    """An empty response log in a temporary directory.

    Never the real one. A suite that wrote to `data/aesthetics-responses.jsonl` would
    append junk to a year of measurements, and a suite that only ever read it could not
    test recording at all.
    """
    return responses.ResponseLog(tmp_path / "responses.jsonl")


@pytest.fixture
def scratch_vision_log(tmp_path) -> responses.ResponseLog:
    """An empty colour-discrimination log, for the same reason: the colour arm appends its
    measurement to the vision log, and the real one is 748 answered trials."""
    return responses.ResponseLog(tmp_path / "vision.jsonl")


@pytest.fixture
def scratch_posterior(tmp_path) -> vision.Posterior:
    """An observer posterior whose sidecar lives beside the scratch logs, never beside the
    real ones. Fresh per test: the posterior is stateful over one log."""
    return vision.Posterior(sidecar_dir=tmp_path / "posterior")


@pytest.fixture
def client(scratch_log, scratch_vision_log, scratch_posterior):
    """A TestClient whose app reads and writes the scratch logs and fits the scratch
    posterior -- all three seams, so the colour arm can be answered end to end."""
    from fastapi.testclient import TestClient

    app.dependency_overrides[get_log] = lambda: scratch_log
    app.dependency_overrides[get_vision_log] = lambda: scratch_vision_log
    app.dependency_overrides[get_posterior] = lambda: scratch_posterior
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def correct_choice(
    trial_number: int,
    answered: list[dict],
    vision_answered: list[dict] = (),
    posterior: vision.Posterior | None = None,
) -> int:
    """Which token id -- or, on the colour arm, which slot -- answers trial `trial_number`.

    Derived the same way the recorder derives it -- the per-trial RNG seeded from the
    trial number alone, or the vision generator's slot permutation -- which is exactly the
    property that lets a response be recorded from the log rather than from whatever the
    page was holding. A test that recomputed it some other way would be testing its own
    arithmetic.
    """
    from theme import trialspec
    from theme.schedule import trial_for
    from theme.server import page_for

    trial = trial_for(trial_number, answered, vision_answered, posterior)
    if trial["mode"] == "discrimination":
        return trial["vision"]["odd_position"]
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
