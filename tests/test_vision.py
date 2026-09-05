"""Does the discrimination generator do what its protocol declares?

Pure in the trial number and the log before it; a declared anchor share; exact slot
balance every four trials; grounds and sizes in blocks; and a record that says whether the
answer was right. The app and the notebook both build trials and rows through this module,
so these are the properties both surfaces inherit.
"""

import collections
import json

import pytest

from theme import vision
from theme.vision import (
    ANCHOR_SHARE,
    APP_GENERATOR,
    APP_SIZES,
    BLOCK,
    GROUND_LIST,
    SIZES,
    Posterior,
    block_for,
    build_entry,
    odd_position_for,
)


@pytest.fixture(scope="module")
def scratch_posterior(tmp_path_factory):
    """A posterior whose sidecar lives in a scratch directory, never beside the real log."""
    return Posterior(sidecar_dir=tmp_path_factory.mktemp("posterior"))


def test_slots_are_balanced_exactly_every_four_trials():
    for start in range(0, 400, 4):
        assert sorted(odd_position_for(n) for n in range(start, start + 4)) == [0, 1, 2, 3]


def test_blocks_hold_a_ground_and_rotate_size_after_every_ground():
    grounds_per_size = len(GROUND_LIST)
    first_block = {block_for(n)[:2] for n in range(BLOCK)}
    assert len(first_block) == 1, "a block holds one ground so the eye can adapt to it"
    sizes_in_order = [block_for(block * BLOCK)[2] for block in range(grounds_per_size * len(SIZES))]
    assert sizes_in_order == [size for size in SIZES for _ in range(grounds_per_size)]


def test_the_app_serves_only_the_glyph_scale():
    sizes = {block_for(n, APP_SIZES)[2] for n in range(BLOCK * len(GROUND_LIST) * len(APP_SIZES))}
    assert sizes == set(APP_SIZES) and 104 not in sizes


def test_a_trial_is_a_pure_function_of_its_number_and_history(scratch_posterior):
    first = vision.trial_for(3, [], posterior=scratch_posterior)
    vision.TRIAL_MEMO.clear()
    again = vision.trial_for(3, [], posterior=scratch_posterior)
    assert first == again
    assert first["odd_color"] != first["base"], "the odd square must differ or there is no question"
    assert first["odd_position"] == odd_position_for(3)


def test_anchors_are_a_declared_minority(scratch_posterior):
    """Roughly the declared 5%: with a fitted slope and lapse, easy trials carry almost no
    information, so the share is kept minimal and it is checked rather than assumed."""
    kinds = collections.Counter(vision.trial_for(n, [], posterior=scratch_posterior)["kind"] for n in range(200))
    share = kinds["anchor"] / 200
    assert abs(share - ANCHOR_SHARE) < 0.05, kinds


def test_the_app_generator_is_stamped_on_its_trials(scratch_posterior):
    trial = vision.trial_for(7, [], sizes=APP_SIZES, generator=APP_GENERATOR, posterior=scratch_posterior)
    assert trial["generator"] == APP_GENERATOR and trial["size_px"] in APP_SIZES


def test_the_record_says_whether_the_answer_was_right(scratch_posterior):
    trial = vision.trial_for(0, [], posterior=scratch_posterior)
    right = build_entry(0, trial, trial["odd_position"], "2026-09-05T10:00:00+00:00")
    wrong = build_entry(0, trial, (trial["odd_position"] + 1) % 4, "2026-09-05T10:00:00+00:00")
    assert right["correct"] and not wrong["correct"]
    for key in ("base", "odd_color", "ground_hex", "size_px", "gap_px", "generator", "odd_position"):
        assert right[key] == trial[key], f"the row must carry {key}, or the trial cannot be re-analysed"
    timed = build_entry(0, trial, 1, "2026-09-05T10:00:00+00:00", timing={"rt_ms": 812.0, "surface": "app"})
    assert timed["rt_ms"] == 812.0 and timed["surface"] == "app"
    assert "rt_ms" not in right, "a notebook row carries no clock rather than a fake one"


def test_a_foreign_log_never_overwrites_the_sidecar(tmp_path):
    """The sidecar caches minutes of refit for the real log. A shorter log -- a test's
    scratch log, a truncated copy -- must not replace it with its own posterior."""
    posterior = Posterior(sidecar_dir=tmp_path)
    posterior.sidecar_meta.write_text(json.dumps({"n": 748, "cells": posterior.n_cells}))
    posterior.sidecar.write_bytes(b"real")
    Posterior(sidecar_dir=tmp_path).logp_for([])
    assert posterior.sidecar.read_bytes() == b"real", "a shorter log overwrote the longer log's sidecar"
    assert json.loads(posterior.sidecar_meta.read_text())["n"] == 748


def test_metadata_without_its_grid_still_guards_the_sidecar(tmp_path):
    """The grid is gitignored and its metadata is tracked, so a worktree or a fresh clone
    has the metadata alone. That is the case in which the guard was skipped and the suite
    rewrote the real metadata to "n": 0 (2026-09-05)."""
    posterior = Posterior(sidecar_dir=tmp_path)
    posterior.sidecar_meta.write_text(json.dumps({"n": 748, "cells": posterior.n_cells}))
    posterior.logp_for([])
    assert json.loads(posterior.sidecar_meta.read_text())["n"] == 748, "an empty log rewrote the metadata"
    assert not posterior.sidecar.exists(), "an empty log has nothing to cache"


def test_a_missing_grid_is_rebuilt_from_the_metadata_length_onward(tmp_path):
    """Metadata claiming a length the binary cannot back must be ignored for LOADING and
    honoured for WRITING: a log at least as long refits from zero and then replaces both."""
    posterior = Posterior(sidecar_dir=tmp_path)
    posterior.sidecar_meta.write_text(json.dumps({"n": 1, "cells": posterior.n_cells}))
    trial = vision.trial_for(0, [], posterior=posterior)
    row = build_entry(0, trial, trial["odd_position"], "2026-09-05T10:00:00+00:00")
    logp = Posterior(sidecar_dir=tmp_path).logp_for([row])
    assert posterior.sidecar.exists() and json.loads(posterior.sidecar_meta.read_text())["n"] == 1
    assert float(abs(logp).max()) > 0, "one answered row must move the posterior off zero"
