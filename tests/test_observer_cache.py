"""A cached observer must describe this calibration log, not just its row count."""

import json

import numpy as np
import pytest

from theme import observer


@pytest.fixture(autouse=True)
def small_grid(monkeypatch):
    """Exercise the real fitting arithmetic on a small grid, using synthetic rows only."""
    names = ("_PHI", "_W1", "_W2", "_BETA", "_LAM", "_TAU0", "_GL", "_GAMMA")
    for name in names:
        monkeypatch.setattr(observer, name, getattr(observer, name)[[0, -1]])
    monkeypatch.setattr(observer, "_SHAPE", (2,) * len(names))


def write_log(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def rows():
    return [
        {"base": "#808080", "odd_color": "#888080", "ground": "day", "correct": True},
        {"base": "#808080", "odd_color": "#808880", "ground": "night", "correct": True},
    ]


@pytest.mark.parametrize("changed_index", [0, 1])
def test_same_length_replacement_refits_every_changed_row(tmp_path, changed_index):
    path = tmp_path / "vision.jsonl"
    history = rows()
    write_log(path, history)
    before = observer.fit(path)
    history[changed_index]["correct"] = False
    write_log(path, history)

    after = observer.fit(path)
    fresh = observer.fit(path, cache=False)
    assert after._p == fresh._p
    assert not np.isclose(after.de_min_day, before.de_min_day)


def test_two_logs_in_one_directory_do_not_share_a_fit(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    history = rows()
    write_log(first, history)
    observer.fit(first)
    history[0]["correct"] = False
    write_log(second, history)
    assert observer.fit(second)._p == observer.fit(second, cache=False)._p


def test_unchanged_records_reuse_fit_despite_json_formatting(tmp_path, monkeypatch):
    path = tmp_path / "vision.jsonl"
    history = rows()
    write_log(path, history)
    expected = observer.fit(path)._p
    path.write_text("\n" + "\n".join(json.dumps(row, sort_keys=True) for row in history) + "\n\n")

    def unexpected_refit(*args, **kwargs):
        pytest.fail("unchanged calibration records should use their cached fit")

    monkeypatch.setattr(observer, "log_posterior", unexpected_refit)
    assert observer.fit(path)._p == expected


def test_legacy_cache_without_content_identity_is_rebuilt(tmp_path):
    path = tmp_path / "vision.jsonl"
    write_log(path, rows())
    expected = observer.fit(path)._p
    legacy = dict(expected)
    legacy.pop("log_sha256", None)
    legacy["de_min_day"] = -123.0
    (tmp_path / "observer-fit.json").write_text(json.dumps(legacy))
    assert observer.fit(path)._p == expected


@pytest.mark.parametrize("option", [{"force": True}, {"cache": False}])
def test_explicit_refit_still_bypasses_matching_cache(tmp_path, option):
    path = tmp_path / "vision.jsonl"
    write_log(path, rows())
    expected = observer.fit(path)._p
    poisoned = dict(expected, de_min_day=-123.0)
    cache_path = tmp_path / "observer-fit.json"
    cache_path.write_text(json.dumps(poisoned))
    assert observer.fit(path, **option)._p == expected
    assert json.loads(cache_path.read_text()) == (poisoned if option.get("cache") is False else expected)
