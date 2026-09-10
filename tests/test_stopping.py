"""Cached evidence is advisory, never a new model or an extra response operation."""

import fcntl
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from theme import stopping

NOW = datetime(2026, 9, 10, tzinfo=UTC)
IDENTITY = {"data": ["a", "b", None], "model": {"observer.py": "v2"}, "candidates": {"model.py": "v1"}}


def verdict(tradeoff=False):
    return SimpleNamespace(
        n_duels=40,
        verdict="plateau",
        lead=0.3,
        credible=[1, 2, 3],
        thetas=[[0.1] * 9],
        progress={
            "duels": 80,
            "back": 25,
            "lead_then": 0.4,
            "lead_now": 0.3,
            "set_then": 5,
            "set_now": 3,
            "duels_to_decide": 999,
        },
        legibility=SimpleNamespace(champion_credibly_slower=tradeoff),
    )


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "summary.json"
    saved = {
        "schema": stopping.SCHEMA,
        "identity": IDENTITY,
        "created": NOW.isoformat(),
        "polarities": {"day": stopping.observation(verdict()), "night": None},
    }
    path.write_text(json.dumps(saved))
    return path


def test_valid_snapshot_is_review_not_stopping_or_completion(cache):
    result = stopping.read_summary("day", IDENTITY, cache, NOW + timedelta(days=20))
    assert result["status"] == "review"
    assert result["age_seconds"] == 20 * 86400
    assert "sustained comfort" in result["reason"]
    assert result["observation"]["set_mass_target"] == 0.5
    assert "duels_to_decide" not in json.dumps(result)
    assert stopping.read_summary("night", IDENTITY, cache, NOW)["status"] == "unknown"


@pytest.mark.parametrize(
    "key,reason",
    [("data", "Responses changed"), ("model", "analysis code changed"), ("candidates", "candidate recipe changed")],
)
def test_each_changed_input_is_unknown(cache, key, reason):
    result = stopping.read_summary("day", {**IDENTITY, key: "changed"}, cache, NOW)
    assert result["status"] == "unknown" and reason in result["reason"]
    assert "observation" not in result


@pytest.mark.parametrize("content", ["", "null", "[]", "{}", "x" * 65537])
def test_missing_malformed_and_oversized(cache, content):
    cache.write_text(content)
    assert stopping.read_summary("day", IDENTITY, cache, NOW)["status"] == "unknown"
    cache.unlink()
    assert stopping.read_summary("day", IDENTITY, cache, NOW)["status"] == "unknown"


def test_future_time_unknown(cache):
    assert stopping.read_summary("day", IDENTITY, cache, NOW - timedelta(seconds=1))["status"] == "unknown"


def test_existing_speed_warning_takes_precedence_without_inventing_threshold(cache):
    data = json.loads(cache.read_text())
    data["polarities"]["day"] = stopping.observation(verdict(tradeoff=True))
    cache.write_text(json.dumps(data))
    result = stopping.read_summary("day", IDENTITY, cache, NOW)
    assert "reading-speed tradeoff" in result["recommendation"]
    assert result["status"] == "review"  # even a concentrated/contradictory fit cannot certify stopping
    assert "Predictive validity" in result["reason"]


def test_malformed_progress_unknown(cache):
    data = json.loads(cache.read_text())
    data["polarities"]["day"]["progress"]["back"] = 100
    cache.write_text(json.dumps(data))
    assert stopping.read_summary("day", IDENTITY, cache, NOW)["status"] == "unknown"


def test_refresh_uses_both_polarities_and_refuses_changed_data(cache, monkeypatch):
    from theme import publish
    from theme import verdict as module

    monkeypatch.setattr(publish, "all_responses", lambda: [])
    seen = []
    monkeypatch.setattr(module, "verdict_for", lambda rows, p: seen.append(p) or verdict())
    identities = iter([IDENTITY, {**IDENTITY, "data": "changed"}])
    monkeypatch.setattr(stopping, "identity", lambda: next(identities))
    before = cache.read_bytes()
    with pytest.raises(RuntimeError, match="Inputs changed"):
        stopping.refresh(cache)
    assert cache.read_bytes() == before and seen == ["day", "night"]
    assert not list(cache.parent.glob("stopping-*.tmp"))
    monkeypatch.setattr(stopping, "identity", lambda: IDENTITY)
    stopping.refresh(cache)
    assert stopping.read_summary("night", IDENTITY, cache)["status"] == "review"


def test_concurrent_refresh_is_refused_before_fitting(cache):
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            stopping.refresh(cache)


def test_pause_endpoint_never_builds_a_trial_or_refreshes(client, monkeypatch):
    from theme import server

    monkeypatch.setattr(server, "payload", lambda *args: pytest.fail("trial work in summary request"))
    monkeypatch.setattr(stopping, "refresh", lambda *args: pytest.fail("fit in summary request"))
    monkeypatch.setattr(stopping, "read_summary", lambda *args: stopping.unknown("fixture missing"))
    assert client.get("/api/stopping/day").json()["status"] == "unknown"


def test_identity_covers_each_log_candidate_recipe_and_locked_environment(tmp_path):
    source = tmp_path / "theme"
    source.mkdir()
    (source / "model.py").write_text("candidate recipe")
    (source / "observer.py").write_text("observer model")
    lock = tmp_path / "pixi.lock"
    lock.write_text("environment one")
    logs = [tmp_path / name for name in ("response", "vision", "lived")]
    baseline = stopping.identity(logs, source)
    logs[2].write_text("lived response")
    assert stopping.identity(logs, source)["data"] != baseline["data"]
    (source / "model.py").write_text("new candidates")
    assert stopping.identity(logs, source)["candidates"] != baseline["candidates"]
    lock.write_text("environment two")
    assert stopping.identity(logs, source)["model"] != baseline["model"]
