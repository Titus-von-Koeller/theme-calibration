"""Pause-only, cached evidence. No fit, palette application or measurement writes on read.

Refresh explicitly with ``python -m theme.stopping --refresh`` between sittings.
This first increment reports the existing verdict's observations but cannot certify
stopping: that verdict does not include predictive validation or sustained comfort.
"""

import argparse
import fcntl
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from . import paths

SCHEMA = 1
CACHE = paths.DATA_DIR / "stopping-summary.json"
NEXT_STEP = "Pause here; review the analysis before choosing whether to do another block."


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(log_paths=None, source=None):
    """Exact input identity, including both measurement logs and lived responses.

    Data changes invalidate rather than extrapolate. The candidate recipe is separate
    from the remaining analysis code, so changing the search never resembles learning.
    """
    source = source or Path(__file__).parent
    log_paths = log_paths or (paths.RESPONSE_LOG, paths.VISION_LOG, paths.LIVED_LOG)
    data = []
    for path in log_paths:
        try:
            data.append(_digest(path.read_bytes()))
        except FileNotFoundError:
            data.append(None)
    recipe = {"model.py", "space.py", "verdict.py"}
    code = {p.name: _digest(p.read_bytes()) for p in sorted(source.glob("*.py"))}
    lock = source.parent / "pixi.lock"
    code["pixi.lock"] = _digest(lock.read_bytes()) if lock.exists() else None
    return {
        "data": data,
        "candidates": {name: digest for name, digest in code.items() if name in recipe},
        "model": {name: digest for name, digest in code.items() if name not in recipe},
    }


def observation(verdict):
    """Only established Verdict fields; never its naive duels-to-decide extrapolation."""
    if verdict is None:
        return None
    progress = verdict.progress
    return {
        "duels": verdict.n_duels,
        "verdict": verdict.verdict,
        "leading_group_mass": verdict.lead,
        "alternatives": len(verdict.credible),
        "set_mass_target": 0.5,  # best_set's existing default, not a completion fraction
        "candidate_digest": _digest(json.dumps([list(map(float, t)) for t in verdict.thetas]).encode()),
        "progress": {k: progress[k] for k in ("duels", "back", "lead_then", "lead_now", "set_then", "set_now")}
        if progress
        else None,
        "reading_tradeoff": bool(verdict.legibility and verdict.legibility.champion_credibly_slower),
        "predictive_validation": "unavailable",
    }


def unknown(reason, **metadata):
    return {"status": "unknown", "recommendation": NEXT_STEP, "reason": reason, **metadata}


def read_summary(polarity, current, cache=CACHE, now=None):
    """Read and validate a bounded cache. Never imports or invokes fitting code."""
    if polarity not in ("day", "night"):
        return unknown("The page condition is unknown.")
    try:
        with cache.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return unknown("The saved analysis summary is invalid.")
        saved = json.loads(raw)
        if saved.get("schema") != SCHEMA:
            return unknown("The saved analysis summary uses a different format; refresh it between sittings.")
        timestamp = datetime.fromisoformat(saved["created"])
        age = ((now or datetime.now(UTC)) - timestamp).total_seconds()
        if age < 0:
            return unknown("The saved analysis time is in the future; check it before relying on it.")
        metadata = {"created": saved["created"], "age_seconds": age, "polarity": polarity}
        for key, reason in (
            ("model", "The analysis code changed; refresh the summary between sittings."),
            ("candidates", "The candidate recipe changed; the previous comparison no longer applies."),
            ("data", "Responses changed since this analysis; refresh it before deciding whether more are useful."),
        ):
            if saved["identity"][key] != current[key]:
                return unknown(reason, **metadata)
        item = saved["polarities"][polarity]
        if item is None:
            return unknown(f"The {polarity} analysis has too little evidence to report a verdict.", **metadata)
        progress = item["progress"]
        if progress is not None and not (
            all(type(progress[k]) is int and progress[k] >= 0 for k in ("duels", "back", "set_then", "set_now"))
            and 0 < progress["back"] <= progress["duels"]
            and all(0 <= progress[k] <= 1 for k in ("lead_then", "lead_now"))
        ):
            return unknown("The saved progress comparison is invalid.")
        # Validate the limited wire schema before presenting any saved number.
        if not (
            isinstance(item["candidate_digest"], str)
            and len(item["candidate_digest"]) == 64
            and 0 <= item["leading_group_mass"] <= 1
            and item["set_mass_target"] == 0.5
            and type(item["alternatives"]) is int
            and item["alternatives"] > 0
            and type(item["duels"]) is int
            and item["duels"] >= 0
            and type(item["reading_tradeoff"]) is bool
            and item["verdict"] in ("single", "plateau", "undecided")
        ):
            return unknown("The saved analysis summary is invalid.")
        reason = (
            "Predictive validity and sustained comfort are not established by this summary. "
            "A plateau alone cannot decide whether more trials help."
        )
        recommendation = NEXT_STEP
        if item["reading_tradeoff"]:
            recommendation = "Review the reading-speed tradeoff before doing more preference trials."
            reason = (
                "The existing verdict flags the champion as credibly slower than the fastest candidate. "
                "Predictive validity and sustained comfort still need review."
            )
        return {
            "status": "review",
            "recommendation": recommendation,
            "reason": reason,
            "observation": item,
            **metadata,
        }
    except OSError, ValueError, TypeError, KeyError, AttributeError:
        return unknown(
            "No usable analysis summary is available. Refresh it between sittings; "
            "more clicks are not automatically the next step."
        )


def refresh(cache=CACHE):
    """Explicit offline work; a concurrent producer is refused, not overwritten."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An analysis refresh is already running") from exc
        before = identity()
        from .publish import all_responses
        from .verdict import verdict_for

        rows = all_responses()
        polarities = {polarity: observation(verdict_for(rows, polarity)) for polarity in ("day", "night")}
        saved = {
            "schema": SCHEMA,
            "created": datetime.now(UTC).isoformat(),
            "identity": before,
            "polarities": polarities,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=cache.parent, prefix="stopping-", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(saved, allow_nan=False) + "\n")
            if identity() != before:
                raise RuntimeError("Inputs changed during analysis; summary was not written")
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", required=True)
    parser.parse_args()
    refresh()


if __name__ == "__main__":
    main()
