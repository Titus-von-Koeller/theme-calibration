"""The response log: reading it, building one record, appending it.

Append-only, one JSON object per line, so concurrent sessions interleave and never
overwrite. Every row carries the whole stimulus -- both thetas, both themes, the surround,
the surface, the pixel size, which side was shown and both timestamps -- because a
measurement whose conditions were not written down cannot be re-analysed later, and this
log has already been re-read under three successive models.

The log is an OBJECT holding a path, not a module-level global. That is what lets a test
exercise the whole click path against a temporary file: a suite that could only run
against the real log would either be read-only and shallow, or would append junk to a
year of measurements. Neither is acceptable, and the choice between them is a design
smell, not a fact of life.

Building a record takes the trial and the page as arguments rather than fetching them, so
it is a pure function of its inputs and can be tested without a server at all. Deciding
WHICH trial a click belongs to is the caller's job, and the caller must derive it from
this log rather than from whatever the page was holding -- otherwise a stale page writes
its answer onto somebody else's row.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from . import paths, vision
from .observer import MODEL_VERSION as OBSERVER_MODEL
from .space import DE_MIN, READING_SIZE_PX, VISION_N, floor_regime, separation_floor
from .trialspec import rng_for

# The neutral surround a duel is judged against, per polarity. A duel keeps this rather
# than taking either candidate's ground, which would advantage that candidate.
DUEL_SURROUND = {"day": "#d8d2cf", "night": "#14161c"}


class ResponseLog:
    """An append-only JSONL file of answered trials."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> list[dict]:
        """Every response recorded so far, oldest first; blank lines skipped.

        A missing file is an empty log rather than an error, which is what a fresh checkout
        with no data/ directory needs.
        """
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def __len__(self) -> int:
        return len(self.read())


#: The real log. Injected rather than imported wherever it can be, so tests get their own.
DEFAULT_LOG = ResponseLog(paths.RESPONSE_LOG)


def surround_for(trial: dict, polarity: str) -> str:
    """The colour the whole page is painted for this trial.

    A duel keeps the polarity's neutral surround, since painting the page with either
    candidate's ground would advantage it; a single-card trial paints the page with the
    theme under test, which is what a theme owning the screen actually looks like.

    Lives here, beside DUEL_SURROUND, because both the page and the recorded row need it
    and they have to agree: a row whose `page_bg` disagreed with the surround that was
    actually painted would describe a stimulus nobody saw.
    """
    if trial["mode"] == "duel":
        return DUEL_SURROUND[polarity]
    return trial["theme_a"]["ground"]


def _stimulus_fields(trial: dict, page: dict | None, reported: dict) -> dict:
    """What was on screen, and when. Shared by every arm; the page fields only where a
    code page was shown."""
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": trial["mode"],
        "kind": trial["kind"],
        "polarity": trial["polarity"],
        **_page_fields(page),
        "surface": trial.get("surface", "editor"),
        "code_px": trial["code_px"],
        **_provenance_fields(trial),
        "theta_a": trial["theta_a"],
        "theme_a": trial["theme_a"],
        "page_bg": surround_for(trial, trial["polarity"]),
        "input_method": reported.get("input_method", "mouse"),
        # rt_ms runs from the LAST reveal -- the first render, or a resume after a pause.
        # A trial that was ever paused is flagged so its time is read as a near-tie
        # downstream rather than as a very slow decision.
        "rt_ms": round(reported["t_click"] - reported["t_render"], 1),
        "t_render": round(reported["t_render"], 1),
        "t_click": round(reported["t_click"], 1),
        "paused": reported.get("pauses", 0) > 0,
    }


def _provenance_fields(trial: dict) -> dict:
    """Which fit chose this stimulus, and which floors it had to clear to exist.

    Tightening a threshold silently re-bases every past duel: the candidate pool is carved
    by the separation floor, and that floor moves whenever the observer is refit on a
    longer vision log. A row recorded under a 6.46 dE floor and one recorded after the
    floor moves are answers to different questions, and nothing in the log said so -- pixel
    size has ridden along since the beginning for exactly this reason, and the thresholds
    that decide which themes EXIST had no such stamp.

    Written per row rather than reconstructed later, because the fit and the floors are
    recoverable from the code and the logs only for as long as neither has moved, which is
    precisely the case where the stamp is not needed. Old rows are not rewritten: what they
    were taken under is genuinely unknown, and inventing it is worse than the gap.

    `fit_fingerprint` is required, never defaulted. A reader's default once repaired a
    missing key in a writer for a whole corpus, and a provenance field that guesses is the
    worst place for that to happen.
    """
    polarity = trial["polarity"]
    floor, _how = separation_floor(polarity, READING_SIZE_PX)
    return {
        "provenance": {
            "fit": trial["fit_fingerprint"],
            "observer": {
                "model": OBSERVER_MODEL,
                "n_trials": VISION_N,
                "de_min": round(float(DE_MIN[polarity]), 3),
            },
            # The floor as it stood for THIS row, in the units the space is carved in, plus
            # the word for which rule produced it. The prose `separation_floor` returns is a
            # paragraph; a row wants the word.
            "floor": {"separation": round(float(floor), 3), "regime": floor_regime()},
        }
    }


def _page_fields(page: dict | None) -> dict:
    """Which code page was shown, and whether it was a first showing.

    Absent, not defaulted, for an arm that shows no code: a reader's default of
    `fresh=True` once repaired a missing key in a writer, so every fallback page was logged
    as a first showing. A row with no page must not carry a freshness flag at all.
    """
    if page is None:
        return {}
    return {
        "snippet": page["id"],
        "snippet_hash": page.get("hash"),
        "snippet_kind": page.get("kind"),
        "snippet_fresh": bool(page.get("fresh", True)),
        "target_kind": page.get("target_kind"),
    }


def _duel_fields(trial: dict, page: dict, reported: dict, rng) -> dict:
    clicked_side = reported["choice"]  # 0 = the left card
    return {
        "theta_b": trial["theta_b"],
        "theme_b": trial["theme_b"],
        "swap": trial["swap"],
        "find_current": rng.choice(page["ident_ids"]) if page["ident_ids"] else None,
        # 0 = theme_a won. The page reports the SIDE it clicked; the swap flag turns that
        # back into which theme it was, so utility stays defined over themes while the side
        # stays available for fitting the position bias.
        "choice": (1 - clicked_side) if trial["swap"] else clicked_side,
    }


def _comprehension_fields(page: dict, reported: dict, rng) -> dict:
    target_span = rng.choice(page["fn_ids"])
    target_name = page["spans"][target_span]["text"]
    # Any occurrence of the name counts. The task was to find the function, not one
    # particular character range, so another call site of the same name answers the
    # question that was actually asked.
    accepted = [
        i for i, span in enumerate(page["spans"]) if span["role"] == "function" and span["text"] == target_name
    ]
    return {
        "target": target_span,
        "target_text": target_name,
        "clicked": reported["choice"],
        "correct": reported["choice"] in accepted,
    }


def _discrimination_fields(trial: dict, reported: dict) -> dict:
    """The bookkeeping half of a discrimination answer. The measurement itself is the row
    `vision_entry` builds for the vision log; this row keeps the app's trial numbering dense
    and says where to find it."""
    stimulus = trial["vision"]
    return {
        "vision_n": trial["vision_n"],
        "size_px": stimulus["size_px"],
        "ground_hex": stimulus["ground_hex"],
        "choice": reported["choice"],
        "correct": reported["choice"] == stimulus["odd_position"],
    }


def vision_entry(trial: dict, reported: dict, ts: str) -> dict:
    """The row a discrimination answer appends to the VISION log: the generator's own record
    plus the clock this surface exists to provide."""
    timing = {
        "surface": "app",
        "input_method": reported.get("input_method", "mouse"),
        "rt_ms": round(reported["t_click"] - reported["t_render"], 1),
        "paused": reported.get("pauses", 0) > 0,
    }
    return vision.build_entry(trial["vision_n"], trial["vision"], reported["choice"], ts, timing)


def _search_fields(trial: dict, page: dict, reported: dict, rng) -> dict:
    target_span = rng.choice(page["ident_ids"])
    return {
        "target": target_span,
        "clicked": reported["choice"],
        "correct": reported["choice"] == target_span,
        "salience": trial["theme_a"]["salience"],
        "find_sal_theta": trial["theta_a"][8],
    }


def build_entry(trial_number: int, trial: dict, page: dict, reported: dict) -> dict:
    """The record for one answered trial, ready to append.

    `reported` is what the surface knows and nothing else can: which token was clicked,
    when the page rendered, when it was clicked, whether it was ever paused. Everything
    else is recomputed here from the trial itself.
    """
    rng = rng_for(trial_number)
    if trial["mode"] == "discrimination":
        return {
            "n": trial_number,
            **_stimulus_fields(trial, None, reported),
            **_discrimination_fields(trial, reported),
        }
    per_arm = {
        "duel": lambda: _duel_fields(trial, page, reported, rng),
        "comprehension": lambda: _comprehension_fields(page, reported, rng),
        "search": lambda: _search_fields(trial, page, reported, rng),
    }[trial["mode"]]
    return {"n": trial_number, **_stimulus_fields(trial, page, reported), **per_arm()}
