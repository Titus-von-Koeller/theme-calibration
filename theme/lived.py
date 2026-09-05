"""Record what living in a theme was like, as a duel the model already understands.

    pixi run lived -- current      # today's applied theme beat the one before it
    pixi run lived -- previous     # the one before it was better
    pixi run lived -- --note "..." current
    pixi run lived -- baseline     # record the hand-chosen layer as the first application

A duel in the instrument is a four-second look at two pages of code. Living in a theme is
a different measurement of the same latent utility: fatigue over hours, prose in the chat
panel, the terminal, the real font rasteriser -- everything the trial surface cannot show.
The instrument already found that snap judgements favour the brightest page allowed
(day duels put the champion at the light wall of the space) while the standing preference,
learned by living in it, is a paper walked DOWN from near-white because it tires. Only a
lived verdict can arbitrate that, so lived verdicts are recorded as duels between the two
most recently applied themes of a polarity, with `surface: "vscode"`, no clock, and the
neutral slope. The factor test the analysis already runs then asks the right question --
does the optimum move between the browser and the editor -- with no new machinery.

Weighted as one duel each, deliberately. A day of living is surely worth more than one
click, but by how much is a modelling claim with no calibration behind it; the honest
first version lets the surface test say whether the two disagree before anyone decides how
to reconcile them.
"""

import argparse
import json
from datetime import UTC, datetime

from . import paths
from .responses import ResponseLog
from .space import ANCHORS

#: The palettes Titus lived in before any measured one was applied: Horizon's own accents
#: with the hand-reasoned override layer of dotfiles settings.jsonc as it stood at commit
#: 41b4530^ (2026-09-05), the day layer's values pixel-measured on 2026-09-02. Not points in
#: theme space -- the lived-duel row carries these hexes as what was seen and a fitted theta
#: (theme.inverse) as what the model reads, with the fit distance beside it.
HAND_PALETTES = {
    "day": {
        "ground": "#f2e4e0",
        "keyword": "#8a31b9",
        "function": "#15646a",
        "string": "#a13a06",
        "ink": "#5c332f",
        "comment": "#665f5b",
        "punct": "#5e5651",
        "find_fill": "#f2ff92",
    },
    "night": {
        "ground": "#1c1e26",
        "keyword": ANCHORS["night"]["keyword"],
        "function": ANCHORS["night"]["function"],
        "string": ANCHORS["night"]["string"],
        "ink": "#c6bec6",
        "comment": "#82858f",
        "punct": "#8a8da0",
        "find_fill": None,
    },
}

#: When the hand layer was applied, per the ledger comments in settings.jsonc. Earlier than
#: any measured application, so it sorts as the first one.
HAND_LAYER_APPLIED = "2026-09-02T00:00:00+00:00"


def latest_applications(applied_rows, polarity, count=2):
    """The most recent `count` distinct palettes applied for one polarity, newest first.

    By timestamp, not by position in the file: a baseline is appended after the palette
    it precedes in time.
    """
    seen, picked = set(), []
    for row in sorted(applied_rows, key=lambda row: row["ts"], reverse=True):
        if row["polarity"] != polarity:
            continue
        key = tuple(row["palette"]["theta"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(row)
        if len(picked) == count:
            break
    return picked


def lived_duel(current, previous, preferred, note=None):
    """One response row: the current application against the previous one.

    theta_a is the current theme, theta_b the previous; `choice` follows the log's
    convention (0 = theme_a won). No rt_ms, because there was no clock; the fit reads a
    clockless duel at the neutral slope.
    """
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "duel",
        "kind": "lived",
        "polarity": current["polarity"],
        "surface": "vscode",
        "theta_a": current["palette"]["theta"],
        "theta_b": previous["palette"]["theta"],
        "theme_a": {k: v for k, v in current["palette"].items() if k not in ("provenance", "theta")},
        "theme_b": {k: v for k, v in previous["palette"].items() if k not in ("provenance", "theta")},
        "applied_a": current["ts"],
        "applied_b": previous["ts"],
        "swap": False,
        "choice": 0 if preferred == "current" else 1,
        "rt_ms": None,
        "paused": False,
        "note": note,
    }


def record_baseline(polarity, applied_log):
    """Append the hand-chosen palette as this polarity's first application, with the theta
    nearest to it and how near that is. Refuses to append twice."""
    from .inverse import nearest_theta

    existing = applied_log.read()
    if any(row.get("kind") == "baseline" and row["polarity"] == polarity for row in existing):
        print(f"the {polarity} baseline is already recorded")
        return
    theta, distance = nearest_theta(HAND_PALETTES[polarity], polarity)
    applied_log.append(
        {
            "ts": HAND_LAYER_APPLIED,
            "recorded": datetime.now(UTC).isoformat(timespec="seconds"),
            "polarity": polarity,
            "kind": "baseline",
            "palette": {**HAND_PALETTES[polarity], "theta": theta, "theta_fit_rms_de": round(distance, 2)},
        }
    )
    print(f"recorded the {polarity} hand layer as the baseline; nearest theta {theta} at RMS {distance:.2f} dE")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "preferred",
        choices=("current", "previous", "baseline"),
        help="which of the two you would rather live in; or `baseline` to record the hand layer",
    )
    parser.add_argument("--polarity", choices=("day", "night"), default="day")
    parser.add_argument("--note", help="anything worth remembering about the comparison")
    args = parser.parse_args(argv)

    if args.preferred == "baseline":
        record_baseline(args.polarity, ResponseLog(paths.APPLIED_LOG))
        return 0
    applied = ResponseLog(paths.APPLIED_LOG).read()
    recent = latest_applications(applied, args.polarity)
    if len(recent) < 2:
        print(
            f"only {len(recent)} distinct {args.polarity} palette(s) have been applied, so there is nothing to "
            f"compare yet. Apply a second one (apply-measured-theme --apply after a new publish) and live in it first."
        )
        return 1
    current, previous = recent
    row = lived_duel(current, previous, args.preferred, args.note)
    ResponseLog(paths.LIVED_LOG).append(row)
    winner = row["theme_a"] if row["choice"] == 0 else row["theme_b"]
    print(
        f"recorded: the {args.polarity} theme applied {current['ts']} (ground {current['palette']['ground']}) "
        f"against the one applied {previous['ts']} (ground {previous['palette']['ground']}); "
        f"you prefer ground {winner['ground']}."
    )
    print(json.dumps({k: row[k] for k in ("polarity", "choice", "applied_a", "applied_b", "note")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
