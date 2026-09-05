"""Record what living in a theme was like, as a duel the model already understands.

    pixi run lived -- current      # today's applied theme beat the one before it
    pixi run lived -- previous     # the one before it was better
    pixi run lived -- --note "..." current

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


def latest_applications(applied_rows, polarity, count=2):
    """The most recent `count` distinct palettes applied for one polarity, newest first."""
    seen, picked = set(), []
    for row in reversed(applied_rows):
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preferred", choices=("current", "previous"), help="which of the two you would rather live in")
    parser.add_argument("--polarity", choices=("day", "night"), default="day")
    parser.add_argument("--note", help="anything worth remembering about the comparison")
    args = parser.parse_args(argv)

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
