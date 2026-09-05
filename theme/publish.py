"""Read the log and publish the champion, without a browser.

    pixi run verdict     # print what the model believes, both polarities
    pixi run publish     # the same, and write data/measured-theme.json for the applier

The notebook is the reading surface; this is the same computation as a command, for the
cases a notebook is wrong for -- a script that applies the theme, a session that wants
the numbers without a kernel, a test that checks them. Both go through `theme.verdict`, so
there is exactly one definition of what the log says.
"""

import argparse

from . import paths
from .preference import rt_exponent
from .responses import ResponseLog
from .space import AXES, DE_MIN, VISION_N
from .thresholds import separation_floor
from .verdict import publish, verdict_for


def all_responses():
    """The instrument's own log plus the lived duels, which the fit reads as one log."""
    return ResponseLog(paths.RESPONSE_LOG).read() + ResponseLog(paths.LIVED_LOG).read()


def describe(verdict):
    """One polarity's verdict as lines of text."""
    lines = [
        f"{verdict.polarity}: {verdict.n_duels} duels, verdict {verdict.verdict}, leader holds "
        f"{100 * verdict.lead:.0f}% of the probability of being best, credible set of {len(verdict.credible)}",
    ]
    if verdict.legibility:
        note = verdict.legibility
        low, high = note.gap_interval
        lines.append(
            f"  legibility: {note.n_timed} timed trials dropped {note.n_excluded} of {note.n_candidates} "
            f"candidates; champion reads in {note.champion_seconds / 1000:.1f} s, "
            f"{note.gap_log_time:+.2f} [{low:+.2f}, {high:+.2f}] log-time against the fastest"
            + (" -- CREDIBLY SLOWER" if note.champion_credibly_slower else "")
        )
    if verdict.progress:
        progress = verdict.progress
        lines.append(
            f"  progress over the last {progress['back']} duels: leader {100 * progress['lead_then']:.0f}% -> "
            f"{100 * progress['lead_now']:.0f}%, credible set {progress['set_then']} -> {progress['set_now']}"
        )
    for key, (n, gain, p_value, wording) in verdict.factors.items():
        lines.append(f"  factor {key}: n={n}, gain {gain:+.4f} nats/duel, p={p_value:.2f}: {wording}")
    settled = sorted((c for c in verdict.consensus if c[1] < 0.55), key=lambda c: c[1])
    open_axes = sorted((c for c in verdict.consensus if c[1] > 0.85), key=lambda c: -c[1])
    lines.append(
        "  settled: "
        + (", ".join(f"{AXES[a]} at {m:.2f}" for a, _r, m in settled) or "nothing")
        + " | open: "
        + (", ".join(AXES[a] for a, _r, _m in open_axes) or "nothing")
    )
    theme = verdict.champion_theme
    lines.append(
        "  champion: ground {ground} keyword {keyword} function {function} string {string} ink {ink} "
        "comment {comment} punct {punct} find {find_fill}".format(**theme)
    )
    for i in verdict.shown[1:]:
        member = verdict.themes[i]
        lines.append(
            f"  shelf {100 * verdict.shown_probability[i]:.0f}%: ground {member['ground']} keyword "
            f"{member['keyword']} function {member['function']} string {member['string']}"
        )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"write {paths.CHAMPION.name} for the applier")
    args = parser.parse_args(argv)

    responses = all_responses()
    exponent, _scores = rt_exponent(responses)
    print(
        f"{len(responses)} responses; vision n={VISION_N}, dE floors day {DE_MIN['day']:.2f} night "
        f"{DE_MIN['night']:.2f}; separation floor day {separation_floor('day')[0]:.2f} "
        f"night {separation_floor('night')[0]:.2f}; reaction-time exponent {exponent}"
    )
    verdicts = {}
    for polarity in ("day", "night"):
        verdict = verdict_for(responses, polarity)
        if verdict is None:
            print(f"{polarity}: too few duels to read")
            continue
        verdicts[polarity] = verdict
        print("\n".join(describe(verdict)))
    if args.write:
        published = publish(responses)
        print(f"wrote {paths.CHAMPION} ({', '.join(published)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
