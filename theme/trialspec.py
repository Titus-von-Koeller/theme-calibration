"""Turning a trial into a page: the prompt, the cards, and the run's instruction.

Both halves of the instrument need this and must agree exactly. The server renders it to
show a trial; the recorder replays it to work out which target was asked for. They agree
because they call the same function with the same seed, never because two pieces of code
were written to match.

The seeding is load-bearing. `rng_for(n)` is deterministic in the trial number alone, so
the target the page asked for is recoverable from the log at any later date without having
stored it -- which is what lets a response be recorded from the LOG rather than from
whatever the browser happened to be holding.
"""

import random

from .stimulus import render_card

# The thing he is hunting for, set as a token rather than as running text: same typeface it
# wears in the code, on a neutral tint so it reads as a quoted string rather than as part
# of the sentence. Neutral on purpose -- a tinted chip in any theme colour would cue the
# search, and the find-highlight hue is one of the axes under test.
MONO = (
    "font-family:'IosevkaLigated Nerd Font Mono',monospace;font-size:20px;"
    "background:color-mix(in srgb, currentColor 9%, transparent);"
    "padding:2px 9px;border-radius:6px;letter-spacing:.01em"
)


def rng_for(n: int) -> random.Random:
    """The per-trial RNG. Deterministic in n, so server and recorder pick the same target."""
    return random.Random(n * 48271 % (2**31))


def gate_text_for(mode: str, polarity: str, run_len: int) -> str:
    """One instruction serves a whole run, so no click is spent re-reading it mid-stride."""
    return {
        "duel": (
            f"A run of {run_len} duels on the {polarity} page: two pages render the same "
            "code — click the one you would rather read. Trust the first pull; a slow "
            "choice reads as a tie."
        ),
        "comprehension": (
            f"A run of {run_len} probes on the {polarity} page: the line above names a "
            "function — click that name in the code as fast as you can find it."
        ),
        "search": (
            f"A run of {run_len} find hunts on the {polarity} page: several matches are "
            "highlighted — click the current one, the strongest highlight, as fast as you "
            "can find it."
        ),
    }[mode]


def stimulus_for(n: int, trial: dict, snip: dict) -> tuple[str, list[dict], int | None]:
    """(prompt html, cards, the current-match id) for one trial.

    A duel shows the SAME page twice under two themes, so nothing varies but the theme;
    the sides are swapped by the trial's own `swap` flag and the swap is logged, so the
    fitted side bias can be subtracted from the utility rather than left as noise.
    """
    rng = rng_for(n)
    surface = trial.get("surface", "editor")
    if trial["mode"] == "duel":
        cur = rng.choice(snip["ident_ids"]) if snip["ident_ids"] else None
        cards = [
            {
                "html": render_card(t, snip, trial["code_px"], find_current=cur, surface=surface),
                "ground": t["ground"],
            }
            for t in (trial["theme_a"], trial["theme_b"])
        ]
        if trial["swap"]:
            cards = cards[::-1]
        prompt = 'Which page would you rather read? <span style="opacity:.55">Click it.</span>'
        return prompt, cards, cur

    if trial["mode"] == "comprehension":
        target = rng.choice(snip["fn_ids"])
        name = snip["spans"][target]["text"]
        cards = [
            {
                "html": render_card(trial["theme_a"], snip, trial["code_px"], task=True, prose=False),
                "ground": trial["theme_a"]["ground"],
            }
        ]
        return f'Click <code style="{MONO}">{name}</code>', cards, None

    cur = rng.choice(snip["ident_ids"])
    cards = [
        {
            "html": render_card(trial["theme_a"], snip, trial["code_px"], find_current=cur, task=True, prose=False),
            "ground": trial["theme_a"]["ground"],
        }
    ]
    prompt = 'Click the <b>current</b> match <span style="opacity:.55">— the strongest highlight.</span>'
    return prompt, cards, cur
