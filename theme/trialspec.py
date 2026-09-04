"""Turning a trial into a page: the prompt, the cards, and the run's instruction.

Both halves of the instrument need this and must agree exactly. The server renders it to
show a trial; the recorder replays it to work out which target was asked for. They agree
because they call the same function with the same seed, never because two pieces of code
were written to match.

The seeding is load-bearing, and so is the ORDER the draws are made in. `rng_for(n)` is
deterministic in the trial number alone, so the target the page asked for is recoverable
from the log at any later date without having stored it -- which is what lets a response be
recorded from the LOG rather than from whatever the browser happened to be holding. Each
arm below spends its trial RNG on exactly one draw, and `theme.responses` spends a fresh
RNG for the same trial on the same draw; adding a second draw on either side, or reordering
them, silently decouples the two.
"""

import html
import random

from .stimulus import render_card

# The target token, set as a token rather than as running text: the same typeface it wears
# in the code, on a neutral tint so it reads as a quoted string rather than as part of the
# sentence. Neutral on purpose -- a tinted chip in any theme colour would cue the search,
# and the find-highlight hue is one of the axes under test.
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


def _card(theme: dict, page: dict, code_px: int, **how) -> dict:
    """One card: its HTML, and the ground the stage is painted with behind it."""
    return {"html": render_card(theme, page, code_px, **how), "ground": theme["ground"]}


def stimulus_for(n: int, trial: dict, page: dict) -> tuple[str, list[dict], int | None]:
    """(prompt html, cards, the current-match id) for one trial.

    A duel shows the SAME page twice under two themes, so nothing varies but the theme;
    the sides are swapped by the trial's own `swap` flag and the swap is logged, so the
    fitted side bias can be subtracted from the utility rather than left as noise.

    The two task arms hand back None as the current-match id when the arm has no find
    layer, which is what tells the caller there is no highlight to reveal.
    """
    rng = rng_for(n)
    surface = trial.get("surface", "editor")

    if trial["mode"] == "duel":
        current_match = rng.choice(page["ident_ids"]) if page["ident_ids"] else None
        cards = [
            _card(theme, page, trial["code_px"], find_current=current_match, surface=surface)
            for theme in (trial["theme_a"], trial["theme_b"])
        ]
        if trial["swap"]:
            cards = cards[::-1]
        prompt = 'Which page would you rather read? <span style="opacity:.55">Click it.</span>'
        return prompt, cards, current_match

    if trial["mode"] == "comprehension":
        # `fn_ids` holds exactly one span on a generated page, so this draw has one
        # outcome there; it is still made through the trial RNG because a control page
        # offers several and the recorder has to reach the same one.
        target = rng.choice(page["fn_ids"])
        name = page["spans"][target]["text"]
        cards = [_card(trial["theme_a"], page, trial["code_px"], task=True, prose=False)]
        # Escaped even though a Python identifier cannot carry markup: this string is
        # built from page content and emitted as HTML, and that is the whole rule.
        return f'Click <code style="{MONO}">{html.escape(name)}</code>', cards, None

    # A find hunt. Every occurrence of the page's repeated identifier is highlighted and
    # one of them is the current match; `codegen` refuses any page with no such identifier,
    # so `ident_ids` is non-empty by the time a page reaches here.
    current_match = rng.choice(page["ident_ids"])
    cards = [
        _card(
            trial["theme_a"],
            page,
            trial["code_px"],
            find_current=current_match,
            task=True,
            prose=False,
        )
    ]
    prompt = 'Click the <b>current</b> match <span style="opacity:.55">— the strongest highlight.</span>'
    return prompt, cards, current_match
