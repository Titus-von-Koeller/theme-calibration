"""The two surfaces an applied theme derives from its ground: the page and the border.

The editor's elevation system (Titus's correction of the first attempt, which sank code
into wells) puts every piece of code on ONE paper -- the plain editor, the notebook cell,
the terminal, the chat panel's code -- and drops the notebook page a small step below it,
so content cells read as raised cards behind a quiet border. When a measured palette
replaces the paper, the page and the border have to move with it or the card metaphor
breaks: a notebook cell would then sit on a different ground from a plain editor, which is
the one thing the system exists to prevent.

The steps are the ones Titus chose by hand for Horizon and liked, read back off those
values in CAM16-UCS rather than guessed: the page sits 2.5 J' below the day paper and 3.4
below the night one, and the border sits 11 to 13 J' off the page, away from the ground's
side of it -- darker by day, lighter by night -- at about one and a half times the
ground's chroma. One constant per step, both polarities, because the differences between
the hand values are within what one hex step moves.

Kept out of `space` because nothing in the search reads these: they belong to the applied
theme, not to the stimulus.
"""

from .color import hex_to_rgb, rgb_to_hex, rgb_to_ucs, ucs_to_rgb

#: How far below the code paper the notebook page sits, in J'.
PAGE_STEP = 3.0

#: How far the card border stands off the page, in J', on the side away from the paper.
BORDER_STEP = 12.0

#: The border carries a little more of the ground's hue than the ground does, so the frame
#: reads as the same material rather than as a grey line.
BORDER_CHROMA_GAIN = 1.5


def derived_surfaces(ground_hex: str, polarity: str) -> dict:
    """{"page": hex, "border": hex} for a code paper at `ground_hex`."""
    lightness, a, b = rgb_to_ucs(hex_to_rgb(ground_hex))[0]
    # Raised means closer to the reader's light: the page drops below the paper on both
    # polarities, and the border then steps further in the same direction by day (darker)
    # and the other way by night (lighter), matching the hand-chosen Horizon values.
    page = [lightness - PAGE_STEP, a, b]
    border_direction = -1.0 if polarity == "day" else 1.0
    border = [page[0] + border_direction * BORDER_STEP, a * BORDER_CHROMA_GAIN, b * BORDER_CHROMA_GAIN]
    page_hex, border_hex = rgb_to_hex(ucs_to_rgb([page, border]))
    return {"page": page_hex, "border": border_hex}
