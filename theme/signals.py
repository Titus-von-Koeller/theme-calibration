"""Signals: the colours that carry meaning by convention, derived from the palette.

An editor has a third kind of colour besides furniture and data. Error red, warning
orange, success green, the git decorations, the diagnostics squiggles, the sixteen ANSI
colours a terminal program assumes, the badge on the activity bar: each carries a meaning
the whole software world agreed on by hue, so the hue is not ours to choose. What IS ours
is everything else about it -- how light, how saturated, and therefore whether it clears the
same floors the body text clears on the measured paper. Left to the theme (Horizon) or to
VSCode's defaults, these colours were the last places the window still spoke a different
palette: a pink badge, a crimson link and a grey rule on a cream page (census, 2026-09-05).

Each signal is a fixed CAM16-UCS hue at the accents' chroma, walked in lightness to the
body-text floor on the palette's ground -- exactly the walk the roles get in `space`, so a
signal is legible by the same measurement. Walked to the FLOOR rather than deeper, so
signals sit a step lighter than the body accents (which the duels put near 7.7:1 by day)
and stay separable from them by lightness even where a hue coincides with a syntax role.
Bright variants for the terminal keep the floor and add chroma: on a light paper "bright"
cannot mean lighter without losing the text.

The applier maps these onto VSCode's keys; this module never names a key.
"""

import numpy as np

from .color import apca_lc, hex_to_rgb, rgb_to_hex, rgb_to_ucs, solve_j, wcag

#: Conventional hues in CAM16-UCS degrees, measured off the sRGB colours the conventions
#: come from (a saturated red, orange, yellow, green, cyan, blue and magenta).
SIGNAL_HUES = {
    "red": 27.0,
    "orange": 52.0,
    "yellow": 97.0,
    "green": 143.0,
    "cyan": 196.0,
    "blue": 274.0,
    "magenta": 339.0,
}

#: What a signal owes the paper: body-text WCAG, and the APCA bar for the small UI text
#: and icons signals are mostly drawn as.
WCAG_FLOOR = 4.5
LC_FLOOR = 45.0

#: Signals take the accents' chroma with a little more, because a signal is a rare surface
#: and mild exaggeration reads as intended there (peak shift, licensed for links, errors
#: and selection in the method reef).
CHROMA_GAIN = 1.1
BRIGHT_CHROMA_GAIN = 1.35

#: How many times the walk raises its target when quantization leaves a signal under a
#: floor. Same shape as `space._walk_to_both_bars`.
WALK_ATTEMPTS = 6
WALK_STEP = 1.12


def accent_chroma(accent_hexes):
    """The chroma signals inherit: the strongest of the palette's accents."""
    ucs = rgb_to_ucs(hex_to_rgb(list(accent_hexes)))
    return float(np.hypot(ucs[:, 1], ucs[:, 2]).max())


def _walk(hues_deg, chroma, ground_hex, night):
    """Hexes for the given hues at `chroma`, walked to the floors on the ground."""
    angles = np.radians(np.asarray(hues_deg, dtype=float))
    ab = np.column_stack([chroma * np.cos(angles), chroma * np.sin(angles)])
    ground = hex_to_rgb(ground_hex)
    grounds = np.repeat(ground, len(ab), axis=0)
    target = np.full(len(ab), WCAG_FLOOR * 1.03)
    hexes = None
    for _ in range(WALK_ATTEMPTS):
        _lightness, rgb = solve_j(ab, ground, target, lighter=night)
        hexes = rgb_to_hex(rgb)
        rendered = hex_to_rgb(hexes)
        short = (wcag(rendered, grounds) < WCAG_FLOOR) | (np.abs(apca_lc(rendered, grounds)) < LC_FLOOR)
        if not short.any():
            break
        target = np.where(short, target * WALK_STEP, target)
    return hexes


def signals_for(ground_hex, accent_hexes, polarity):
    """{name: hex} for every signal, plus `name_bright` variants, on this ground."""
    night = polarity == "night"
    chroma = accent_chroma(accent_hexes)
    names = list(SIGNAL_HUES)
    hues = [SIGNAL_HUES[name] for name in names]
    normal = _walk(hues, chroma * CHROMA_GAIN, ground_hex, night)
    bright = _walk(hues, chroma * BRIGHT_CHROMA_GAIN, ground_hex, night)
    result = dict(zip(names, normal, strict=True))
    result.update({f"{name}_bright": hex_ for name, hex_ in zip(names, bright, strict=True)})
    return result


def hue_of(hex_):
    """CAM16-UCS hue of a colour in degrees, for checking a signal kept its convention."""
    ucs = rgb_to_ucs(hex_to_rgb(hex_))[0]
    return float(np.degrees(np.arctan2(ucs[2], ucs[1])) % 360.0)
