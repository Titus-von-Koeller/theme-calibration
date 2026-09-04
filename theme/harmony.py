"""Ou & Luo (2006) two-colour harmony, transcribed from the published model.

Kept in its own module because it changes for exactly one reason: a correction against
the paper. Nothing here knows about the nine axes, the anchors or the pool, and nothing
here should learn about them — `space.raw_prior` is where this model gets combined with
Berlyne complexity and a warmth term into the prior the search actually uses. That
combination changes often; this transcription should not change at all.

Evaluated in CIELAB because that is where the model is published. The rest of the
instrument works in CAM16-UCS, and converting to the model's own space rather than
re-fitting its ~20 constants into another one is the cheaper kind of honesty.

Prior-mean duty only: it tilts where the search starts, the responses decide where it
ends. `space` re-exports both names, since that was their import path before the split.
"""

import math

import colour
import numpy as np

from .color import hex_to_rgb


def lab(hexes):
    """CIELAB under D65 for one or more hexes."""
    rgb = np.atleast_2d(hex_to_rgb(hexes))
    return colour.XYZ_to_Lab(
        colour.sRGB_to_XYZ(rgb),
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
    )


def ou_luo_pair(lab1, lab2):
    """Two-colour harmony CH = HC + HL + HH for a pair of CIELAB colours.

    The names spell out the paper's quantities: `chromatic_effect` is its H_C,
    `lightness_effect` its H_L (the sum of its L-sum and delta-L terms), and
    `hue_effect` its H_SY, evaluated once per colour and summed.
    """
    lightness_1, a_1, b_1 = lab1
    lightness_2, a_2, b_2 = lab2
    chroma_1, chroma_2 = math.hypot(a_1, b_1), math.hypot(a_2, b_2)
    hue_1 = math.degrees(math.atan2(b_1, a_1)) % 360
    hue_2 = math.degrees(math.atan2(b_2, a_2)) % 360
    hue_difference = math.radians((hue_1 - hue_2 + 180) % 360 - 180)
    hue_distance = 2 * math.sqrt(max(chroma_1 * chroma_2, 0.0)) * abs(math.sin(hue_difference / 2))
    colour_distance = math.hypot(hue_distance, (chroma_1 - chroma_2) / 1.46)
    chromatic_effect = 0.04 + 0.53 * math.tanh(0.8 - 0.045 * colour_distance)
    lightness_effect = (0.28 + 0.54 * math.tanh(-3.88 + 0.029 * (lightness_1 + lightness_2))) + (
        0.14 + 0.15 * math.tanh(-2 + 0.2 * abs(lightness_1 - lightness_2))
    )

    def hue_effect(lightness, chroma, hue):
        chroma_weight = 0.5 + 0.5 * math.tanh(-2 + 0.5 * chroma)
        hue_preference = -0.08 - 0.14 * math.sin(math.radians(hue + 50)) - 0.07 * math.sin(math.radians(2 * hue + 90))
        yellowness = (90 - hue) / 10
        # y - exp(y) peaks at -1, so this exponent cannot overflow for any hue in
        # [0, 360) and the cap never binds. It stays as a guard against a hue arriving
        # unwrapped, which is a caller bug that would otherwise surface as an OverflowError
        # from inside a prior term.
        yellow_effect = ((0.22 * lightness - 12.8) / 10) * math.exp(min(yellowness - math.exp(yellowness), 50))
        return chroma_weight * (hue_preference + yellow_effect)

    return (
        chromatic_effect
        + lightness_effect
        + hue_effect(lightness_1, chroma_1, hue_1)
        + hue_effect(lightness_2, chroma_2, hue_2)
    )
