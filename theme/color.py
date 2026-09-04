import colour
import numpy as np

# The color engine. All appearance math runs in CAM16-UCS (Li et al. 2017 via
# colour-science) under fixed, documented viewing conditions: D65 white, average
# surround, L_A 40, Y_b 20 — a desktop monitor in a lit room. The screen itself is
# uncalibrated (parked in the queue), which limits absolute claims, not the relative
# structure the instrument learns.
VC = colour.VIEWING_CONDITIONS_CAM16["Average"]
WHITE_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
XYZ_W = colour.xy_to_XYZ(WHITE_XY) * 100.0
LA, YB = 40.0, 20.0

#: sRGB -> relative luminance, WCAG 2.x, applied to LINEARIZED channels.
WCAG_LUMINANCE_COEFF = np.array([0.2126, 0.7152, 0.0722])

#: sRGB -> screen luminance, APCA-W3 0.1.9, applied to ENCODED channels under a simple
#: 2.4 power. Deliberately not the WCAG coefficients and deliberately not the piecewise
#: transfer function: APCA specifies both differently, and mixing the two reports a
#: contrast neither model would.
APCA_LUMINANCE_COEFF = np.array([0.2126729, 0.7151522, 0.0721750])

#: The APCA-W3 0.1.9 (4g) constants, named as the specification names them.
APCA_BLACK_THRESHOLD, APCA_BLACK_CLAMP = 0.022, 1.414
APCA_NORM_BG, APCA_NORM_TEXT = 0.56, 0.57
APCA_REV_BG, APCA_REV_TEXT = 0.65, 0.62
APCA_SCALE, APCA_OFFSET, APCA_LOW_CLIP = 1.14, 0.027, 0.1


#: Byte -> two lowercase hex digits, so `rgb_to_hex` formats by table lookup instead of
#: building a format string per channel.
_HEX_PAIRS = [f"{byte:02x}" for byte in range(256)]


def hex_to_rgb(hexes):
    strings = [hexes] if isinstance(hexes, str) else list(hexes)
    return np.array([[int(s.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for s in strings])


def rgb_to_hex(rgb):
    """An (n, 3) array of channels in [0, 1] -> a list of `#rrggbb` strings.

    This is the quantization the contrast floors are ultimately promised on, so the
    rounding lives in exactly one place. np.rint rounds half to even, which is what
    Python's round() did here before it was vectorized.
    """
    rgb = np.clip(np.atleast_2d(rgb), 0, 1)
    codes = np.rint(255 * rgb).astype(np.uint8)
    return ["#" + _HEX_PAIRS[row[0]] + _HEX_PAIRS[row[1]] + _HEX_PAIRS[row[2]] for row in codes]


def rgb_to_ucs(rgb):
    xyz = colour.sRGB_to_XYZ(np.atleast_2d(rgb)) * 100.0
    appearance = colour.XYZ_to_CAM16(xyz, XYZ_W, L_A=LA, Y_b=YB, surround=VC)
    return colour.JMh_CAM16_to_CAM16UCS(np.stack([appearance.J, appearance.M, appearance.h], axis=-1))


def ucs_to_rgb(ucs):
    """CAM16-UCS J'a'b' -> sRGB, CLIPPED into gamut.

    The clip matters to every caller: a requested J'a'b' outside sRGB comes back as a
    different colour than was asked for, silently. `solve_j` therefore cannot assume the
    colour it gets back has the lightness it bisected to, which is why the floors are
    re-measured on the returned colour rather than trusted from the request.
    """
    jmh = colour.CAM16UCS_to_JMh_CAM16(np.atleast_2d(ucs))
    appearance = colour.CAM_Specification_CAM16(J=jmh[..., 0], M=jmh[..., 1], h=jmh[..., 2])
    xyz = colour.CAM16_to_XYZ(appearance, XYZ_W, L_A=LA, Y_b=YB, surround=VC)
    return np.clip(colour.XYZ_to_sRGB(xyz / 100.0), 0.0, 1.0)


def ucs_dist(hex_a, hex_b):
    return np.linalg.norm(rgb_to_ucs(hex_to_rgb(hex_a)) - rgb_to_ucs(hex_to_rgb(hex_b)), axis=-1)


def rel_lum(rgb):
    rgb = np.atleast_2d(rgb)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ WCAG_LUMINANCE_COEFF


def wcag(rgb_a, rgb_b):
    lum_a, lum_b = rel_lum(rgb_a), rel_lum(rgb_b)
    brighter, darker = np.maximum(lum_a, lum_b), np.minimum(lum_a, lum_b)
    return (brighter + 0.05) / (darker + 0.05)


def apca_lc(txt_rgb, bg_rgb):
    """APCA-W3 0.1.9 (4g) lightness contrast, signed; |Lc| 60 ~ body-text bar."""

    def screen_luminance(rgb):
        luminance = (np.atleast_2d(rgb) ** 2.4) @ APCA_LUMINANCE_COEFF
        # The black soft-clamp, which stops near-black pairs from reporting contrast the
        # eye does not get. Raised under the mask so the fractional power never sees a
        # negative base.
        near_black = luminance < APCA_BLACK_THRESHOLD
        lift = np.where(near_black, APCA_BLACK_THRESHOLD - luminance, 0.0) ** APCA_BLACK_CLAMP
        return luminance + np.where(near_black, lift, 0.0)

    text_lum, ground_lum = screen_luminance(txt_rgb), screen_luminance(bg_rgb)
    sapc = np.where(
        ground_lum > text_lum,
        (ground_lum**APCA_NORM_BG - text_lum**APCA_NORM_TEXT) * APCA_SCALE,
        (ground_lum**APCA_REV_BG - text_lum**APCA_REV_TEXT) * APCA_SCALE,
    )
    return np.where(
        np.abs(sapc) < APCA_LOW_CLIP,
        0.0,
        np.where(sapc > 0, (sapc - APCA_OFFSET) * 100, (sapc + APCA_OFFSET) * 100),
    )


def composite(fg_hex, alpha, bg_hex):
    """fg at `alpha` over bg, as hex — contrast is only ever stated on composited color."""
    return composite_many([fg_hex], alpha, [bg_hex])[0]


def composite_many(fg_hexes, alpha, bg_hexes):
    """`composite` over equal-length sequences of hexes, in one array round trip.

    `alpha` is a scalar or one value per pair. A scalar is used as-is rather than
    broadcast through an array, so the single-pair case pays nothing for the batching.
    """
    foreground, background = hex_to_rgb(fg_hexes), hex_to_rgb(bg_hexes)
    weight = alpha if np.isscalar(alpha) else np.asarray(alpha, dtype=float).reshape(-1, 1)
    return rgb_to_hex(weight * foreground + (1 - weight) * background)


def solve_j(ab, ground_rgb, ratio, lighter, iters=16):
    """Walk lightness to the contrast bar: bisect J' per row of `ab` (CAM16-UCS a', b')
    so each color meets its `ratio` (scalar or per-row) WCAG contrast against the
    ground — the theme-design rule "keep hue and saturation, walk lightness" made
    executable. Vectorized: one inverse-CAM16 call per bisection step for all rows.

    Two things the bracket relies on, neither of them free:

    Contrast against a fixed ground is V-shaped in J', not monotone — it falls as the
    color approaches the ground's lightness and rises again past it. Bisection lands on
    the intended branch only because the grounds this instrument builds sit at the ends
    of the range: a night page is dark enough that nothing BELOW it clears 4.5:1, and a
    day page light enough that nothing above it does. On a mid-lightness ground the other
    root would be reachable and this would find it half the time.

    Nothing here reports failure. When the requested `ratio` is unreachable inside
    J' 2..98 — or when the (J', a', b') asked for falls outside sRGB and comes back
    clipped — the bisection converges on a bound and returns a color that simply misses
    its target. Callers must re-measure contrast on what comes back; the hard floors in
    `space._assemble` are what actually stop such a color being shown, and they guard the
    floors rather than the per-row `ratio`, so a saturated `ratio` surfaces as a theme
    whose reported body_ratio is under what its theta asked for, not as a refusal.

    16 halvings of the 96-unit range leave a bracket 0.0015 J' wide, two orders of
    magnitude finer than the 8-bit quantization the result is rounded to anyway.
    """
    ab = np.atleast_2d(ab)
    count = len(ab)
    targets = np.broadcast_to(np.asarray(ratio, dtype=float), (count,))
    lightness_lo, lightness_hi = np.full(count, 2.0), np.full(count, 98.0)
    grounds = np.atleast_2d(ground_rgb)
    grounds = np.repeat(grounds, count, 0) if len(grounds) == 1 else grounds
    for _ in range(iters):
        midpoint = (lightness_lo + lightness_hi) / 2
        below_bar = wcag(ucs_to_rgb(np.column_stack([midpoint, ab])), grounds) < targets
        # Contrast rises with J when the text is lighter than the ground, falls when
        # it is darker; too-low contrast moves the bound that walks away from the page.
        if lighter:
            lightness_lo = np.where(below_bar, midpoint, lightness_lo)
            lightness_hi = np.where(below_bar, lightness_hi, midpoint)
        else:
            lightness_hi = np.where(below_bar, midpoint, lightness_hi)
            lightness_lo = np.where(below_bar, lightness_lo, midpoint)
    lightness = (lightness_lo + lightness_hi) / 2
    return lightness, ucs_to_rgb(np.column_stack([lightness, ab]))
