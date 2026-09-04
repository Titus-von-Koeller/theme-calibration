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


def hex_to_rgb(hexes):
    _h = [hexes] if isinstance(hexes, str) else list(hexes)
    return np.array([[int(s.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for s in _h])


def rgb_to_hex(rgb):
    rgb = np.clip(np.atleast_2d(rgb), 0, 1)
    return ["#" + "".join(f"{round(255 * float(v)):02x}" for v in row) for row in rgb]


def rgb_to_ucs(rgb):
    _xyz = colour.sRGB_to_XYZ(np.atleast_2d(rgb)) * 100.0
    _spec = colour.XYZ_to_CAM16(_xyz, XYZ_W, L_A=LA, Y_b=YB, surround=VC)
    return colour.JMh_CAM16_to_CAM16UCS(np.stack([_spec.J, _spec.M, _spec.h], axis=-1))


def ucs_to_rgb(ucs):
    _jmh = colour.CAM16UCS_to_JMh_CAM16(np.atleast_2d(ucs))
    _spec = colour.CAM_Specification_CAM16(J=_jmh[..., 0], M=_jmh[..., 1], h=_jmh[..., 2])
    _xyz = colour.CAM16_to_XYZ(_spec, XYZ_W, L_A=LA, Y_b=YB, surround=VC)
    return np.clip(colour.XYZ_to_sRGB(_xyz / 100.0), 0.0, 1.0)


def ucs_dist(hex_a, hex_b):
    _a = rgb_to_ucs(hex_to_rgb(hex_a))
    _b = rgb_to_ucs(hex_to_rgb(hex_b))
    return np.linalg.norm(_a - _b, axis=-1)


def rel_lum(rgb):
    rgb = np.atleast_2d(rgb)
    _lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return _lin @ np.array([0.2126, 0.7152, 0.0722])


def wcag(rgb_a, rgb_b):
    _la, _lb = rel_lum(rgb_a), rel_lum(rgb_b)
    _hi, _lo = np.maximum(_la, _lb), np.minimum(_la, _lb)
    return (_hi + 0.05) / (_lo + 0.05)


def apca_lc(txt_rgb, bg_rgb):
    """APCA-W3 0.1.9 (4g) lightness contrast, signed; |Lc| 60 ~ body-text bar."""

    def _y(rgb):
        _v = (np.atleast_2d(rgb) ** 2.4) @ np.array([0.2126729, 0.7151522, 0.0721750])
        _lift = np.where(_v < 0.022, 0.022 - _v, 0.0) ** 1.414
        return _v + np.where(_v < 0.022, _lift, 0.0)

    _yt, _yb = _y(txt_rgb), _y(bg_rgb)
    _sapc = np.where(_yb > _yt, (_yb**0.56 - _yt**0.57) * 1.14, (_yb**0.65 - _yt**0.62) * 1.14)
    return np.where(np.abs(_sapc) < 0.1, 0.0, np.where(_sapc > 0, (_sapc - 0.027) * 100, (_sapc + 0.027) * 100))


def composite(fg_hex, alpha, bg_hex):
    """fg at `alpha` over bg, as hex — contrast is only ever stated on composited color."""
    _fg, _bg = hex_to_rgb(fg_hex)[0], hex_to_rgb(bg_hex)[0]
    return rgb_to_hex(alpha * _fg + (1 - alpha) * _bg)[0]


def solve_j(ab, ground_rgb, ratio, lighter, iters=16):
    """Walk lightness to the contrast bar: bisect J' per row of `ab` (CAM16-UCS a', b')
    so each color meets its `ratio` (scalar or per-row) WCAG contrast against the
    ground — the theme-design rule "keep hue and saturation, walk lightness" made
    executable. Vectorized: one inverse-CAM16 call per bisection step for all rows."""
    ab = np.atleast_2d(ab)
    _n = len(ab)
    _ratio = np.broadcast_to(np.asarray(ratio, dtype=float), (_n,))
    _lo, _hi = np.full(_n, 2.0), np.full(_n, 98.0)
    _g = np.atleast_2d(ground_rgb)
    _g = np.repeat(_g, _n, 0) if len(_g) == 1 else _g
    for _ in range(iters):
        _mid = (_lo + _hi) / 2
        _rgb = ucs_to_rgb(np.column_stack([_mid, ab]))
        _too_low = wcag(_rgb, _g) < _ratio
        # Contrast rises with J when the text is lighter than the ground, falls when
        # it is darker; too-low contrast moves the bound that walks away from the page.
        if lighter:
            _lo = np.where(_too_low, _mid, _lo)
            _hi = np.where(_too_low, _hi, _mid)
        else:
            _hi = np.where(_too_low, _mid, _hi)
            _lo = np.where(_too_low, _lo, _mid)
    _j = (_lo + _hi) / 2
    return _j, ucs_to_rgb(np.column_stack([_j, ab]))
