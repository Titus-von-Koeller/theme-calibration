import math

import numpy as np

from . import paths
from .color import apca_lc, composite, hex_to_rgb, rgb_to_hex, rgb_to_ucs, solve_j, ucs_to_rgb, wcag
from .observer import fit as observer_fit

VISION_LOG = paths.VISION_LOG

# The observer is fit in ONE place — _observer.py, the home of the measurement<->
# preference interlock. It refits lazily from the shared vision log (cached beside it),
# so every new vision trial sharpens these constraints automatically and no instrument
# carries its own copy of the model. v2 fits the psychometric slope, the lapse, a
# chromatic confusion-axis rotation, and threshold as a smooth function of ground
# lightness — all in CAM16-UCS, the same geometry this notebook searches.
if VISION_LOG.exists():
    VISION_FIT = observer_fit(VISION_LOG)
    DE_MIN = {"day": VISION_FIT.de_min_day, "night": VISION_FIT.de_min_night}
    THRESH_DETAIL = {"day": VISION_FIT.de_dir_day, "night": VISION_FIT.de_dir_night}
    VISION_N = VISION_FIT.n
else:
    # No vision data on this machine: the v2.0 fit at 748 trials (2026-09-03),
    # flagged in the analysis so the substitution is never silent.
    DE_MIN = {"day": 3.2, "night": 2.5}
    THRESH_DETAIL = {"day": {}, "night": {}}
    VISION_N = 0


# Theme space and its realization. Nine axes, each in [0, 1]; polarity (light page /
# dark page) is a block factor, not an axis — trials alternate in blocks and the model
# carries it as a tenth, binary coordinate so learning transfers between the two
# without conflating them.
AXES = [
    "ground lightness",
    "ground warmth",
    "accent hue rotation",
    "accent chroma",
    "body contrast",
    "hue spread",
    "comment recede",
    "find hue",
    "find salience",
]

# Horizon's own token colors anchor the accent hues — evolve, don't repaint. Night
# anchors carry alpha in the theme file and are composited onto their page before any
# appearance math (the rule that caught the 30%-alpha comments). Literals are ONE
# family on purpose: Horizon's day string #F6661E and number #F77D26 sit ~3 dE apart
# in CAM16-UCS — inside 2x your measured day threshold — so a string/number split
# would search a distinction your eyes cannot cash.
ANCHORS = {
    "day": {
        "keyword": "#8A31B9",
        "function": "#1D8991",
        "string": "#F6661E",
        "ground": "#FDF0ED",
    },
    "night": {
        "keyword": composite("#B877DB", 0.902, "#1C1E26"),
        "function": composite("#25B0BC", 0.902, "#1C1E26"),
        "string": composite("#FAB795", 0.902, "#1C1E26"),
        "ground": "#1C1E26",
    },
}
ROLE_ORDER = ("keyword", "function", "string")


def anchor_polar(polarity):
    _ucs = rgb_to_ucs(hex_to_rgb([ANCHORS[polarity][r] for r in ROLE_ORDER]))
    _m = np.linalg.norm(_ucs[:, 1:], axis=1)
    _h = np.degrees(np.arctan2(_ucs[:, 2], _ucs[:, 1])) % 360
    return _h, _m


ANCHOR_HM = {p: anchor_polar(p) for p in ("day", "night")}

# Realization and prior are pure functions of (theta, polarity); the caches make the
# per-trial local-refinement candidates (and every posterior call over the pool) pay
# for their appearance math exactly once per kernel.
REALIZE_CACHE = {}
PRIOR_CACHE = {}


def theta_key(theta, polarity):
    return (tuple(round(float(_v), 6) for _v in theta), polarity)


# NOTE (marimo name mangling, measured 2026-09-03): a cell-local (underscore) name
# referenced from inside an exported function resolves only if it is defined ABOVE
# that function in the cell — a later definition stays unmangled in the function body
# and NameErrors at call time under `marimo run`, invisibly to script execution.
# Helpers therefore precede their exported callers.
def realize_uncached(theta, polarity):
    _t = np.asarray(theta, dtype=float)
    _night = polarity == "night"
    # Ground: lightness within the polarity's family, warmth as a signed warm/cool axis.
    _gj = 8.0 + 14.0 * _t[0] if _night else 86.0 + 9.0 * _t[0]
    _w = 2.0 * _t[1] - 1.0
    _gh = math.radians(74.0 if _w >= 0 else 256.0)
    _gm = abs(_w) * 6.0
    _g_ucs = np.array([_gj, _gm * math.cos(_gh), _gm * math.sin(_gh)])
    _g_rgb = ucs_to_rgb(_g_ucs[None])[0]
    _ground = rgb_to_hex(_g_rgb)[0]

    # Accents: rotate and spread Horizon's hues, scale their chroma, then walk each
    # color's lightness to the body-contrast bar (capped: 12:1 was chosen over the
    # theme's native 18:1 because near-maximum contrast vibrates).
    _h0, _m0 = ANCHOR_HM[polarity]
    _mu = math.degrees(math.atan2(np.sin(np.radians(_h0)).mean(), np.cos(np.radians(_h0)).mean())) % 360
    _rot = (_t[2] - 0.5) * 120.0
    _spread = 0.4 + 1.2 * _t[5]
    _dh = (_h0 - _mu + 180.0) % 360.0 - 180.0
    _hues = (_mu + _spread * _dh + _rot) % 360.0
    _chroma = (0.6 + 0.8 * _t[3]) * _m0
    _r_body = 4.5 + 4.5 * _t[4]
    _ab = np.column_stack([_chroma * np.cos(np.radians(_hues)), _chroma * np.sin(np.radians(_hues))])
    # Neutral family (ink, comment, punctuation): the ground's own hue at a whisper of
    # chroma, so page and text agree in temperature.
    _nd = np.array([math.cos(_gh), math.sin(_gh)])
    _r_comment = max(4.5, _r_body * (0.55 + 0.35 * _t[6]))
    _r_punct = max(4.5, _r_body * 0.75)
    _neut_ab = np.array([_nd * 1.5, _nd * 2.0, _nd * 1.5])  # ink, comment, punct
    _all_ab = np.vstack([_ab, _neut_ab])
    _ratios = np.minimum([_r_body] * 3 + [max(_r_body, 5.5), _r_comment, _r_punct], 12.0)
    # WCAG sets the first target; APCA is the stricter master on dark grounds (4.5:1
    # there is only Lc ~54), so rows short of their Lc floor walk further from the page
    # until both bars hold. The 1.03 margin keeps bisection from converging a hair under.
    _target = _ratios * 1.03
    _lc_floor = np.array([60.0, 60.0, 60.0, 60.0, 45.0, 45.0])
    _g6 = np.repeat(_g_rgb[None], 6, 0)
    for _ in range(4):
        _js, _rgbs = solve_j(_all_ab, _g_rgb, _target, lighter=_night)
        _lc = apca_lc(_rgbs, _g6)
        _short = np.abs(_lc) < _lc_floor
        if not _short.any():
            break
        _target = np.where(_short, np.minimum(_target * 1.18, 14.0), _target)
    _hexes = rgb_to_hex(_rgbs)
    _roles = dict(zip(["keyword", "function", "string", "ink", "comment", "punct"], _hexes, strict=True))

    # Find highlight: a fill near the page's lightness whose loudness is the salience
    # axis. Emitted with alpha (how VSCode layers it); every constraint and every
    # rendered pixel uses the composited result.
    _s = _t[8]
    _fh = math.radians(360.0 * _t[7])
    _fm = 8.0 + 26.0 * _s
    _fj = _gj + (4.0 + 14.0 * _s) * (1 if _night else -1)
    _fill = rgb_to_hex(ucs_to_rgb(np.array([[_fj, _fm * math.cos(_fh), _fm * math.sin(_fh)]])))[0]
    _cur = composite(_fill, 0.85, _ground)
    _oth = composite(_fill, 0.45, _ground)

    # Hard floors, checked on what will actually render. One CAM16 conversion for all
    # eight colors, then plain numpy distances: colour-science's cost is per call, not
    # per color, and this block once spent 14 calls per theme (measured: 1.0 s of a
    # 1.7 s duel generation).
    _de = DE_MIN[polarity]
    _lc = apca_lc(_rgbs, _g6)
    _rr = wcag(_rgbs, _g6)
    if (_rr < 4.5 - 1e-6).any() or (np.abs(_lc[:4]) < 60).any() or (np.abs(_lc[4:]) < 45).any():
        return None
    _names = ["keyword", "function", "string", "ink", "comment"]
    _u = rgb_to_ucs(hex_to_rgb([_roles[r] for r in _names] + [_ground, _cur, _oth]))
    _K, _F, _S, _I, _C, _G, _CUR, _OTH = range(8)

    def _d(a, b):
        return float(np.linalg.norm(_u[a] - _u[b]))

    for _i in (_K, _F, _S):
        if _d(_i, _I) < 2 * _de:
            return None
        for _j2 in (_K, _F, _S):
            if _j2 > _i and _d(_i, _j2) < 2 * _de:
                return None
    if _d(_C, _I) < _de:
        return None
    if _d(_CUR, _G) < 1.5 * _de or _d(_CUR, _OTH) < _de:
        return None
    # Text must survive sitting on either fill.
    _fills = hex_to_rgb([_cur, _oth])
    if (wcag(np.repeat(_rgbs[3:4], 2, 0), _fills) < 4.0).any() or (
        wcag(np.repeat(_rgbs[2:3], 2, 0), _fills) < 3.5
    ).any():
        return None
    _sal = min(_d(_CUR, _i) for _i in (_G, _K, _F, _S, _I))
    return {
        "ground": _ground,
        **_roles,
        "number": _roles["string"],
        "variable": _roles["ink"],
        "find_fill": _fill,
        "find_current": _cur,
        "find_other": _oth,
        "salience": round(_sal, 2),
        "body_ratio": round(float(_rr[:4].min()), 2),
    }


def realize(theta, polarity):
    """theta in [0,1]^9 -> a full, floor-satisfying theme (hexes + meta), or None when
    the hard constraints cannot be met. Floors are constraints, never objectives: WCAG
    4.5:1 and APCA |Lc| >= 60 for body tokens (comments >= 4.5:1, |Lc| >= 45), and
    pairwise CAM16-UCS separation >= 2x your measured 104-px threshold between any two
    colored roles and ink — doubled because discrimination collapses toward glyph
    scale; the comprehension probes measure the truth of that margin directly."""
    _key = theta_key(theta, polarity)
    if _key in REALIZE_CACHE:
        return REALIZE_CACHE[_key]
    _theme = realize_uncached(theta, polarity)
    REALIZE_CACHE[_key] = _theme
    return _theme


# ------------------------------------------------------------------ the prior mean
def lab(hexes):
    _xyz = np.atleast_2d(hex_to_rgb(hexes))
    import colour as _colour

    return _colour.XYZ_to_Lab(
        _colour.sRGB_to_XYZ(_xyz),
        _colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
    )


def ou_luo_pair(lab1, lab2):
    """Two-colour harmony CH = HC + HL + HH, Ou & Luo (2006), transcribed from the
    published model. Prior-mean duty only: it tilts where the search starts, your
    clicks decide where it ends."""
    _L1, _a1, _b1 = lab1
    _L2, _a2, _b2 = lab2
    _C1, _C2 = math.hypot(_a1, _b1), math.hypot(_a2, _b2)
    _h1, _h2 = math.degrees(math.atan2(_b1, _a1)) % 360, math.degrees(math.atan2(_b2, _a2)) % 360
    _dhab = math.radians((_h1 - _h2 + 180) % 360 - 180)
    _dH = 2 * math.sqrt(max(_C1 * _C2, 0.0)) * abs(math.sin(_dhab / 2))
    _dC = math.hypot(_dH, (_C1 - _C2) / 1.46)
    _hc = 0.04 + 0.53 * math.tanh(0.8 - 0.045 * _dC)
    _hl = (0.28 + 0.54 * math.tanh(-3.88 + 0.029 * (_L1 + _L2))) + (0.14 + 0.15 * math.tanh(-2 + 0.2 * abs(_L1 - _L2)))

    def _hsy(_L, _C, _h):
        _ec = 0.5 + 0.5 * math.tanh(-2 + 0.5 * _C)
        _hs = -0.08 - 0.14 * math.sin(math.radians(_h + 50)) - 0.07 * math.sin(math.radians(2 * _h + 90))
        _y = (90 - _h) / 10
        _ey = ((0.22 * _L - 12.8) / 10) * math.exp(min(_y - math.exp(_y), 50))
        return _ec * (_hs + _ey)

    return _hc + _hl + _hsy(_L1, _C1, _h1) + _hsy(_L2, _C2, _h2)


def raw_prior(theta, polarity, theme):
    _t = np.asarray(theta, dtype=float)
    _hx = [theme[r] for r in ROLE_ORDER] + [theme["ground"]]
    _labs = lab(_hx)
    _pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    _harm = float(np.mean([ou_luo_pair(_labs[a], _labs[b]) for a, b in _pairs]))
    # Berlyne: pleasure peaks at intermediate complexity — interior optima on the
    # complexity axes, never a monotone pull to either wall.
    _berlyne = -1.2 * sum((float(_t[i]) - 0.55) ** 2 for i in (3, 4, 5))
    # Ecological-valence stand-in until Titus names his loved colors: his stated warm
    # preference, gently.
    _warm = 0.5 * (float(_t[1]) - 0.5)
    return _harm + _berlyne + _warm


# A fixed, deterministic candidate pool per polarity: the acquisition shops here (plus
# per-trial local refinements around the champion), the prior is standardized here, and
# infeasible corners are carved away by the floors rather than penalized.
pool_rng = np.random.default_rng(0xA55)
POOL_THETA = pool_rng.random((512, 9))
POOL = {}
PRIOR_STATS = {}
for pool_polarity in ("day", "night"):
    pool_items = []
    for pool_idx in range(len(POOL_THETA)):
        pool_th = POOL_THETA[pool_idx]
        pool_realized = realize(pool_th, pool_polarity)
        if pool_realized is not None:
            pool_items.append((pool_th, pool_realized, raw_prior(pool_th, pool_polarity, pool_realized)))
    pool_priors = np.array([it[2] for it in pool_items])
    PRIOR_STATS[pool_polarity] = (float(pool_priors.mean()), float(pool_priors.std() + 1e-9))
    POOL[pool_polarity] = [(it[0], it[1]) for it in pool_items]


def prior_mean(theta, polarity, theme=None):
    """Standardized prior utility (mean 0, sd 0.8 over the feasible pool) so the GP's
    signal variance, not the prior's arbitrary units, sets the scale."""
    _key = theta_key(theta, polarity)
    if _key in PRIOR_CACHE:
        return PRIOR_CACHE[_key]
    theme = theme or realize(theta, polarity)
    if theme is None:
        _val = 0.0
    else:
        _m, _s = PRIOR_STATS[polarity]
        _val = 0.8 * (raw_prior(theta, polarity, theme) - _m) / _s
    PRIOR_CACHE[_key] = _val
    return _val
