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
# The roles solve_j walks to the contrast bar, in the order their rows appear.
WALKED_ROLES = ("keyword", "function", "string", "ink", "comment", "punct")
#: APCA floors per walked role. Body tokens carry meaning; comments are context.
LC_FLOORS = np.array([60.0, 60.0, 60.0, 60.0, 45.0, 45.0])


def _grounds(thetas, night):
    """The page colour for each theta: lightness within the polarity's family, warmth as a
    signed warm/cool axis."""
    lightness = 8.0 + 14.0 * thetas[:, 0] if night else 86.0 + 9.0 * thetas[:, 0]
    warmth = 2.0 * thetas[:, 1] - 1.0
    hue = np.where(warmth >= 0, math.radians(74.0), math.radians(256.0))
    chroma = np.abs(warmth) * 6.0
    ucs = np.column_stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)])
    return lightness, hue, ucs_to_rgb(ucs)


def _role_chromaticities(thetas, polarity, ground_hue):
    """The a', b' each walked role starts from, and the contrast ratio it must reach.

    Accents rotate and spread Horizon's own hues and scale their chroma; the neutral family
    (ink, comment, punctuation) takes the GROUND's hue at a whisper of chroma, so page and
    text agree in temperature. Lightness is not set here -- that is what solve_j walks.
    """
    anchor_hues, anchor_chroma = ANCHOR_HM[polarity]
    mean_hue = (
        math.degrees(math.atan2(np.sin(np.radians(anchor_hues)).mean(), np.cos(np.radians(anchor_hues)).mean())) % 360
    )
    rotation = (thetas[:, 2] - 0.5) * 120.0
    spread = 0.4 + 1.2 * thetas[:, 5]
    offsets = (anchor_hues - mean_hue + 180.0) % 360.0 - 180.0
    hues = (mean_hue + spread[:, None] * offsets[None, :] + rotation[:, None]) % 360.0
    chroma = (0.6 + 0.8 * thetas[:, 3])[:, None] * anchor_chroma[None, :]
    accent_ab = np.stack([chroma * np.cos(np.radians(hues)), chroma * np.sin(np.radians(hues))], axis=-1)

    neutral_direction = np.stack([np.cos(ground_hue), np.sin(ground_hue)], axis=-1)
    neutral_ab = neutral_direction[:, None, :] * np.array([1.5, 2.0, 1.5])[None, :, None]

    # 12:1 caps the body bar: the theme's native 18:1 is near-maximum contrast, where light
    # bleeds into the glyph edges and fine strokes appear to vibrate.
    body = 4.5 + 4.5 * thetas[:, 4]
    comment = np.maximum(4.5, body * (0.55 + 0.35 * thetas[:, 6]))
    punctuation = np.maximum(4.5, body * 0.75)
    ratios = np.minimum(np.column_stack([body, body, body, np.maximum(body, 5.5), comment, punctuation]), 12.0)
    return np.concatenate([accent_ab, neutral_ab], axis=1), ratios


def _walk_to_both_bars(role_ab, ground_rgb, ratios, night):
    """Walk every role's lightness until WCAG and APCA both hold.

    WCAG sets the first target; APCA is the stricter master on dark grounds (4.5:1 there is
    only Lc ~54), so rows short of their Lc floor walk further from the page until both
    bars hold. The 1.03 margin keeps bisection from converging a hair under.

    Batched across themes, which is the whole reason this is a separate function. The cost
    of colour-science is per CALL, not per colour -- measured, a call converting one colour
    costs 312 us and a call converting sixty-four costs 325 us -- so walking six roles for
    one theme at a time spent 3.8 s of a 4 s trial on Python-level validation inside the
    library. Iterating over all themes at once makes that one call per bisection step.
    """
    rows = role_ab.reshape(-1, 2)
    grounds = np.repeat(ground_rgb, role_ab.shape[1], axis=0)
    target = ratios.reshape(-1) * 1.03
    floors = np.tile(LC_FLOORS, len(role_ab))
    for _ in range(4):
        _lightness, rgb = solve_j(rows, grounds, target, lighter=night)
        short = np.abs(apca_lc(rgb, grounds)) < floors
        if not short.any():
            break
        target = np.where(short, np.minimum(target * 1.18, 14.0), target)
    return rgb.reshape(*role_ab.shape[:2], 3), grounds.reshape(*role_ab.shape[:2], 3)


def _find_fills(thetas, ground_lightness, night):
    """The find highlight: a fill near the page's lightness whose loudness is the salience
    axis. Emitted with alpha, because that is how VSCode layers it; every constraint and
    every rendered pixel uses the composited result."""
    salience = thetas[:, 8]
    hue = 2.0 * math.pi * thetas[:, 7]
    chroma = 8.0 + 26.0 * salience
    lightness = ground_lightness + (4.0 + 14.0 * salience) * (1 if night else -1)
    return ucs_to_rgb(np.column_stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)]))


def _assemble(role_rgb, ground_rgb, fill_rgb, separations, polarity):
    """One theme, or None if it breaks a floor.

    Floors are checked on what will actually RENDER -- composited fills included -- and are
    hard constraints, never objectives: every candidate shown is already legible, and the
    only question ever asked is which is better.
    """
    threshold = DE_MIN[polarity]
    ground_hex = rgb_to_hex(ground_rgb)[0]
    role_hex = rgb_to_hex(role_rgb)
    roles = dict(zip(WALKED_ROLES, role_hex, strict=True))

    # Checked on the QUANTIZED colours -- the 8-bit values the page will actually write --
    # not on the unrounded ones the bisection produced. Rounding to hex moves a colour by
    # up to half a step, which is enough to cross a floor: a property test found a theme
    # whose function and string tokens passed at Lc 60.27 and 60.06 and rendered at 59.89
    # and 59.83. Both were shown. The floors are a promise about pixels, so they have to be
    # measured on pixels.
    rendered = hex_to_rgb(role_hex)
    rendered_ground = np.repeat(hex_to_rgb([ground_hex]), len(role_hex), axis=0)
    lc = apca_lc(rendered, rendered_ground)
    contrast = wcag(rendered, rendered_ground)
    if (contrast < 4.5 - 1e-6).any() or (np.abs(lc[:4]) < 60).any() or (np.abs(lc[4:]) < 45).any():
        return None
    fill_hex = rgb_to_hex(fill_rgb)[0]
    current = composite(fill_hex, 0.85, ground_hex)
    other = composite(fill_hex, 0.45, ground_hex)

    keyword, function, string, ink, comment, ground, cur, oth = range(8)

    def gap(a, b):
        return float(np.linalg.norm(separations[a] - separations[b]))

    for accent in (keyword, function, string):
        if gap(accent, ink) < 2 * threshold:
            return None
        for other_accent in (keyword, function, string):
            if other_accent > accent and gap(accent, other_accent) < 2 * threshold:
                return None
    if gap(comment, ink) < threshold:
        return None
    if gap(cur, ground) < 1.5 * threshold or gap(cur, oth) < threshold:
        return None
    # Text must survive sitting on either fill.
    fills = hex_to_rgb([current, other])
    if (wcag(np.repeat(rendered[3:4], 2, 0), fills) < 4.0).any() or (
        wcag(np.repeat(rendered[2:3], 2, 0), fills) < 3.5
    ).any():
        return None

    return {
        "ground": ground_hex,
        **roles,
        "number": roles["string"],
        "variable": roles["ink"],
        "find_fill": fill_hex,
        "find_current": current,
        "find_other": other,
        "salience": round(min(gap(cur, i) for i in (ground, keyword, function, string, ink)), 2),
        "body_ratio": round(float(contrast[:4].min()), 2),
    }


def _realize_batch(thetas, polarity):
    """The colour work for a batch of thetas: a list in the same order, None where refused.

    Batched because colour-science's cost is per CALL rather than per colour -- measured, a
    call converting one colour costs 312 us and a call converting sixty-four costs 325 us.
    Uncached; go through realize_many.
    """
    table = np.asarray(thetas, dtype=float).reshape(-1, 9)
    night = polarity == "night"
    ground_lightness, ground_hue, ground_rgb = _grounds(table, night)
    role_ab, ratios = _role_chromaticities(table, polarity, ground_hue)
    role_rgb, _walked_grounds = _walk_to_both_bars(role_ab, ground_rgb, ratios, night)
    fill_rgb = _find_fills(table, ground_lightness, night)

    # One conversion for every colour of every theme: the eight whose pairwise CAM16-UCS
    # distances the separation floors are stated in.
    ground_hexes = rgb_to_hex(ground_rgb)
    fill_hexes = rgb_to_hex(fill_rgb)
    separation_hexes = []
    for i, ground_hex in enumerate(ground_hexes):
        role_hex = rgb_to_hex(role_rgb[i])
        separation_hexes.extend(
            [
                *role_hex[:5],
                ground_hex,
                composite(fill_hexes[i], 0.85, ground_hex),
                composite(fill_hexes[i], 0.45, ground_hex),
            ]
        )
    separations = rgb_to_ucs(hex_to_rgb(separation_hexes)).reshape(len(table), 8, 3)

    return [_assemble(role_rgb[i], ground_rgb[i], fill_rgb[i], separations[i], polarity) for i in range(len(table))]


#: How conspicuous a find highlight must be before a TIMED hunt may use it, in multiples
#: of the measured discrimination threshold.
#:
#: These are two different perceptual questions and the instrument was conflating them.
#: DE_MIN is a DISCRIMINATION threshold: can two patches be told apart, side by side, at
#: 104 px. Finding one highlighted token in a page of code is VISUAL SEARCH, which needs
#: conspicuity -- the target has to win against every distractor at a glance -- and that
#: takes many multiples of a discrimination step. realize() only ever required the current
#: highlight to sit 1.5x the threshold from the ground, so themes at ~2x came through, and
#: the active sampler sought exactly those out because an unexplored corner is where a
#: GP's variance is highest.
#:
#: 4x, from his own hunts. Over 33 usable trials, salience correlates with log find time at
#: -0.43 (day) and -0.37 (night) -- more conspicuous really is faster -- and splitting at
#: the median gives 3489 ms against 2066 ms by day, 2897 against 2225 by night. A faint
#: highlight costs well over a second, which is not a measurement of the theme but of
#: patience. 4x excludes roughly the slowest quarter of what has been shown.
#:
#: Applied per ARM, deliberately, and not inside realize(). A quiet highlight is a
#: legitimate thing to PREFER, and theta 8 exists to find that trade-off, so duels must
#: keep exploring it. What a faint highlight destroys is the timed hunt, so that is where
#: the floor belongs.
CONSPICUITY_FLOOR = 4.0


def conspicuous_enough(theme, polarity):
    """Is this theme's find highlight loud enough for a timed hunt to mean anything?

    `salience` is the minimum CAM16-UCS distance from the current highlight to the ground
    and to every coloured role -- distance from everything it has to win against, which is
    the right measure for search rather than for discrimination.
    """
    return theme is not None and theme["salience"] >= CONSPICUITY_FLOOR * DE_MIN[polarity]


def realize_many(thetas, polarity):
    """Realize a batch of thetas; a list in the same order, None where refused.

    Cache first, then one batched call for whatever is left. Both halves matter and the
    measurement says why. Batching alone, ignoring the cache, made the search THREE TIMES
    SLOWER: a trial re-proposes most of its candidates from one sitting to the next, so the
    cache was already answering the majority of them, and recomputing a batch is slower
    than not computing at all. Caching alone leaves a cold sitting paying 312 us of
    library-validation overhead per theme, which was 3.8 s of a 4 s trial.
    """
    table = np.asarray(thetas, dtype=float).reshape(-1, 9)
    keys = [theta_key(row, polarity) for row in table]
    missing = [i for i, key in enumerate(keys) if key not in REALIZE_CACHE]
    if missing:
        for i, theme in zip(missing, _realize_batch(table[missing], polarity), strict=True):
            REALIZE_CACHE[keys[i]] = theme
    return [REALIZE_CACHE[key] for key in keys]


def realize_uncached(theta, polarity):
    """One theme, ignoring the cache. Defined through the batch path so there is exactly
    one implementation of the colour rules -- two copies of a constraint set drift, and a
    drifted floor is a stimulus nobody chose."""
    return _realize_batch(np.asarray(theta, dtype=float)[None, :], polarity)[0]


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
