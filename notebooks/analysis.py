# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes -- correct for a notebook holding a model on the GPU, and
# fatal for a trial loop, whose whole point is that the next trial appears on click.
import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *A sidecar to calibrate-vision — that instrument measured what your eyes can distinguish;
    this one learns what they prefer.*

    # Calibrating the theme against your taste

    Legibility floors are measurable and measured; above them, theming has been guesswork.
    This notebook replaces the guess with a model: a latent **aesthetic utility** over a
    CAM16-UCS-parametrized theme space — page lightness and warmth, the accent set's hue,
    chroma, contrast and spread, how far comments recede, and VSCode's find-highlight as its
    own salience-versus-beauty axis. Each trial is one of three quick acts:

    - **A duel**: two candidate pages render the same real code in the fonts and pixel sizes
      you actually read. Click the one you would rather live in — trust the first pull.
    - **A comprehension probe**: one page, one instruction — *click the function name*. Your
      time to land on it measures what is truly easy to grasp, not what merely looks tidy.
    - **A find hunt**: the page shows search matches; click the current one. Time-to-find
      calibrates how loud `editor.findMatchBackground` must be before it stops earning its
      salience.

    Under the trials sits preferential Bayesian optimization: a Gaussian-process posterior
    over utility, a Bradley–Terry likelihood over your choices sharpened by reaction time
    (a fast, consistent click is strong evidence; a slow one reads as a near-tie, the way
    drift-diffusion models read decision time), and each duel *generated* to be maximally
    informative — the model's best guess against the challenger that would teach it most,
    with a small share of uniform probes as insurance against the model fooling itself.
    The candidates it chooses between are **bred fresh every trial**, not drawn from a fixed
    list: the themes it already rates highly, their mutated and recombined children, and a
    steady trickle of low-discrepancy newcomers, so the search can sit between any two
    themes it has shown you and can always still reach ground it has never visited.
    Every page is **code you have never seen** — generated, or lifted from a corner of the
    standard library — because a reused page turns time-to-find into a memory test.
    Your measured discrimination thresholds (from `calibration-responses.jsonl`, re-expressed
    in CAM16-UCS) and APCA/WCAG contrast floors are **hard constraints, never objectives**:
    every candidate you see is already legible; you are only ever asked which is *better*.

    Trials run in twenty-four-trial blocks per polarity (light page, dark page), each a run
    of sixteen duels, then four comprehension probes, then four find hunts — same-kind
    trials batched so one instruction serves a run and you never switch task mid-stride; a
    begin button gates each run. Blocks by polarity so your adaptation state is part of the
    measurement, not noise in it — and the **whole page**, not just the band, takes the
    ground under test, because in full screen the surround is most of what your eyes adapt
    to. A duel keeps the polarity's neutral surround, since the two candidates have
    different grounds and painting the page with either would advantage it; a single-card
    trial paints the page with the theme under test, which is what a theme owning the screen
    actually looks like. Every response appends to `aesthetics-responses.jsonl` beside this
    file with the full stimulus, the surround and both timestamps; sittings accumulate.

    Nothing is asked of you but clicks. Which colors you love is **inferred, never
    declared**: the prior mean carries only the field's general harmony models, and your own
    hues emerge from the duels — which is why the search deliberately keeps exploring hue
    rather than settling on lightness alone, and why a stated favourite would be worth less
    than a measured one anyway.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import math
    import random
    from datetime import datetime, timezone
    from pathlib import Path

    import colour
    import numpy as np
    import pandas as pd
    from scipy.stats import qmc

    LOG = Path(__file__).parent / "aesthetics-responses.jsonl"
    VISION_LOG = Path(__file__).parent / "calibration-responses.jsonl"
    # Where the instrument publishes its current answer for the applier to read. A result
    # that only exists as text on a page is a result someone has to retype.
    CHAMPION = Path(__file__).parent / "measured-theme.json"
    return CHAMPION, LOG, VISION_LOG, colour, datetime, json, math, np, pd, qmc, random, timezone


@app.cell(hide_code=True)
def _(colour, np):
    # The color engine. All appearance math runs in CAM16-UCS (Li et al. 2017 via
    # colour-science) under fixed, documented viewing conditions: D65 white, average
    # surround, L_A 40, Y_b 20 — a desktop monitor in a lit room. The screen itself is
    # uncalibrated (parked in the queue), which limits absolute claims, not the relative
    # structure the instrument learns.
    _VC = colour.VIEWING_CONDITIONS_CAM16["Average"]
    _WHITE_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    _XYZ_W = colour.xy_to_XYZ(_WHITE_XY) * 100.0
    _LA, _YB = 40.0, 20.0

    def hex_to_rgb(hexes):
        _h = [hexes] if isinstance(hexes, str) else list(hexes)
        return np.array([[int(s.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)] for s in _h])

    def rgb_to_hex(rgb):
        rgb = np.clip(np.atleast_2d(rgb), 0, 1)
        return ["#" + "".join(f"{round(255 * float(v)):02x}" for v in row) for row in rgb]

    def rgb_to_ucs(rgb):
        _xyz = colour.sRGB_to_XYZ(np.atleast_2d(rgb)) * 100.0
        _spec = colour.XYZ_to_CAM16(_xyz, _XYZ_W, L_A=_LA, Y_b=_YB, surround=_VC)
        return colour.JMh_CAM16_to_CAM16UCS(np.stack([_spec.J, _spec.M, _spec.h], axis=-1))

    def ucs_to_rgb(ucs):
        _jmh = colour.CAM16UCS_to_JMh_CAM16(np.atleast_2d(ucs))
        _spec = colour.CAM_Specification_CAM16(J=_jmh[..., 0], M=_jmh[..., 1], h=_jmh[..., 2])
        _xyz = colour.CAM16_to_XYZ(_spec, _XYZ_W, L_A=_LA, Y_b=_YB, surround=_VC)
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

    return apca_lc, composite, hex_to_rgb, rgb_to_hex, rgb_to_ucs, solve_j, ucs_dist, ucs_to_rgb, wcag


@app.cell(hide_code=True)
def _(VISION_LOG):
    # The observer is fit in ONE place — _observer.py, the home of the measurement<->
    # preference interlock. It refits lazily from the shared vision log (cached beside it),
    # so every new vision trial sharpens these constraints automatically and no instrument
    # carries its own copy of the model. v2 fits the psychometric slope, the lapse, a
    # chromatic confusion-axis rotation, and threshold as a smooth function of ground
    # lightness — all in CAM16-UCS, the same geometry this notebook searches.
    from _observer import fit as _observer_fit

    if VISION_LOG.exists():
        _fit = _observer_fit(VISION_LOG)
        DE_MIN = {"day": _fit.de_min_day, "night": _fit.de_min_night}
        THRESH_DETAIL = {"day": _fit.de_dir_day, "night": _fit.de_dir_night}
        VISION_N = _fit.n
    else:
        # No vision data on this machine: the v2.0 fit at 748 trials (2026-09-03),
        # flagged in the analysis so the substitution is never silent.
        DE_MIN = {"day": 3.2, "night": 2.5}
        THRESH_DETAIL = {"day": {}, "night": {}}
        VISION_N = 0
    return DE_MIN, THRESH_DETAIL, VISION_N


@app.cell(hide_code=True)
def _(DE_MIN, apca_lc, composite, hex_to_rgb, math, np, rgb_to_hex, rgb_to_ucs, solve_j, ucs_dist, ucs_to_rgb, wcag):
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
    _ANCHORS = {
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
    _ROLE_ORDER = ("keyword", "function", "string")

    def _anchor_polar(polarity):
        _ucs = rgb_to_ucs(hex_to_rgb([_ANCHORS[polarity][r] for r in _ROLE_ORDER]))
        _m = np.linalg.norm(_ucs[:, 1:], axis=1)
        _h = np.degrees(np.arctan2(_ucs[:, 2], _ucs[:, 1])) % 360
        return _h, _m

    _ANCHOR_HM = {p: _anchor_polar(p) for p in ("day", "night")}

    # Realization and prior are pure functions of (theta, polarity); the caches make the
    # per-trial local-refinement candidates (and every posterior call over the pool) pay
    # for their appearance math exactly once per kernel.
    _REALIZE_CACHE = {}
    _PRIOR_CACHE = {}

    def _theta_key(theta, polarity):
        return (tuple(round(float(_v), 6) for _v in theta), polarity)

    # NOTE (marimo name mangling, measured 2026-09-03): a cell-local (underscore) name
    # referenced from inside an exported function resolves only if it is defined ABOVE
    # that function in the cell — a later definition stays unmangled in the function body
    # and NameErrors at call time under `marimo run`, invisibly to script execution.
    # Helpers therefore precede their exported callers.
    def _realize_uncached(theta, polarity):
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
        _h0, _m0 = _ANCHOR_HM[polarity]
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
        _key = _theta_key(theta, polarity)
        if _key in _REALIZE_CACHE:
            return _REALIZE_CACHE[_key]
        _theme = _realize_uncached(theta, polarity)
        _REALIZE_CACHE[_key] = _theme
        return _theme

    # ------------------------------------------------------------------ the prior mean
    def _lab(hexes):
        _xyz = np.atleast_2d(hex_to_rgb(hexes))
        import colour as _colour

        return _colour.XYZ_to_Lab(
            _colour.sRGB_to_XYZ(_xyz),
            _colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
        )

    def _ou_luo_pair(lab1, lab2):
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
        _hl = (0.28 + 0.54 * math.tanh(-3.88 + 0.029 * (_L1 + _L2))) + (
            0.14 + 0.15 * math.tanh(-2 + 0.2 * abs(_L1 - _L2))
        )

        def _hsy(_L, _C, _h):
            _ec = 0.5 + 0.5 * math.tanh(-2 + 0.5 * _C)
            _hs = -0.08 - 0.14 * math.sin(math.radians(_h + 50)) - 0.07 * math.sin(math.radians(2 * _h + 90))
            _y = (90 - _h) / 10
            _ey = ((0.22 * _L - 12.8) / 10) * math.exp(min(_y - math.exp(_y), 50))
            return _ec * (_hs + _ey)

        return _hc + _hl + _hsy(_L1, _C1, _h1) + _hsy(_L2, _C2, _h2)

    def _raw_prior(theta, polarity, theme):
        _t = np.asarray(theta, dtype=float)
        _hx = [theme[r] for r in _ROLE_ORDER] + [theme["ground"]]
        _labs = _lab(_hx)
        _pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        _harm = float(np.mean([_ou_luo_pair(_labs[a], _labs[b]) for a, b in _pairs]))
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
    _pool_rng = np.random.default_rng(0xA55)
    POOL_THETA = _pool_rng.random((512, 9))
    POOL = {}
    _PRIOR_STATS = {}
    for _p in ("day", "night"):
        _items = []
        for _idx in range(len(POOL_THETA)):
            _th = POOL_THETA[_idx]
            _theme = realize(_th, _p)
            if _theme is not None:
                _items.append((_th, _theme, _raw_prior(_th, _p, _theme)))
        _pr = np.array([_it[2] for _it in _items])
        _PRIOR_STATS[_p] = (float(_pr.mean()), float(_pr.std() + 1e-9))
        POOL[_p] = [(_it[0], _it[1]) for _it in _items]

    def prior_mean(theta, polarity, theme=None):
        """Standardized prior utility (mean 0, sd 0.8 over the feasible pool) so the GP's
        signal variance, not the prior's arbitrary units, sets the scale."""
        _key = _theta_key(theta, polarity)
        if _key in _PRIOR_CACHE:
            return _PRIOR_CACHE[_key]
        theme = theme or realize(theta, polarity)
        if theme is None:
            _val = 0.0
        else:
            _m, _s = _PRIOR_STATS[polarity]
            _val = 0.8 * (_raw_prior(theta, polarity, theme) - _m) / _s
        _PRIOR_CACHE[_key] = _val
        return _val

    return AXES, POOL, POOL_THETA, prior_mean, realize


@app.cell(hide_code=True)
def _():
    import html as _html
    import io as _io
    import keyword as _kw
    import tokenize as _tokenize

    import _codegen

    # Stimuli are real code, embedded verbatim from this repo's own notebooks (07's
    # training loop, 05's model, _palette's tint) — the code Titus actually reads, not
    # lorem ipsum. Embedded rather than read at render time so the stimulus set is stable
    # across sessions; each record carries the snippet id.
    _SOURCES = {
        "train-loop": (
            "07-optimization-loop.py",
            """def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = (loss.item(), batch * 64 + len(X))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
""",
            "loss",
        ),
        "build-model": (
            "05-build-model.py",
            """class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
""",
            "nn",
        ),
        "tint": (
            "_palette.py",
            '''def tint(color, toward_white):
    """The palette hue mixed toward a card's white, as a literal hex.

    For renderers that cannot take scheme names (graphviz, raw CSS):
    fills stay derived from the constants above instead of hand-tuned
    hexes appearing per notebook.
    """
    channels = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * toward_white):02x}" for c in channels)
''',
            "color",
        ),
        "tensor-ops": (
            "02-tensors.py",
            """y2 = tensor_2.matmul(tensor_2.T)
torch.matmul(tensor_2, tensor_2.T, out=y3)
z1 = tensor_2 * tensor_2
z2 = tensor_2.mul(tensor_2)
# This computes the element-wise product; z1, z2 will match
torch.mul(tensor_2, tensor_2, out=z3)
t1 = torch.cat([tensor_2, tensor_2, tensor_2], dim=1)
agg = tensor_2.sum()
agg_item = agg.item()
""",
            "tensor_2",
        ),
    }

    def _tokenize_roles(code):
        """Role spans via the stdlib tokenizer: (text, role, line, col). Definition and
        call names are `function`; control words `keyword`; strings and numbers are the
        one literal family; dotted-name reads and everything else recede as variable/punct."""
        _spans = []
        _toks = list(_tokenize.generate_tokens(_io.StringIO(code).readline))
        _prev_sig = None
        for _i, _tok in enumerate(_toks):
            _typ, _txt, (_sr, _sc), (_er, _ec), _ = _tok
            if _typ in (_tokenize.NEWLINE, _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT, _tokenize.ENDMARKER):
                continue
            if _typ == _tokenize.COMMENT:
                _role = "comment"
            elif _typ == _tokenize.STRING or _typ in (
                getattr(_tokenize, "FSTRING_START", -1),
                getattr(_tokenize, "FSTRING_MIDDLE", -2),
                getattr(_tokenize, "FSTRING_END", -3),
            ):
                _role = "string"
            elif _typ == _tokenize.NUMBER:
                _role = "number"
            elif _typ == _tokenize.OP:
                _role = "punct"
            elif _typ == _tokenize.NAME:
                if _kw.iskeyword(_txt):
                    _role = "keyword"
                elif _prev_sig in ("def", "class"):
                    _role = "function"
                else:
                    _nxt = next(
                        (_t2 for _t2 in _toks[_i + 1 :] if _t2.type not in (_tokenize.NL, _tokenize.NEWLINE)),
                        None,
                    )
                    _role = "function" if (_nxt is not None and _nxt.string == "(") else "variable"
            else:
                _role = "variable"
            _spans.append({"text": _txt, "role": _role, "sr": _sr, "sc": _sc, "er": _er, "ec": _ec})
            if _typ == _tokenize.NAME or (_typ == _tokenize.OP and _txt in "()[]{}.,:"):
                _prev_sig = _txt
        return _spans

    # One page per trial, never the same twice: _codegen writes it. The four embedded
    # sources above stay as the cold-start and as a familiarity CONTROL -- a page he knows
    # is the reference against which a fresh page's reaction time is read -- but they are
    # no longer the corpus. Memoized per seed so the widget, the recorder and the analysis
    # cell all resolve the same page without regenerating it.
    # _CONTROL is built BEFORE snippet_for on purpose: a cell-local name referenced
    # inside an exported function resolves only if it is defined above that function,
    # and only under `marimo run`/`edit` -- a script run shares one namespace and
    # never mangles, so the wrong order passes every check and fails only when served.
    _CONTROL = []
    for _sid, (_prov, _code, _ident) in _SOURCES.items():
        _sp = _tokenize_roles(_code)
        _CONTROL.append(
            {
                "id": _sid,
                "provenance": _prov,
                "code": _code,
                "spans": _sp,
                "fn_ids": [_i for _i, _s in enumerate(_sp) if _s["role"] == "function"],
                "ident": _ident,
                "ident_ids": [_i for _i, _s in enumerate(_sp) if _s["text"] == _ident],
                "hash": f"control-{_sid}",
                "kind": "control",
            }
        )

    _SNIP_MEMO = {}

    def snippet_for(seed, width=None, target_kind=None, lines=None):
        """The page for this trial seed: fresh procedural or obscure-stdlib code.

        width is the column ceiling: two duel cards side by side hold about eighty columns
        at 14px, and the stimulus <pre> is overflow:hidden, so a wider line would be
        silently clipped -- a clipped stimulus is a different stimulus. Line count and role
        mix stay at the generator's calibrated default: freshness alone makes the
        comprehension probe hard now that no page is ever shown twice, and a longer page
        would trade away the identical-role-statistics property that lets two reaction
        times be compared at all.
        """
        _key = (int(seed), width, target_kind, lines)
        if _key in _SNIP_MEMO:
            return _SNIP_MEMO[_key]
        # Width and length are PREFERENCES; freshness is the requirement. The generator
        # cannot promise every shape for every seed -- a narrow 28-line page is a tall
        # order, and asking for 64 columns alone lost half the seeds -- so the request
        # relaxes in a declared order: hold the narrow column and shorten, then widen a
        # step and shorten again, and only if every combination fails fall back to a
        # control page, which is code he has already seen and therefore the last resort.
        _lines_ladder = [int(lines), int(lines) - 4, int(lines) - 8, None] if lines else [None]
        _width_ladder = [int(width), int(width) + 8, int(width) + 16, None] if width else [None]
        _s = None
        for _w in _width_ladder:
            for _ln in _lines_ladder:
                try:
                    _kw = {}
                    if _w:
                        _kw["max_width"] = _w
                    if _ln:
                        _kw["lines"] = _ln
                    if target_kind:
                        _kw["target_kind"] = target_kind
                    _s = dict(_codegen.snippet(int(seed), **_kw))
                    _s.setdefault("ident", _s.get("target"))
                    break
                except Exception:
                    continue
            if _s is not None:
                break
        if _s is None:
            _s = _CONTROL[int(seed) % len(_CONTROL)]
        _SNIP_MEMO[_key] = _s
        return _s

    _PROSE_TAIL = (
        "The consumer holds the lock only while it copies out, so a slow reader delays the "
        "next fill rather than corrupting the one in flight."
    )
    _OUTPUT_TAIL = "queue depth 3  drained 1284  blocked 0.4%  last fill 2.1 ms"
    _PROSE = (
        "A buffer is filled once per frame and drained by the consumer thread; the queue "
        "length bounds how far the two can drift apart before a reader blocks."
    )

    # The three surfaces Titus actually reads. A theme is one theme, but it is *seen* in
    # three arrangements, and the one that wins on a bare code page need not win where
    # prose and code interleave. Surface is a stimulus factor, not a theme axis: utility
    # stays defined over the theme, and the surface is logged so a later analysis can test
    # for a surface-by-theme interaction rather than assuming there is none.
    #
    #   editor    a page of code with a line of prose above it -- the plain editor
    #   panel     the Claude Code chat surface: serif turns, a raised code card between
    #             them, the proportions of an assistant answer
    #   notebook  the marimo/VSCode notebook: a centred prose column at the measured 42rem
    #             reading measure, then a raised code card, then an output block
    SURFACES = ("editor", "panel", "notebook")

    # And the size he ACTUALLY reads each one at, from his own settings.jsonc: the global
    # editor.fontSize is unset so ordinary editors sit at VSCode's default 14, notebook code
    # cells are customised to 16, and the chat panel's code follows the editor at 14.
    #
    # This matters more than it looks. Duels ran at 12 and 13px on the reasoning that a full
    # screen wants small type -- but 12 and 13 are sizes he never reads code at, so a
    # preference measured there was being applied to reading at 14 and 16. Contrast
    # sensitivity falls with glyph scale, which is exactly why the colour floors are doubled
    # against the 104px threshold, so "measure at one size, apply at another" is not a free
    # assumption in a colour experiment. Each surface is now shown at its true size, which
    # also stops size and surface from being independently varied for no reason: in his real
    # day they covary, and it is the real pairing whose theme is wanted.
    READING_PX = {"editor": 14, "panel": 14, "notebook": 16}

    def render_card(theme, snippet, code_px, find_current=None, task=False, prose=True, surface="editor"):
        """One candidate page as HTML: prose in IBM Plex Serif 17px, code in Iosevka at the
        true editor pixel size, on the candidate ground. find_current=None hides the find
        layer; an int marks that occurrence as the current match, the rest as plain
        highlights. task=True makes every span a click target (data-tid), visually inert.
        surface selects the arrangement (see SURFACES above)."""
        _lines = snippet["code"].split("\n")
        _cursor = {}
        _out = []
        _card_open, _card_close = "", ""
        if surface in ("panel", "notebook"):
            # Machine text sits on a raised card a step off the page, the grammar the
            # applied theme uses: flat tinted panel means aside, raised card means
            # artifact. The step is taken in the ground's own hue, never toward grey.
            _g = theme["ground"].lstrip("#")
            _rgb = [int(_g[_k : _k + 2], 16) for _k in (0, 2, 4)]
            _dark = sum(_rgb) < 384
            _step = 12 if _dark else -10
            _card_bg = "#" + "".join(f"{max(0, min(255, _v + _step)):02x}" for _v in _rgb)
            _edge = "#" + "".join(f"{max(0, min(255, _v + (26 if _dark else -22))):02x}" for _v in _rgb)
            _shadow = "0 1px 3px -1px rgba(0,0,0,.35), 0 5px 14px -6px rgba(0,0,0,.28)"
            _card_open = (
                f'<div style="background:{_card_bg};border:1px solid {_edge};border-radius:4px;'
                f'padding:12px 14px;box-shadow:{_shadow};overflow:hidden">'
            )
            _card_close = "</div>"
        if prose:
            _measure = "42rem" if surface == "notebook" else "34em"
            _centre = "margin:0 auto 14px auto" if surface == "notebook" else "margin:0 0 14px 0"
            _out.append(
                f"<div style=\"font-family:'IBM Plex Serif',serif;font-size:17px;line-height:1.6;"
                f'color:{theme["ink"]};max-width:{_measure};{_centre}">{_html.escape(_PROSE)}</div>'
            )
        _out.append(_card_open)
        _out.append(
            f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;font-size:{code_px}px;"
            f'line-height:1.5;margin:0;white-space:pre;overflow:hidden;color:{theme["punct"]}">'
        )
        _find_ids = set(snippet["ident_ids"]) if find_current is not None else set()
        for _i, _s in enumerate(snippet["spans"]):
            _r, _c = _s["sr"] - 1, _s["sc"]
            _pr, _pc = _cursor.get("r", 0), _cursor.get("c", 0)
            while _pr < _r:
                _out.append("\n")
                _pr, _pc = _pr + 1, 0
            if _c > _pc:
                _out.append(_lines[_r][_pc:_c])
            _style = f"color:{theme[_s['role']]}"
            if _s["role"] == "comment":
                _style += ";font-style:italic"
            if _i in _find_ids:
                _fill = theme["find_current"] if _i == find_current else theme["find_other"]
                _style += f";background:{_fill};border-radius:2px"
            _tid = f' data-tid="{_i}"' if task else ""
            _out.append(f'<span style="{_style}"{_tid}>{_html.escape(_s["text"])}</span>')
            _cursor = {"r": _s["er"] - 1, "c": _s["ec"]}
        _out.append("</pre>")
        _out.append(_card_close)
        if surface == "panel":
            # The diff card, because a Claude Code turn is mostly diffs and their colours
            # are part of what he reads all day. Both backgrounds are DERIVED, not searched:
            # the theme already carries a cool role colour and a warm one, and mixing each
            # into the ground keeps added/removed on the cool/warm polarity that survives
            # colour-vision deficiency while adding no dimension to a nine-dimensional
            # space that is already the binding constraint on convergence. Line text stays
            # the code ink -- a diff recolours the field, never the code.
            def _mix(_hex, _t):
                _a = theme["ground"].lstrip("#")
                _b = _hex.lstrip("#")
                return "#" + "".join(
                    f"{round(int(_a[_k : _k + 2], 16) * (1 - _t) + int(_b[_k : _k + 2], 16) * _t):02x}"
                    for _k in (0, 2, 4)
                )

            _add_bg, _del_bg = _mix(theme["function"], 0.16), _mix(theme["string"], 0.16)
            _sign = theme["comment"]
            _diff = [
                ("-", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table)", _del_bg),
                ("+", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table, 12)", _add_bg),
                (" ", "    with sift_gantry(opaline_voussoir_walk) as vernal_cistern_gate:", None),
                ("+", "        prime_mullion_stub = 128", _add_bg),
            ]
            _rows = []
            for _mark, _text, _bg in _diff:
                _style = f"display:block;padding:0 6px;color:{theme['punct']}"
                if _bg:
                    _style += f";background:{_bg}"
                _rows.append(
                    f'<span style="{_style}"><span style="color:{_sign}">{_mark}</span>{_html.escape(_text)}</span>'
                )
            _out.append(
                f"{_card_open}<div style=\"font-family:'IBM Plex Serif',serif;font-size:13px;"
                f'color:{theme["comment"]};margin:0 0 6px 0">edited _codegen.py</div>'
                f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;"
                f"font-size:{code_px}px;line-height:1.5;margin:0;white-space:pre;"
                f'overflow:hidden">' + "".join(_rows) + f"</pre>{_card_close}"
            )
        if surface == "panel" and prose:
            # An assistant turn continues after the code: the second serif block is what
            # makes this the chat surface rather than a card on a page.
            _out.append(
                f"<div style=\"font-family:'IBM Plex Serif',serif;font-size:17px;line-height:1.6;"
                f'color:{theme["ink"]};max-width:34em;margin:12px 0 0 0">'
                f"{_html.escape(_PROSE_TAIL)}</div>"
            )
        if surface == "notebook":
            # A notebook cell is code plus its output, so the output block is part of the
            # stimulus: mono, one step of ink below the code, on the page rather than the card.
            _out.append(
                f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;"
                f"font-size:{code_px}px;line-height:1.5;margin:8px 0 0 0;"
                f'color:{theme["comment"]};white-space:pre;overflow:hidden">'
                f"{_html.escape(_OUTPUT_TAIL)}</pre>"
            )
        return "".join(_out)

    DUEL_WIDTH = _codegen.DUEL_WIDTH
    return DUEL_WIDTH, READING_PX, SURFACES, render_card, snippet_for


@app.cell(hide_code=True)
def _(LOG, json, mo):
    _existing = [json.loads(_line) for _line in LOG.read_text().splitlines() if _line.strip()] if LOG.exists() else []
    get_responses, set_responses = mo.state(_existing)
    # The first trial of a sitting (and of every run) is gated behind a begin button;
    # inside a run the previous click anchors the clock, so render time is the baseline.
    SESSION_START_N = len(_existing)
    return SESSION_START_N, get_responses, set_responses


@app.cell(hide_code=True)
def _(DUEL_WIDTH, POOL, READING_PX, SURFACES, math, np, prior_mean, qmc, random, realize):
    # The preference model: a Gaussian process over theme space with a Bradley-Terry
    # likelihood on duels, fit by Laplace approximation — Chu & Ghahramani's preferential
    # GP, QUEST+'s generate-the-most-informative-trial loop on top. Reaction time enters
    # the likelihood drift-diffusion-style: decision time falls as the utility gap grows,
    # so a fast click steepens that duel's slope and a slow one flattens it toward a tie.
    # Length-scales are ARD: one per axis, estimated from the data rather than fixed, so
    # axes his choices ignore get long scales and stop costing sample efficiency. Nine
    # dimensions at ~100 duels is the binding constraint on how fast this converges, and
    # ARD is the cheapest honest way to shrink the effective dimension.
    _LS0 = np.array([0.35] * 9 + [0.9])
    _SF2 = 4.0

    def _kmat(A, B, ls=None):
        _l = _LS0 if ls is None else ls
        _d2 = (((A[:, None, :] - B[None, :, :]) / _l) ** 2).sum(-1)
        _r = np.sqrt(_d2 + 1e-12)
        return _SF2 * (1 + np.sqrt(5) * _r + 5 * _r**2 / 3) * np.exp(-np.sqrt(5) * _r)

    def _ard_scales(X, duels, lam):
        """Per-axis length-scales from a ridge-regularized linear Bradley-Terry fit.

        The principled route is maximizing the Laplace log-marginal-likelihood over ten
        log-length-scales, which costs a hundred-odd GP refits per trial and would make
        the instrument wait on itself. A linear BT model on the winner-minus-loser axis
        differences is the same question asked cheaply -- which axes move his choices --
        and its coefficient magnitudes plug straight in as relevances. Empirical-Bayes
        shortcut, deliberately: the fit runs in milliseconds and the GP keeps the
        nonlinearity.
        """
        # Shrinkage toward isotropy, because relevance is not identifiable early: with 60
        # duels the estimated ranking of nine axes was measured to be noise (0 of 4
        # simulated runs recovered the truly active axes, against reliable recovery at
        # 400). Blending toward the isotropic default with weight n/160 keeps a thin log
        # from distorting the kernel and converges on full ARD as duels accumulate.
        if len(duels) < 12:
            return _LS0.copy()
        _w_ard = min(1.0, len(duels) / 160.0)
        _D = np.array([(X[_w] - X[_l]) * _lm for (_w, _l), _lm in zip(duels, lam, strict=True)])
        _w = np.zeros(_D.shape[1])
        for _ in range(60):
            _p = 1.0 / (1.0 + np.exp(-(_D @ _w)))
            _g = _D.T @ (1.0 - _p) - 2.0 * _w
            _H = -(_D.T * (_p * (1 - _p))) @ _D - 2.0 * np.eye(_D.shape[1])
            _step = np.linalg.solve(_H, -_g)
            _w = _w + _step
            if np.abs(_step).max() < 1e-10:
                break
        _rel = np.abs(_w) / max(float(np.abs(_w).max()), 1e-9)
        _ls = 0.30 / np.sqrt(np.clip(_rel, 0.10, 1.0))
        _ls = np.clip(_ls, 0.25, 1.4)
        _ls = (1.0 - _w_ard) * _LS0 + _w_ard * _ls
        _ls[9] = 0.9
        return _ls

    def _coords(theta, polarity):
        return np.concatenate([np.asarray(theta, dtype=float), [1.0 if polarity == "night" else 0.0]])

    def duels_from(responses, rt_p=0.5):
        """(X, duel index pairs, per-duel slopes, prior mean at X) from the log's duels."""
        _pts, _index = [], {}
        _duels, _rts, _paused, _sides = [], [], [], []
        for _r in responses:
            if _r.get("mode") != "duel" or _r.get("choice") not in (0, 1):
                continue
            _ids = []
            for _th in (_r["theta_a"], _r["theta_b"]):
                _key = (tuple(round(float(_v), 6) for _v in _th), _r["polarity"])
                if _key not in _index:
                    _index[_key] = len(_pts)
                    _pts.append(_coords(_th, _r["polarity"]))
                _ids.append(_index[_key])
            _win = _ids[_r["choice"]]
            _lose = _ids[1 - _r["choice"]]
            _duels.append((_win, _lose))
            _rts.append(float(_r.get("rt_ms", 2500.0)))
            _paused.append(bool(_r.get("paused")))
            # Which SIDE the winner was displayed on. Measured 2026-09-03 over 79 duels:
            # he picks the right-hand card 61% of the time (z = -1.91 against no bias).
            # Unmodelled, that lands on the utility as noise; as a fitted term it is
            # subtracted out. Reconstructible from the log, so no past duel is wasted.
            _shown = (1 - _r["choice"]) if _r.get("swap") else _r["choice"]
            _sides.append(1.0 if _shown == 0 else -1.0)
        if not _pts:
            return None
        _X = np.array(_pts)
        _paused = np.array(_paused)
        _clean = np.array(_rts)[~_paused]
        _rt_med = float(np.median(_clean)) if len(_clean) >= 8 else 2500.0
        # The exponent is FITTED, not assumed (see rt_exponent below). p = 0.5 was a
        # hand-rolled square root; p = 0 means the clock is ignored entirely, so the same
        # search that calibrates this channel also tests whether it earns its keep.
        _lam = np.clip((_rt_med / np.maximum(np.array(_rts), 200.0)) ** rt_p, 0.6, 1.8)
        # A paused trial's time says nothing about the utility gap: its choice still counts,
        # at the neutral slope, neither sharpened nor flattened by the clock.
        _lam[_paused] = 1.0
        _m = np.array([prior_mean(_x[:9], "night" if _x[9] > 0.5 else "day") for _x in _X])
        return _X, _duels, _lam, _m, np.array(_sides)

    def fit_laplace(X, duels, lam, m, sides=None, ls=None):
        """Laplace posterior over utilities, alternating with the position-bias term.

        delta is one number shared by every duel: the log-odds advantage of the card on
        the left. f and delta are identifiable because side is randomized independently
        of theme, and they are fitted by alternation -- f given delta by Newton, then
        delta given f by its own one-dimensional Newton -- which converges in two or
        three rounds at this scale.
        """
        _n = len(X)
        _K = _kmat(X, X, ls) + 1e-6 * np.eye(_n)
        _Ki = np.linalg.inv(_K)
        _f = m.copy()
        _W = np.zeros((_n, _n))  # replaced each Newton step; kept for the final _cov
        _sd = np.zeros(len(duels)) if sides is None else np.asarray(sides, dtype=float)
        _delta = 0.0
        for _round in range(3):
            # One BLAS product per Newton step instead of a Python loop over duels. Each
            # duel contributes q_k (e_win - e_lose)(e_win - e_lose)^T to the Hessian, which
            # is exactly D^T diag(q) D for the difference matrix D -- so the whole update is
            # two matrix products. Measured on the live log: the loop cost 108 ms per fit,
            # an np.add.at scatter cost 166 ms (add.at is unbuffered and slow), and this
            # costs 128 ms -- SLOWER than the loop at today's 121 duels, because building D
            # dominates at this size. Kept anyway: the loop pays one interpreter trip per
            # duel per Newton step, so it degrades linearly in log length where this is one
            # BLAS call, and 20 ms is noise against a 350 ms trial. Revisit only if a fit
            # ever dominates again. Identical arithmetic either way -- the recovery tests
            # reproduce every number.
            _D = np.zeros((len(duels), _n))
            for _k, (_w, _l) in enumerate(duels):
                _D[_k, _w] += 1.0
                _D[_k, _l] -= 1.0
            _lm_v = np.asarray(lam, dtype=float)
            _Dl = _D * _lm_v[:, None]
            for _ in range(60):
                _z = _Dl @ _f + _delta * _sd
                _p = 1.0 / (1.0 + np.exp(-_z))
                _g = _Dl.T @ (1.0 - _p)
                _q = _lm_v * _lm_v * _p * (1.0 - _p)
                _W = (_D * _q[:, None]).T @ _D
                _step = np.linalg.solve(_Ki + _W, _g - _Ki @ (_f - m))
                _f = _f + _step
                if np.abs(_step).max() < 1e-8:
                    break
            if sides is None or len(duels) < 12:
                break
            _gap = _Dl @ _f
            for _ in range(40):
                _p = 1.0 / (1.0 + np.exp(-(_gap + _delta * _sd)))
                _gd = float(_sd @ (1.0 - _p)) - 4.0 * _delta
                _hd = -float((_sd * _sd) @ (_p * (1 - _p))) - 4.0
                _d_step = -_gd / _hd
                _delta = float(np.clip(_delta + _d_step, -2.0, 2.0))
                if abs(_d_step) < 1e-10:
                    break
        _cov = np.linalg.inv(_Ki + _W)
        return _f, _cov, _Ki, _delta

    def predict(X, f, m, cov, Ki, Xs, ms, ls=None):
        _ks = _kmat(Xs, X, ls)
        _mu = ms + _ks @ (Ki @ (f - m))
        _A = Ki - Ki @ cov @ Ki
        _var = np.maximum(_SF2 - np.einsum("ij,jk,ik->i", _ks, _A, _ks), 1e-9)
        return _mu, _var, _ks, _A

    def posterior_joint(fit, thetas, polarity):
        """Mean and FULL covariance over candidates -- what P(best) needs.

        Marginal variances cannot answer "which of these is the best theme": candidates
        near each other in theme space share almost all their uncertainty, and ignoring
        that correlation would scatter the probability of being best across a cluster of
        effectively identical pages.
        """
        _Xs = np.array([_coords(_t, polarity) for _t in thetas])
        _ms = np.array([prior_mean(_t, polarity) for _t in thetas])
        _ls = fit.get("ls")
        _ks = _kmat(_Xs, fit["X"], _ls)
        _mu = _ms + _ks @ (fit["Ki"] @ (fit["f"] - fit["m"]))
        _A = fit["Ki"] - fit["Ki"] @ fit["cov"] @ fit["Ki"]
        _cov = _kmat(_Xs, _Xs, _ls) - _ks @ _A @ _ks.T
        _cov = 0.5 * (_cov + _cov.T) + 1e-8 * np.eye(len(thetas))
        return _mu, _cov

    def _h2(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    _GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(9)
    _GH_W = _GH_W / _GH_W.sum()

    def _posterior_over(fit, thetas, polarity):
        _X, _duels, _lam, _m = fit["X"], fit["duels"], fit["lam"], fit["m"]
        _Xs = np.array([_coords(_t, polarity) for _t in thetas])
        _ms = np.array([prior_mean(_t, polarity) for _t in thetas])
        return predict(_X, fit["f"], _m, fit["cov"], fit["Ki"], _Xs, _ms, fit.get("ls"))

    _FIT_MEMO = {}

    def cv_logloss(responses, rt_p, folds=5, seed=0):
        """Held-out log-loss of predicted duel outcomes at a given RT exponent.

        Cross-validation rather than marginal likelihood: the Laplace approximation makes
        the latter awkward to compare across likelihoods, while held-out predictive accuracy
        asks the question that matters -- does weighting a duel by how fast he answered it
        predict his NEXT answer better than ignoring the clock?
        """
        _d = duels_from(responses, rt_p)
        if _d is None:
            return None
        _X, _duels, _lam, _m, _sides = _d
        if len(_duels) < 5 * folds:
            return None
        _rng = np.random.default_rng(seed)
        _order = _rng.permutation(len(_duels))
        _ls = _ard_scales(_X, _duels, _lam)
        _total, _n = 0.0, 0
        for _k in range(folds):
            _test = set(_order[_k::folds].tolist())
            _tr = [_i for _i in range(len(_duels)) if _i not in _test]
            if len(_tr) < 8:
                continue
            _f, _cov, _Ki, _delta = fit_laplace(_X, [_duels[_i] for _i in _tr], _lam[_tr], _m, _sides[_tr], _ls)
            for _i in _test:
                _w, _l = _duels[_i]
                _z = _lam[_i] * (_f[_w] - _f[_l]) + _delta * _sides[_i]
                _p = 1.0 / (1.0 + np.exp(-_z))
                _total -= np.log(max(_p, 1e-9))
                _n += 1
        return None if _n == 0 else _total / _n

    _RTP_MEMO = {}

    def rt_exponent(responses, grid=(0.0, 0.25, 0.5, 0.75), refit_every=25):
        """The RT exponent that predicts his next answer best, refit occasionally.

        Returns (best exponent, {exponent: held-out log-loss}). Zero is in the grid on
        purpose: if ignoring the clock predicts as well, the channel is noise dressed as
        evidence and the model should say so rather than carry a flattering heuristic.
        """
        _nd = sum(1 for _r in responses if _r.get("mode") == "duel" and _r.get("choice") in (0, 1))
        _bucket = _nd // refit_every
        if _bucket in _RTP_MEMO:
            return _RTP_MEMO[_bucket]
        _scores = {}
        for _p in grid:
            _v = cv_logloss(responses, _p)
            if _v is not None:
                _scores[_p] = _v
        _out = (0.5, {}) if not _scores else (min(_scores, key=_scores.get), _scores)
        if len(_RTP_MEMO) > 3:
            _RTP_MEMO.pop(next(iter(_RTP_MEMO)))
        _RTP_MEMO[_bucket] = _out
        return _out

    def fitted(responses, rt_p=None):
        # Keyed by how many duels have been answered: the fit is a pure function of the
        # log, three cells ask for the same one, and it is the cubic-cost step. Only the
        # newest entry is kept -- an older fit is never asked for again.
        _key = sum(1 for _r in responses if _r.get("mode") == "duel" and _r.get("choice") in (0, 1))
        if rt_p is None:
            rt_p = rt_exponent(responses)[0]
        _key = (_key, rt_p)
        if _key in _FIT_MEMO:
            return _FIT_MEMO[_key]
        _d = duels_from(responses, rt_p)
        if _d is None:
            return None
        _X, _duels, _lam, _m, _sides = _d
        _ls = _ard_scales(_X, _duels, _lam)
        _f, _cov, _Ki, _delta = fit_laplace(_X, _duels, _lam, _m, _sides, _ls)
        _out = {
            "X": _X,
            "duels": _duels,
            "lam": _lam,
            "m": _m,
            "f": _f,
            "cov": _cov,
            "Ki": _Ki,
            "ls": _ls,
            "delta": _delta,
            "sides": _sides,
            "rt_p": rt_p,
        }
        # A few entries rather than one: the progress readout fits the log as it stood some
        # duels ago and compares, which needs two fits alive at once.
        if len(_FIT_MEMO) > 4:
            _FIT_MEMO.pop(next(iter(_FIT_MEMO)))
        _FIT_MEMO[_key] = _out
        return _out

    def mu_at(fit, thetas, polarity):
        """Posterior-mean utility at arbitrary thetas — the analysis cell's window in."""
        return _posterior_over(fit, thetas, polarity)[0]

    # ---- candidate generation: global reach PLUS bred refinement --------------------
    #
    # The pool was 512 points drawn once with a fixed seed, and the only refinement was 48
    # jittered children of the single argmax champion. Measured against a synthetic
    # two-mode utility (see the escape test in the commit that added this), that design
    # has good REACH -- 512 uniform points cover nine dimensions well enough for Thompson
    # sampling to discover a distant better mode -- and poor RESOLUTION: nothing can sit
    # between pool points except near one champion, at a fixed step size.
    #
    # The first attempt at fixing it replaced the pool with bred children and lost the
    # reach, scoring *worse* in simulation. So candidates are now reach and refinement
    # together, every trial:
    #
    #   standing    the full pool plus a SMALL fresh scrambled-Sobol block (64), advanced
    #               by trial number. The pool is a codebook: revisiting the same points
    #               concentrates information there and sharpens the posterior, where a
    #               fully churning candidate set spreads every duel over ground never
    #               seen again -- measured, a 512-per-trial immigrant flood scored worse
    #               than no immigrants at all. Sixty-four is the measured sweet spot: a
    #               trickle of genuinely new ground each trial, never enough to drown the
    #               codebook, and enough that no region stays permanently unvisited.
    #   elites      the best already-evaluated themes, chosen for spread as well as for
    #               posterior mean, so refinement is not confined to one basin.
    #   mutation    Gaussian children of each elite, per-axis sigma proportional to the
    #               ARD length-scale: fine steps where utility actually turns, coarse
    #               where the model has learned that nothing rides.
    #   crossover   uniform per-axis recombination between elite pairs. Worth having
    #               because the axes are semi-separable (ground, accent set, comment
    #               recession, find-highlight): a good ground and a good accent set
    #               recombine into a plausible page, the building-block case where
    #               crossover earns its keep rather than adding noise.
    #
    # Infeasible children are dropped by the floors rather than penalized, so the whole
    # candidate set is legible-by-construction.
    def _sobol_block(n_log2, offset_blocks):
        """A power-of-two block from one fixed scrambled Sobol sequence.

        Deterministic in the block index, so trial n always draws the same immigrants and
        successive trials continue the sequence instead of resampling the same clumps.
        random() rather than random_base2(): the latter also demands that the TOTAL drawn
        be a power of two, which a fast-forwarded engine cannot satisfy. n itself is a
        power of two, which is what the balance property needs.
        """
        _n = 2**n_log2
        _eng = qmc.Sobol(d=9, scramble=True, seed=0xC0FFEE)
        _skip = (offset_blocks * _n) % 65536
        if _skip:
            _eng.fast_forward(_skip)
        return _eng.random(_n)

    def candidates(fit, polarity, nprng, n_trial=0, n_elite=10, n_mut=20, n_cross=48, imm_log2=6):
        """(candidates, index where the standing global stratum ends) for this trial."""
        _out, _seen = [], set()

        def _add(_t, _theme=None):
            _t = np.clip(np.asarray(_t, dtype=float), 0.0, 1.0)
            _key = tuple(np.round(_t, 4))
            if _key in _seen:
                return
            _th = _theme if _theme is not None else realize(_t, polarity)
            if _th is None:
                return
            _seen.add(_key)
            _out.append((_t, _th))

        for _t, _theme in POOL[polarity]:
            _add(_t, _theme)
        for _imm in _sobol_block(imm_log2, n_trial):
            _add(_imm)
        _n_standing = len(_out)
        if fit is None:
            return _out, _n_standing
        _want = 1.0 if polarity == "night" else 0.0
        _arch = [_x[:9] for _x in fit["X"] if abs(_x[9] - _want) < 0.5]
        _seed_set = _arch + [_c[0] for _c in _out]
        _mu = _posterior_over(fit, _seed_set, polarity)[0]
        _ls = fit.get("ls")
        _top = np.argsort(-_mu)[: 6 * n_elite]
        # Elites for spread as well as for mean: the best few, then the most different
        # among the rest of the leaders, so refinement is not confined to one basin.
        # Deliberately NOT Thompson-sampled elites: tried, and measured clearly worse
        # (reach 3/12 runs, t = -2.6). Refining around a high-variance region spends the
        # mutation budget on noise and displaces elites that are actually good; explore
        # belongs in the standing stratum, refine belongs where the mean is high.
        _keep = [int(_i) for _i in _top[: n_elite // 2]]
        _w = 1.0 / (_LS0[:9] if _ls is None else _ls[:9])
        _P = np.array([np.asarray(_seed_set[int(_i)]) * _w for _i in _top])
        _top_list = list(_top)
        while len(_keep) < n_elite and len(_keep) < len(_top):
            _chosen = [_top_list.index(_i) for _i in _keep if _i in _top_list]
            if not _chosen:
                _chosen = [0]
            _d = np.min(np.linalg.norm(_P[:, None, :] - _P[None, _chosen, :], axis=-1), axis=1)
            _d[_chosen] = -1.0
            _keep.append(int(_top[int(np.argmax(_d))]))
        _elites = [np.asarray(_seed_set[_i]) for _i in _keep]
        _sig = 0.25 * (_LS0[:9] if _ls is None else _ls[:9])
        for _e in _elites:
            _add(_e)
            for _child in np.clip(_e[None, :] + nprng.normal(0, _sig, (n_mut, 9)), 0, 1):
                _add(_child)
        if len(_elites) >= 2:
            for _ in range(n_cross):
                _i, _j = nprng.choice(len(_elites), 2, replace=False)
                _mask = nprng.random(9) < 0.5
                _add(np.where(_mask, _elites[_i], _elites[_j]))
        return _out, _n_standing

    # ---- the legibility surface: what the timed arms are FOR ---------------------------
    #
    # Until now the comprehension probes and find hunts were only described in the analysis
    # -- a median, a slope -- and never touched the verdict, so a third of every sitting's
    # clicks bought nothing. They measure a different quantity from preference: not which
    # page he would rather live in, but how fast he can actually find a name in it. So they
    # get their own function over the same theme space.
    #
    # A Gaussian process on log time-to-click, which is closed-form (no Laplace, no Newton)
    # because the observation is a number rather than a comparison: log-RT is roughly
    # normal, its noise is multiplicative, and the same ARD length-scales carry over since
    # the axes that move preference are the ones likely to move legibility. Correct and
    # never-paused trials only -- a paused clock measures the break and a wrong click
    # measures something else entirely.
    #
    # Preference chooses; legibility constrains. That is the program's constitution applied
    # one level deeper: the contrast floors keep a page readable in principle, and this
    # keeps it readable in fact.
    def rt_fit(responses, polarity, ls=None, noise_share=0.45):
        _X, _y, _mode, _px = [], [], [], []
        for _r in responses:
            if _r.get("mode") not in ("comprehension", "search"):
                continue
            if _r.get("polarity") != polarity or not _r.get("correct") or _r.get("paused"):
                continue
            _rt = float(_r.get("rt_ms") or 0.0)
            if _rt < 250.0 or _rt > 30000.0:
                continue
            _X.append(_coords(_r["theta_a"], polarity))
            _y.append(np.log(_rt))
            _mode.append(1.0 if _r["mode"] == "search" else 0.0)
            _px.append(float(_r.get("code_px") or 15.0))
        if len(_X) < 8:
            return None
        _X = np.array(_X)
        _y = np.array(_y)
        _mode = np.array(_mode)
        # A PER-ARM baseline, not one global mean. A find hunt highlights every match and
        # asks which is current; a comprehension probe gives a bare page and a name. The
        # second is systematically slower, and folding both into one mean would push that
        # constant difference into the theme surface as if some regions of theme space were
        # slow -- when what was slow was the task. Each arm's own mean is removed, and the
        # surface then models only what the THEME does to the clock. Needs both arms
        # present to be worth doing; with one arm this collapses to the global mean.
        _has_both = 0 < float(_mode.mean()) < 1
        _m_probe = float(_y[_mode == 0].mean()) if (_mode == 0).any() else float(_y.mean())
        _m_hunt = float(_y[_mode == 1].mean()) if (_mode == 1).any() else float(_y.mean())
        _base = (np.where(_mode > 0.5, _m_hunt, _m_probe) if _has_both else np.full(len(_y), _y.mean())).astype(float)
        _mu0 = _m_probe if _has_both else float(_y.mean())
        # And a per-SIZE offset, for exactly the reason there is a per-arm one. Glyph scale
        # moves reading time on its own, and the timed arms have not always run at one size:
        # they were 15 or 16 before the stimulus was pinned to the sizes he actually reads at
        # (14 in editors, 16 in notebook cells). Without this the step from one size regime
        # to the next lands on the theme surface as if some region of theme space had got
        # slower on the day the size changed. Only fitted where a size has enough trials to
        # mean anything; the rest fall back to the arm's own mean.
        _px = np.asarray(_px)
        for _arm in (0.0, 1.0):
            for _v in np.unique(_px):
                _sel = (_px == _v) & (_mode == _arm)
                # A cell needs enough trials for its own mean to beat the arm's; below that
                # the arm mean is the better estimate and the cell keeps it.
                if _sel.sum() >= 6:
                    _base[_sel] = float(_y[_sel].mean())
        # Signal and noise variance estimated from the data rather than borrowed from the
        # preference kernel: the preference GP's prior sd of 2 means a factor of seven in
        # log time, which produced a predicted span of 1.4 to 14 seconds -- nonsense on a
        # task he completes in two to four. Total variance is what log-RT actually shows,
        # and reaction time is famously noisy, so a large share of it is called noise
        # (0.45): the surface then claims a real difference only where the data insists.
        _resid = _y - _base
        _total = max(float(_resid.var()), 1e-4)
        _sf2 = max((1.0 - noise_share) * _total, 1e-4)
        _noise = max(noise_share * _total, 1e-4)
        _K = (_sf2 / _SF2) * _kmat(_X, _X, ls) + _noise * np.eye(len(_X))
        try:
            _Ki = np.linalg.inv(_K)
        except np.linalg.LinAlgError:
            return None
        return {
            "X": _X,
            "y": _base + _resid,
            "resid": _resid,
            "base": _base,
            "mu0": _mu0,
            "m_probe": _m_probe,
            "m_hunt": _m_hunt,
            "Ki": _Ki,
            "ls": ls,
            "n": len(_X),
            "sf2": _sf2,
            "noise": _noise,
        }

    def rt_at(rf, thetas, polarity):
        """Posterior mean and variance of log time-to-click at arbitrary themes."""
        _Xs = np.array([_coords(_t, polarity) for _t in thetas])
        _scale = rf["sf2"] / _SF2
        _ks = _scale * _kmat(_Xs, rf["X"], rf.get("ls"))
        _mu = rf["mu0"] + _ks @ (rf["Ki"] @ rf["resid"])
        _var = np.maximum(rf["sf2"] - np.einsum("ij,jk,ik->i", _ks, rf["Ki"], _ks), 1e-9)
        return _mu, _var

    def rt_penalty(rf, thetas, polarity, tol=0.10, confidence=0.9):
        """Which candidates are CREDIBLY slower to read than the fastest, and by how much.

        Returns (excluded mask, predicted seconds). A candidate is excluded only when the
        posterior says it is worse than the best by more than `tol` in log time with at
        least `confidence` probability -- so a thin or noisy RT log excludes nothing, which
        is the correct behaviour rather than a convenient one. The floor is relative: the
        question is never "is this page fast enough" in the abstract but "is it needlessly
        slower than a page he likes just as much".
        """
        _mu, _var = rt_at(rf, thetas, polarity)
        _best = float(np.min(_mu))
        _sd = np.sqrt(_var + float(np.min(_var)))
        # P(mu_i - best > tol) under a normal, without the covariance between i and the
        # argmin: conservative, which is the right direction for a constraint.
        _z = (_mu - _best - tol) / np.maximum(_sd, 1e-9)
        _p_worse = 0.5 * (1.0 + np.vectorize(math.erf)(_z / np.sqrt(2.0)))
        return _p_worse > confidence, np.exp(_mu)

    _BEST_MEMO = {}

    def best_set(fit, polarity, thetas, samples=2048, mass=0.5, seed=0, radius=0.9):
        """Which theme is best, or which SET is -- as a distribution over argmaxes.

        Three things have to be right for this to answer the question honestly.

        Sample the JOINT posterior, because candidates near each other share almost all
        their uncertainty and marginals would scatter the probability of being best across
        a cluster of effectively identical pages.

        Then GROUP before counting. A candidate set of eight hundred contains many pages
        that differ by less than he could ever see, and each sibling steals argmax mass
        from the others: measured on the real log, the leader held 1.6% while the report
        claimed a plateau -- a number that says nothing about whether one theme leads. Mass
        belongs to a perceptually distinct group, not to a coordinate.

        And read the verdict off CUMULATIVE mass, not an absolute cutoff. The credible set
        is the smallest group of groups holding `mass` of the argmax probability: one group
        over half of it is a winner; a handful sharing it is a real plateau; and when even
        the top group is thin, the honest answer is that the log cannot yet tell -- which
        is a state this reports rather than dressing up as a plateau.
        """
        # Memoized on the fit's identity, the polarity and the candidate set: the analysis
        # asks for the same verdict three times per polarity (the shelf, and the two
        # historical fits behind the progress readout), and each call is a Cholesky over
        # eight hundred candidates.
        _ck = (id(fit), polarity, len(thetas), samples, mass, seed, radius, float(np.sum(thetas[0])))
        if _ck in _BEST_MEMO:
            return _BEST_MEMO[_ck]
        _mu, _cov = posterior_joint(fit, thetas, polarity)
        try:
            _L = np.linalg.cholesky(_cov)
        except np.linalg.LinAlgError:
            _w, _V = np.linalg.eigh(_cov)
            _L = _V * np.sqrt(np.maximum(_w, 1e-12))
        _Z = np.random.default_rng(seed).standard_normal((len(thetas), samples))
        _F = _mu[:, None] + _L @ _Z
        _p = np.bincount(np.argmax(_F, axis=0), minlength=len(thetas)) / float(samples)

        # Group into perceptually distinct themes: greedy, best-first, in length-scale
        # scaled theta space, so a group is "themes his eyes and this model cannot
        # separate" rather than an arbitrary grid cell.
        _w_ax = 1.0 / (_LS0[:9] if fit.get("ls") is None else fit["ls"][:9])
        _P = np.array([np.asarray(_t) * _w_ax for _t in thetas])
        _order = np.argsort(-_p)
        _reps, _group_of = [], np.full(len(thetas), -1)
        for _i in _order:
            if _reps:
                _d = np.linalg.norm(_P[_reps] - _P[_i], axis=1)
                _j = int(np.argmin(_d))
                if _d[_j] <= radius:
                    _group_of[_i] = _j
                    continue
            _group_of[_i] = len(_reps)
            _reps.append(int(_i))
        _gp = np.zeros(len(_reps))
        for _i in range(len(thetas)):
            _gp[_group_of[_i]] += _p[_i]
        _gorder = np.argsort(-_gp)
        _keep, _acc = [], 0.0
        for _g in _gorder:
            _keep.append(int(_g))
            _acc += _gp[_g]
            if _acc >= mass:
                break
        _lead = float(_gp[_gorder[0]])
        _verdict = "single" if _lead > 0.5 else ("plateau" if _lead > 0.12 else "undecided")
        _res = {
            "p_best": _p,
            "order": _order,
            "groups": _reps,
            "group_p": _gp,
            "group_order": _gorder,
            "credible": [_reps[_g] for _g in _keep],
            "credible_p": [float(_gp[_g]) for _g in _keep],
            "lead": _lead,
            "mu": _mu,
            "verdict": _verdict,
        }
        if len(_BEST_MEMO) > 8:
            _BEST_MEMO.pop(next(iter(_BEST_MEMO)))
        _BEST_MEMO[_ck] = _res
        return _res

    def axis_consensus(bs, thetas):
        """Which axes his clicks have SETTLED, and which are still open.

        The plateau readout says how many themes are still in contention; it does not say
        what they disagree about. Measured on the four leading day themes: their grounds sit
        within 4 units of one cream, while their keyword hues run violet, dark green, dark
        red and blue. Reading "four distinct themes" against four pages that look alike at a
        glance is confusing; reading "the ground is decided, the accent hue is not" says
        what the remaining duels are for.

        Per axis, the posterior-weighted spread of theta under P(best), against the 0.289 of
        a uniform axis. Small means the mass has collected on one value; near 1 means the
        clicks have not distinguished anything along it yet."""
        _p = np.asarray(bs["p_best"], dtype=float)
        _T = np.asarray(thetas, dtype=float)
        if _p.sum() <= 0 or len(_T) == 0:
            return []
        _p = _p / _p.sum()
        _m = _p @ _T
        _sd = np.sqrt(np.maximum(_p @ (_T - _m) ** 2, 0.0))
        return [(_a, float(_sd[_a] / 0.2887), float(_m[_a])) for _a in range(_T.shape[1])]

    def progress_report(responses, polarity, thetas, back=25):
        """Is another sitting worth clicking? Compare the verdict now with the verdict as
        it stood `back` duels ago, on the SAME candidate set so the comparison is about
        evidence rather than about which themes happened to be bred.

        Two honest numbers come out of it: how the leader's share of the argmax mass moved,
        and how much the credible set shrank. The extrapolation to "duels still needed" is
        deliberately labelled naive -- it assumes the current rate continues, which it will
        not exactly, and it is there to answer "another hundred or another thousand" rather
        than to promise a finish line.
        """
        _duels = [_r for _r in responses if _r.get("mode") == "duel" and _r.get("choice") in (0, 1)]
        if len(_duels) < back + 12:
            return None
        _now = fitted(responses)
        _cut = len(_duels) - back
        _seen, _hist = 0, []
        for _r in responses:
            if _r.get("mode") == "duel" and _r.get("choice") in (0, 1):
                if _seen >= _cut:
                    continue
                _seen += 1
            _hist.append(_r)
        _then = fitted(_hist)
        if _then is None:
            return None
        _b_now = best_set(_now, polarity, thetas, seed=17)
        _b_then = best_set(_then, polarity, thetas, seed=17)
        _lead_gain = _b_now["lead"] - _b_then["lead"]
        _need = None
        if _lead_gain > 1e-3 and _b_now["lead"] < 0.5:
            _need = int(np.ceil((0.5 - _b_now["lead"]) / (_lead_gain / back)))
        return {
            "duels": len(_duels),
            "lead_now": _b_now["lead"],
            "lead_then": _b_then["lead"],
            "set_now": len(_b_now["credible"]),
            "set_then": len(_b_then["credible"]),
            "back": back,
            "duels_to_decide": _need,
        }

    def spread_out(thetas, idx, k, ls=None):
        """k maximally different members of a set -- greedy max-min in scaled theta space.

        A plateau is only useful if its members actually look different; picking the top-k
        by probability would return k variations of one page.
        """
        if not idx:
            return []
        _w = 1.0 / (_LS0[:9] if ls is None else ls[:9])
        _P = np.array([np.asarray(thetas[_i]) * _w for _i in idx])
        _pick = [0]
        while len(_pick) < min(k, len(idx)):
            _d = np.min(np.linalg.norm(_P[:, None, :] - _P[None, _pick, :], axis=-1), axis=1)
            _d[_pick] = -1.0
            _pick.append(int(np.argmax(_d)))
        return [idx[_i] for _i in _pick]

    # LOAD-BEARING placement: above the function that closes over it. marimo mangles a
    # cell-local underscore name only where it has already seen the assignment, so a memo
    # declared BELOW its user resolves fine under `marimo edit` and raises NameError under
    # `marimo run` the moment another cell calls in. Same trap as _CONTROL in the stimulus
    # cell. Underscore-prefixed names must be defined before the functions that use them.
    _SURF_MEMO = {}

    def factor_effect(responses, polarity, key, nperm=200, seed=7, min_n=24):
        """Does the preferred theme depend on some logged property of how it was SHOWN?

        The same question for any stimulus factor -- which surface, what pixel size, which
        kind of code -- because the machinery is identical and a second copy of a
        permutation test is a second place for it to be subtly wrong. `key` names the field
        in the log; its distinct values become the levels.

        See surface_effect below for what the test does and why the null is permutation."""
        _ds = [
            _r
            for _r in responses
            if _r.get("mode") == "duel"
            and _r.get(key) is not None
            and _r.get("polarity") == polarity
            and not _r.get("paused")
            and _r.get("choice") in (0, 1)
        ]
        # Recomputed every EIGHTH duel, not every click. 200 permutations x 5-fold Newton
        # fits costs about 2.8 s per factor, and with two factors over two polarities that
        # was 8.3 s of the analysis on every single answer -- which is not just slow, it is
        # long enough for two widget re-renders to overlap and leave a full-screen orphan
        # stage over the page (measured 2026-09-04, and it is what "the screen just blanked"
        # was). Truncating to a whole bucket keeps the memo key honest: within a bucket the
        # INPUT is identical, so the cached answer is the exact answer for the data named.
        _ds = _ds[: (len(_ds) // 8) * 8]
        _levels = sorted({_r[key] for _r in _ds}, key=str)
        if len(_ds) < min_n or len(_levels) < 2:
            return len(_ds), 0.0, 1.0, f"not enough {polarity} duels with a {key} to compare"
        _key = ("f", key, polarity, hash(tuple((_r["choice"], str(_r[key]), _r["theta_a"][0]) for _r in _ds)))
        if _key in _SURF_MEMO:
            return _SURF_MEMO[_key]
        _S = np.array([_levels.index(_r[key]) for _r in _ds])
        _K = len(_levels)
        _X = np.array(
            [
                # choice 0 = theme_a won (duels_from's convention; `swap` governs only which
                # SIDE a card appeared on, not which theme it was).
                (np.array(_r["theta_a"]) - np.array(_r["theta_b"])) * (1.0 if _r["choice"] == 0 else -1.0)
                for _r in _ds
            ]
        )

        def _cvll(_X, _S, _nax, _seed):
            """Held-out Bradley-Terry log-loss. _nax = 0 is one shared utility; _nax > 0 adds
            a sum-to-zero per-level tilt on the _nax leading axes, the cheapest form the
            interaction can take and so the one with the best chance of showing in the data
            there is."""
            _r = np.random.default_rng(_seed)
            _idx = _r.permutation(len(_X))
            _tot, _n = 0.0, 0
            for _f in range(5):
                _te = _idx[_f::5]
                _tr = np.setdiff1d(_idx, _te)
                if len(_tr) < 10:
                    continue

                def _feat(_Xa, _Sa):
                    if not _nax:
                        return _Xa
                    _cols = [_Xa]
                    for _j in range(_nax):
                        for _sv in range(_K - 1):
                            _cols.append(
                                np.where(_Sa == _sv, _Xa[:, _j], np.where(_Sa == _K - 1, -_Xa[:, _j], 0.0))[:, None]
                            )
                    return np.hstack(_cols)

                _F = _feat(_X[_tr], _S[_tr])
                _th = np.zeros(_F.shape[1])
                for _ in range(60):
                    _p = 1.0 / (1.0 + np.exp(-(_F @ _th)))
                    _g = _F.T @ (1.0 - _p) - _th
                    _H = (_F * (_p * (1 - _p))[:, None]).T @ _F + np.eye(len(_th))
                    _th = _th + np.linalg.solve(_H, _g)
                _z = _feat(_X[_te], _S[_te]) @ _th
                _tot += float(np.sum(-np.log1p(np.exp(-_z))))
                _n += len(_te)
            return _tot / max(_n, 1)

        def _gain(_S, _seeds):
            return float(
                np.mean([_cvll(_X, _S, 1, _s) for _s in range(_seeds)])
                - np.mean([_cvll(_X, _S, 0, _s) for _s in range(_seeds)])
            )

        _obs = _gain(_S, 6)
        _rng = np.random.default_rng(seed)
        _null = np.array([_gain(_rng.permutation(_S), 2) for _ in range(nperm)])
        _p = float((_null >= _obs).mean())
        if _p < 0.02:
            _v = f"{key} changes the optimum -- one theme is the wrong answer shape"
        elif _p < 0.10:
            _v = f"suggestive; keep {key} balanced and re-read"
        else:
            _v = f"no {key} effect this data can see"
        _out = (len(_ds), _obs, _p, _v)
        _SURF_MEMO[_key] = _out
        return _out

    def surface_effect(responses, polarity, nperm=200, seed=7):
        """Does the preferred theme depend on WHICH surface it is seen on?

        A theme is one theme, but it is seen in an editor, in the Claude Code panel, and in
        a notebook, and those differ in measure, in surrounding chrome and in whether prose
        sits next to the code. If the optimum moves between them, a single theme is the
        wrong shape of answer and the instrument should be searching three.

        Asked so a null answer means something. A per-surface tilt on the utility must EARN
        its extra parameters on HELD-OUT choices -- fit alone always improves. Then the
        earned amount is compared against its own permutation null: the same thetas, the
        same clicks, only the surface labels shuffled. That null is exact, and it is
        necessary, because at these counts adding two parameters clears a fixed threshold
        by chance in roughly one run in five (measured under a true null: 3 to 7 runs of 24).

        Returns (n, delta, p, verdict). Verdict is deliberately three-state for the same
        reason the main one is: "quiet" is not "absent" when the test has little power. At
        48 duels a tilt of 1 logit was detected 1 run in 12, so read a quiet answer as "not
        visible here" and collect more rather than as "settled".
        """
        return factor_effect(responses, polarity, "surface", nperm=nperm, seed=seed)

    def schedule_mode(n, n_duels):
        """Twenty-four-trial polarity blocks, each a run of sixteen duels, then four
        comprehension probes, then four find hunts — same-kind trials batched so one
        instruction serves a whole run and no click is spent re-reading. All-duel until the
        model has something to probe."""
        _pol = ("day", "night")[(n // 24) % 2]
        if n_duels < 6:
            return _pol, "duel"
        _slot = n % 24
        if _slot < 16:
            return _pol, "duel"
        if _slot < 20:
            return _pol, "comprehension"
        return _pol, "search"

    def duel_surface(n, n_duels):
        """Which of the three surfaces duel n is shown on.

        NOT `n % 3`. The schedule's block is 24 trials of which the first 16 are duels, and
        3 divides 24, so a modular rotation never de-phases: editor takes 6 of every 16
        duels and the other two 5 each, forever, and slot 0 is editor every single run. The
        log showed exactly that lock -- 6/5/5 by day, 12/10/10 by night. It is both a
        standing 20% over-sample of one surface and a hard confound between surface and
        position within the run, where first-of-run means the freshest eyes and the largest
        adaptation step from whatever was on screen before.

        Instead: consecutive groups of three duels each get a shuffled permutation of the
        three surfaces. Balance is exact every three duels rather than asymptotic, and the
        shuffle decorrelates surface from run position. Deterministic in the duel index, so
        replaying a log reconstructs the same assignment."""
        _d = n if n_duels < 6 else (n // 24) * 16 + min(n % 24, 16)
        _perm = list(SURFACES)
        random.Random(0xC0FFEE + _d // 3).shuffle(_perm)
        return _perm[_d % 3]

    def run_info(n, n_duels):
        """(polarity, mode, position within the run, run length) for trial n."""
        _pol, _mode = schedule_mode(n, n_duels)
        if n_duels < 6:
            return _pol, _mode, min(n_duels, 5), 6
        _slot = n % 24
        if _slot < 16:
            return _pol, _mode, _slot, 16
        if _slot < 20:
            return _pol, _mode, _slot - 16, 4
        return _pol, _mode, _slot - 20, 4

    # Deterministic given the log, so a memo keyed by trial number is a pure cache: three
    # cells ask for the same trial and pay for one fit.
    _TRIAL_MEMO = {}

    def trial_for(n, responses):
        """The nth trial, generated to maximize expected information about the utility.

        Duels: candidates are bred fresh (see candidates() -- elites, mutation, crossover,
        Sobol immigrants), one arm is a Thompson sample's argmax over them (explore where
        the optimum might be), the other the challenger with maximal expected information
        gain about the duel's outcome — plus a 7% share of uniform feasible pairs against
        model misspecification and, once a champion exists, a 5% share of
        champion-vs-worst anchors that double as
        engagement breathers and sanity checks. Comprehension probes ride the Thompson
        argmax; find hunts hold the champion's page and sweep the find axes uniformly."""
        if n in _TRIAL_MEMO:
            return _TRIAL_MEMO[n]
        _hist = responses[:n]
        _n_duels = sum(1 for _r in _hist if _r.get("mode") == "duel")
        _pol, _mode = schedule_mode(n, _n_duels)
        _rng = random.Random(n * 2654435761 % (2**31))
        _nprng = np.random.default_rng(n * 7919 + 13)
        _pool = POOL[_pol]
        _fit = fitted(_hist) if _n_duels >= 4 else None

        def _pick_pool(k):
            _idx = _rng.sample(range(len(_pool)), k)
            return [_pool[_i] for _i in _idx]

        _kind = "probe"
        if _mode == "duel":
            if _fit is None or _rng.random() < 0.07:
                (_ta, _tha), (_tb, _thb) = _pick_pool(2)
            else:
                _bred, _n_std = candidates(_fit, _pol, _nprng, n_trial=n)
                _cand = [_b[0] for _b in _bred]
                _cthemes = [_b[1] for _b in _bred]
                _mu, _var, _ks, _A = _posterior_over(_fit, _cand, _pol)
                if _rng.random() < 0.054:
                    _kind = "anchor"
                    _i1, _i2 = int(np.argmax(_mu)), int(np.argmin(_mu))
                else:
                    _kind = "eig"
                    _samp = _mu + np.sqrt(_var) * _nprng.standard_normal(len(_mu))
                    # Stratified Thompson: the explore/exploit split is DECLARED, not left
                    # to however many candidates each stratum happened to contribute.
                    # Measured: adding local children silently pulled the sampled argmax
                    # toward the incumbent's basin and cost reach (paired diff -0.14 on the
                    # two-mode test). Drawing the champion arm from the global stratum half
                    # the time restores it without giving up refinement.
                    _lo, _hi = (_n_std, len(_cand)) if (_rng.random() < 0.5 and _n_std < len(_cand)) else (0, _n_std)
                    _i1 = _lo + int(np.argmax(_samp[_lo:_hi]))
                    _cross = _kmat(
                        np.array([_coords(_t, _pol) for _t in _cand]),
                        np.array([_coords(_cand[_i1], _pol)]),
                        _fit.get("ls"),
                    )[:, 0] - np.einsum("ij,jk,k->i", _ks, _A, _ks[_i1])
                    _mud = _mu - _mu[_i1]
                    _s2 = np.maximum(_var + _var[_i1] - 2 * _cross, 1e-9)
                    _pbar = 1.0 / (1.0 + np.exp(-_mud / np.sqrt(1 + np.pi * _s2 / 8)))
                    _cond = (
                        _h2(1.0 / (1.0 + np.exp(-(_mud[:, None] + np.sqrt(_s2)[:, None] * _GH_X[None, :])))) @ _GH_W
                    )
                    _eig = _h2(_pbar) - _cond
                    _eig[_i1] = -1.0
                    _i2 = int(np.argmax(_eig))
                _ta, _tha = _cand[_i1], _cthemes[_i1]
                _tb, _thb = _cand[_i2], _cthemes[_i2]
            _snip = n * 7919 + 17
            _surface = duel_surface(n, len(responses))
            _trial = {
                "mode": "duel",
                # Both arms share surface and page: a duel varies the theme, nothing else.
                # A duel is judged full screen, so the sample must BE a page -- a fourteen
                # line block adrift in half a screen tells him nothing about how a screen
                # of this theme reads. Long enough to fill the half, and smaller type,
                # which is also what a full screen at this pixel density looks like in the
                # editor itself. Both stay logged as stimulus parameters.
                "snippet_width": DUEL_WIDTH,
                "snippet_lines": 28,
                "surface": _surface,
                "kind": _kind,
                "polarity": _pol,
                "theta_a": [round(float(_v), 6) for _v in _ta],
                "theta_b": [round(float(_v), 6) for _v in _tb],
                "theme_a": _tha,
                "theme_b": _thb,
                "snippet": _snip,
                # The size he reads THIS surface at (see READING_PX): the stimulus is then
                # the thing the answer is for, rather than a shrunken proxy for it.
                "code_px": READING_PX[_surface],
                "swap": _rng.random() < 0.5,
                "find_current": None,  # filled by the widget cell from the snippet
            }
        elif _mode == "comprehension":
            if _fit is not None and _rng.random() > 0.25:
                _bred = candidates(_fit, _pol, _nprng, n_trial=n)[0]
                _mu, _var, _ks, _A = _posterior_over(_fit, [_b[0] for _b in _bred], _pol)
                _samp = _mu + np.sqrt(_var) * _nprng.standard_normal(len(_mu))
                # Among the pages he might plausibly live in (the top of a Thompson draw),
                # probe the one whose reading time the model is least sure of. Probing a
                # page he would never choose measures legibility nobody will use; probing
                # the champion again measures what is already known.
                _top_idx = np.argsort(-_samp)[: max(8, len(_samp) // 20)]
                _rf_now = rt_fit(_hist, _pol, _fit.get("ls"))
                if _rf_now is not None:
                    _vv = rt_at(_rf_now, [_bred[int(_i)][0] for _i in _top_idx], _pol)[1]
                    _ta, _tha = _bred[int(_top_idx[int(np.argmax(_vv))])]
                else:
                    _ta, _tha = _bred[int(np.argmax(_samp))]
            else:
                _ta, _tha = _pool[_rng.randrange(len(_pool))]
            # Comprehension probes require a CALL-site target (Titus spotted this): a name
            # at its `def` sits at a line start, at a predictable indent, one or two to a
            # page, and is found far faster than the same name inside an expression. Mixing
            # the two kinds puts a step in the task's difficulty, and reaction time then
            # measures which kind was drawn rather than how the theme reads -- 12 of 60
            # probe pages were handing out the easy kind.
            _snip = n * 7919 + 17
            _trial = {
                "mode": "comprehension",
                "surface": "editor",
                "target_kind": "call",
                # A page, not a snippet: fourteen lines centred on an 8K screen is an island
                # spanning a quarter of the field, and a probe needs distractors to reject --
                # accuracy was saturated at 100% over twenty probes, and a 28-line page
                # offers ~97 identifiers to reject instead of ~28.
                "snippet_lines": 28,
                "kind": "task",
                "polarity": _pol,
                "theta_a": [round(float(_v), 6) for _v in _ta],
                "theme_a": _tha,
                "snippet": _snip,
                # A size he actually reads at. 15 was not one: his editors sit at 14 and his
                # notebook code cells at 16, so a legibility constraint measured at 15 was
                # constraining a size that never appears. These arms run on the editor
                # surface, so they take the editor's size. The per-size baseline in rt_fit
                # absorbs the step from the earlier 15/16 trials.
                "code_px": READING_PX["editor"],
            }
        else:  # search
            if _fit is not None:
                _bred = candidates(_fit, _pol, _nprng, n_trial=n)[0]
                _mu = _posterior_over(_fit, [_b[0] for _b in _bred], _pol)[0]
                _base = np.array(_bred[int(np.argmax(_mu))][0])
            else:
                _base = np.array(_pool[_rng.randrange(len(_pool))][0])
            # Sweep the find axes where the LEGIBILITY SURFACE is least certain, not
            # uniformly. Measured after 29 uniform hunts: the surface's posterior sd along
            # these axes (~0.38 log-units, a factor of 1.5 in time) dwarfed the effect it
            # was trying to see (a 10-15% swing), so uniform coverage was not identifying
            # them -- while ax8 meanwhile ranks second of nine for PREFERENCE, so the
            # question is worth answering. Uncertainty sampling is the standard active
            # choice for a GP regression and costs one posterior evaluation over a grid.
            # A quarter of hunts stay uniform, because an acquisition that only ever probes
            # its own uncertainty can leave a region unvisited that it is wrongly confident
            # about.
            _bt = _base.copy()
            _rf_now = rt_fit(_hist, _pol, _fit.get("ls") if _fit else None)
            if _rf_now is not None and _rng.random() > 0.25:
                _g = np.linspace(0.05, 0.95, 7)
                _cands_h = []
                for _v7 in _g:
                    for _v8 in _g:
                        _c = _base.copy()
                        _c[7], _c[8] = _v7, _v8
                        _cands_h.append(_c)
                _var_h = rt_at(_rf_now, _cands_h, _pol)[1]
                _bt = _cands_h[int(np.argmax(_var_h))]
            else:
                _bt[7], _bt[8] = _rng.random(), _rng.random()
            _tha = realize(_bt, _pol)
            if _tha is None:
                _idx = _rng.randrange(len(_pool))
                _bt, _tha = np.array(_pool[_idx][0]), _pool[_idx][1]
            _snip = n * 7919 + 17
            _trial = {
                "mode": "search",
                "surface": "editor",
                "snippet_lines": 28,
                "kind": "task",
                "polarity": _pol,
                "theta_a": [round(float(_v), 6) for _v in _bt],
                "theme_a": _tha,
                "snippet": _snip,
                "code_px": READING_PX["editor"],
            }
        _TRIAL_MEMO[n] = _trial
        return _trial

    return (
        axis_consensus,
        best_set,
        candidates,
        factor_effect,
        fitted,
        mu_at,
        posterior_joint,
        progress_report,
        rt_exponent,
        rt_at,
        rt_fit,
        rt_penalty,
        run_info,
        schedule_mode,
        spread_out,
        surface_effect,
        trial_for,
    )


@app.cell(hide_code=True)
def _(get_responses, mo):
    # The trial number doubles as a staleness indicator: if it disagrees with the stimulus
    # below, the surface lagged and the guard is dropping clicks. Instructions live in the
    # instrument's own bar, where the eye already is.
    _n = len(get_responses())
    mo.hstack([mo.md(f"**Trial {_n + 1}**")], justify="center")
    return


@app.cell(hide_code=True)
def _(SESSION_START_N, get_responses, mo, random, render_card, run_info, schedule_mode, snippet_for, trial_for):
    _n = len(get_responses())
    _t = trial_for(_n, get_responses())
    _nd = sum(1 for _r in get_responses() if _r.get("mode") == "duel")
    _pol, _mode, _pos, _len = run_info(_n, _nd)
    # Gate (begin button) at the first trial of a sitting and at every run boundary — the
    # moments where a new instruction must be read; inside a run the previous click is
    # the anchor and the clock starts at render.
    _nd_prev = _nd - (1 if _n > 0 and get_responses()[-1].get("mode") == "duel" else 0)
    _gate = _n == SESSION_START_N or (_n > 0 and schedule_mode(_n - 1, _nd_prev) != (_pol, _mode))
    _rng = random.Random(_n * 48271 % (2**31))
    _snip = snippet_for(_t["snippet"], _t.get("snippet_width"), _t.get("target_kind"), _t.get("snippet_lines"))
    _neutral = {"day": "#d8d2cf", "night": "#14161c"}[_t["polarity"]]
    # A single-card trial has ONE ground, so band, page and card are one continuous field:
    # a neutral band around the card would put a third colour between the sample and the
    # page that was just painted to match it. A duel keeps the neutral, since its band
    # surrounds two different grounds and must favour neither.
    _strip = _neutral if _t["mode"] == "duel" else _t["theme_a"]["ground"]
    # A duel's surround must not favour either arm, so it stays the polarity's neutral; a
    # single-card trial paints the page with the theme under test.
    _page_bg = _neutral if _t["mode"] == "duel" else _t["theme_a"]["ground"]
    # Whether the page runs marimo's dark theme is decided by the ground it took, not by
    # the polarity label: a light-ish night candidate should still get light-theme prose.
    _pg = _page_bg.lstrip("#")
    _page_dark = sum(int(_pg[_k : _k + 2], 16) for _k in (0, 2, 4)) < 384
    _ptxt = {"day": "#3a3532", "night": "#b8bcc6"}[_t["polarity"]]

    import anywidget
    import traitlets

    class _ThemeTrial(anywidget.AnyWidget):
        # The clock's baseline is the latest reveal; the click stamps the end; both ride the
        # synced traits into the record. First click only — later clicks and clicks on an
        # orphaned stale widget record nothing (the guard double-checks the trial number).
        # Gated trials (first of a sitting, first of a run) start behind an opaque cover
        # with the run's instruction and a begin button; the rest reveal at render, since
        # the click that produced them is the anchor. Pausing re-covers the stimulus (an
        # exposed one lets a decision form off the clock) and swallows clicks; revealing
        # again re-baselines. Tab-hide and 25 s of idling auto-pause. A trial paused after
        # its first reveal carries paused=true: its time is read as a near-tie, never as
        # evidence.
        _esm = """
        function render({ model, el }) {
          // Declared FIRST because several builders below read it. It used to sit beside
          // the stimulus row, one hundred lines after the instruction bar that reads it,
          // which put the bar inside its temporal dead zone: render threw a ReferenceError,
          // and because el.replaceChildren runs last the previous trial's markup stayed on
          // screen with a dead button -- indistinguishable from the instrument hanging.
          // Every render gets a generation number, and only the LATEST one is allowed to
          // move the frame. marimo tears down and rebuilds this widget per trial, and the
          // two lifecycles overlap: an older render's setFrame could fire after a newer one
          // had mounted, park the new wrap back inside the (zero-height) cell and leave the
          // full-screen stage on top of the page holding nothing -- a blank coloured screen
          // with a live trial hidden behind it (Titus, 2026-09-04: "the screen just blanked
          // after pressing the function"). Patching each ordering as it appears is
          // whack-a-mole; making a stale render a no-op removes the whole class.
          window.__themeTrialGen = (window.__themeTrialGen || 0) + 1;
          const myGen = window.__themeTrialGen;
          const current = () => myGen === window.__themeTrialGen;
          const isDuel = model.get("mode") === "duel";
          let t0 = -1;               // the clock's baseline: the latest reveal
          let revealed = false;      // every trial starts hidden
          let pausedNow = false;
          let pauses = 0;            // pauses AFTER the first reveal only
          el.style.cssText = "display:block;width:100%";
          // The WHOLE page takes the surround, not just the band. Titus judges these in
          // full screen, and in the vision instrument's own rule adaptation state is part
          // of the measurement: a dark candidate read inside a light page is measured in
          // the wrong adaptation state, and the surround dominates the field in full
          // screen. For a duel the surround is the polarity's fixed neutral -- the two
          // candidates have DIFFERENT grounds and painting the page with either would
          // advantage it -- while a single-card trial takes the candidate's own ground,
          // which is what a theme owning the screen actually looks like.
          // The page joins the polarity under test. Titus judges these in full screen,
          // where the surround is most of what the eye adapts to, and adaptation state is
          // part of the measurement by the vision instrument's own rule -- a dark
          // candidate read inside a light page is measured in the wrong state.
          //
          // Two earlier attempts were wrong in instructive ways. Painting body with
          // guessed container selectors left marimo's own content column white over the
          // dark field; walking up from this widget and clearing ancestor backgrounds
          // fixed the field but not the PROSE, which lives in sibling cells and stayed
          // dark-on-dark. The framework already has the switch: marimo keys its whole
          // theme off a `dark` class on the root element, so flipping that gets every
          // container, every piece of prose and every default ink coherently, and only
          // the exact ground still has to be painted on top.
          const surround = model.get("page_bg");
          document.documentElement.classList.toggle("dark", !!model.get("page_dark"));
          let pageStyle = document.getElementById("theme-trial-surround");
          if (!pageStyle) {
            pageStyle = document.createElement("style");
            pageStyle.id = "theme-trial-surround";
            document.head.appendChild(pageStyle);
          }
          // The exact ground goes through marimo's OWN custom property rather than over
          // the top of it: its page container is .bg-background reading
          // --background: light-dark(#fff, #181c1a), so setting that variable paints the
          // real field the real colour, and cards keep a step off it. Overriding a
          // framework with !important on guessed selectors is the smell that its hook has
          // not been found yet; this is the hook.
          const step = model.get("page_dark") ? 14 : -12;
          const hex = surround.replace("#", "");
          const card = "#" + [0, 2, 4].map((k) => {
            const v = parseInt(hex.slice(k, k + 2), 16) + step;
            return Math.max(0, Math.min(255, v)).toString(16).padStart(2, "0");
          }).join("");
          pageStyle.textContent =
            ":root, html { --background: " + surround + " !important;" +
            " --card: " + card + " !important; --popover: " + card + " !important; }" +
            "html, body { background: " + surround + " !important; }" +
            "body { transition: background 140ms linear; }";
          const wrap = document.createElement("div");
          // Full-bleed: marimo's prose column is ~700 px, too narrow for two code pages
          // at true editor sizes; the band breaks out to the viewport, capped at 1400 px.
          // Inline while gated (so the page around it stays reachable), viewport-owning
          // once a duel is revealed: he judges these in full screen and half a screen of
          // unrelated page would be half the adapting field. Pausing returns it inline.
          const inlineCss =
            `background:${model.get("strip_bg")};padding:18px;` +
            `border-radius:10px;display:flex;flex-direction:column;gap:14px;` +
            `position:relative;left:50%;transform:translateX(-50%);` +
            `width:min(96vw, 1400px);box-sizing:border-box;color:${model.get("ink")}`;
          // EVERY trial takes the screen once revealed, not only duels: he judges in full
          // screen, and a comparison that owns the field while a probe shares it with the
          // page would be measured in two different conditions.
          const fullCss =
            `background:${model.get("strip_bg")};padding:0;display:flex;` +
            // Above every piece of marimo's own chrome (its logo sat over the instruction
            // chip, its scrollbar showed at the edge) and, since the stage is parented to
            // <body>, outside marimo's stacking context rather than merely bidding against
            // it. The frame fills the stage; the stage owns the position.
            `flex-direction:column;gap:0;position:absolute;inset:0;` +
            `box-sizing:border-box;color:${model.get("ink")}`;
          wrap.style.cssText = inlineCss;
          const setFrame = (full) => {
            // Reparented to <body> rather than trusting a big z-index: marimo's logo sat
            // over the instruction bar even at z-index 2147483000, because z-index only
            // orders siblings within a stacking context and the widget's container is
            // inside one of marimo's. Moving the frame to the root context is the fix that
            // holds however the host rearranges its own furniture; it returns to its slot
            // when the frame goes inline, so nothing leaks.
            if (!current()) return;
            if (full) {
              // A PERSISTENT stage owned by the page, not by the widget. The widget is torn
              // down and rebuilt for every trial, and removing the frame each time left a
              // gap in which marimo's own loading placeholders showed through -- Titus:
              // "the loading graphic with the weird transparent things". The stage survives
              // the gap holding the previous trial, and the next render swaps its contents
              // in one step, so the screen is never anything but a trial. It also keeps its
              // ground colour, so even a slow kernel shows a uniform page rather than
              // chrome.
              // Exactly one stage, always. Two renders that overlap both find none and
              // both create one; the teardown path then removes only the first, and the
              // second stays -- full screen, painted with the page ground, and empty. That
              // orphan IS the blank screen. Collapsing duplicates on every pass makes the
              // race harmless instead of relying on it not happening.
              const stages = Array.from(document.querySelectorAll("#theme-trial-stage"));
              let host = stages[0];
              stages.slice(1).forEach((s) => s.remove());
              if (!host) {
                host = document.createElement("div");
                host.id = "theme-trial-stage";
                document.body.appendChild(host);
              }
              host.style.cssText =
                "position:fixed;inset:0;z-index:2147483000;background:" + model.get("strip_bg");
              if (wrap.parentElement !== host) {
                host.replaceChildren(wrap);
              }
            } else if (!full && wrap.parentElement !== el) {
              // ALL of them, for the same reason.
              document.querySelectorAll("#theme-trial-stage").forEach((s) => s.remove());
              el.appendChild(wrap);
            }
            wrap.style.cssText = full ? fullCss : inlineCss;
            top.style.padding = full ? "14px 20px 10px 20px" : "0";
            stage.style.flex = full ? "1 1 auto" : "0 0 auto";
            stage.style.minHeight = full ? "0" : "";
          };
          // The top bar carries only WHERE YOU ARE -- which run (chip, left), how far in
          // and how to pause (right). What to DO is not here: it belongs directly over the
          // code, centred, and lives in the stage below.
          //
          // It used to sit in this row, left-bound next to the chip, while the code was
          // centred several hundred pixels lower and to the right. On a full-screen 8k
          // display that is a real saccade from reading the instruction to starting the
          // search, and on a timed arm that travel is inside the measurement -- the same
          // reason the mouse-travel asymmetry had to go.
          const top = document.createElement("div");
          top.style.cssText = "display:flex;align-items:center;gap:16px;justify-content:space-between";
          const chip = document.createElement("div");
          chip.textContent = model.get("chip");
          chip.style.cssText = "font-family:'IBM Plex Serif',serif;font-size:12px;" +
            "letter-spacing:.14em;text-transform:uppercase;opacity:.75;white-space:nowrap;" +
            "border:1px solid currentColor;border-radius:999px;padding:3px 12px";
          const prompt = document.createElement("div");
          prompt.innerHTML = model.get("prompt_html");
          // Centred on the same axis as the card, sitting just above it, and large enough
          // to be read without leaning in. Its own line rather than a cell in a flex row,
          // so nothing to its left or right can push it off centre.
          prompt.style.cssText = "font-family:'IBM Plex Serif',serif;font-size:21px;" +
            "line-height:1.35;text-align:center;align-self:center;max-width:min(46em, 92vw);" +
            "margin:0 auto 26px auto;flex:0 0 auto";
          const keys = document.createElement("div");
          keys.textContent = isDuel ? "← →  or click" : "space pauses";
          keys.style.cssText = "font-family:'IBM Plex Serif',serif;font-size:12px;" +
            "opacity:.45;white-space:nowrap;letter-spacing:.04em";
          const progress = document.createElement("div");
          progress.textContent = model.get("progress");
          progress.style.cssText = "font-family:'IBM Plex Serif',serif;font-size:13px;" +
            "opacity:.6;white-space:nowrap;font-variant-numeric:tabular-nums";
          top.appendChild(chip);
          const btnStyle = "font-family:'IBM Plex Serif',serif;background:transparent;" +
            "color:inherit;border:1px solid currentColor;border-radius:8px;cursor:pointer";
          const pauseBtn = document.createElement("button");
          pauseBtn.textContent = "pause";
          pauseBtn.title = "hide the trial; the clock re-baselines when you reveal it again";
          pauseBtn.style.cssText = btnStyle + ";font-size:13px;opacity:.55;padding:2px 10px;" +
            "visibility:hidden";
          const meta = document.createElement("div");
          meta.style.cssText = "display:flex;align-items:center;gap:16px";
          meta.appendChild(keys);
          meta.appendChild(progress);
          meta.appendChild(pauseBtn);
          top.appendChild(meta);
          // The stimulus row keeps its box in the layout at all times; the cover is an
          // opaque overlay on exactly that box, so reveal/pause never move the page.
          const stage = document.createElement("div");
          // Grows in the full-screen frame so the halves reach the bottom of the viewport;
          // inline it keeps its content height.
          // Centres its children AS A GROUP, so the instruction and the code travel
          // together: the prompt sits directly above the card at a fixed gap rather than
          // pinned to the top of the screen with the card floating far below it.
          stage.style.cssText = "position:relative;display:flex;flex-direction:column;" +
            "justify-content:center";
          // A duel splits the VIEWPORT rather than laying two cards on a shared page.
          // Each half is full-bleed in its own ground, so each candidate is judged in its
          // own adaptation state -- the same reason the page takes the ground on a
          // single-card trial. A neutral surround would put every card on a mismatched
          // field, and painting the shared page with either candidate's ground would
          // advantage that one; splitting is the only arrangement that is both matched and
          // symmetric. No gap and no radius between the halves: a gutter would reintroduce
          // a third colour between the two things being compared.
          const row = document.createElement("div");
          // Edge to edge, because he judges in full screen and a centred pair gave back
          // adaptation area to a neutral surround for no gain. Each half owns its ground
          // with no gutter between them -- the point of splitting.
          row.style.cssText = isDuel
            ? "display:flex;gap:0;align-items:stretch;width:100%;visibility:hidden;flex:1 1 auto"
            : "display:flex;gap:16px;justify-content:center;align-items:center;" +
              "width:100%;visibility:hidden;flex:1 1 auto";
          const cover = document.createElement("div");
          cover.style.cssText = `position:absolute;inset:0;display:flex;flex-direction:column;` +
            `align-items:center;justify-content:center;gap:16px;border-radius:10px;` +
            `background:${model.get("strip_bg")};border:1px dashed currentColor;` +
            `font-family:'IBM Plex Serif',serif;font-size:15px;box-sizing:border-box`;
          const coverText = document.createElement("div");
          coverText.style.cssText = "opacity:.75;font-size:17px;max-width:38em;text-align:center;" +
            "line-height:1.5";
          const goBtn = document.createElement("button");
          goBtn.style.cssText = btnStyle + ";font-size:16px;padding:8px 26px;letter-spacing:.02em";
          cover.appendChild(coverText);
          cover.appendChild(goBtn);
          const setCover = (text, label) => {
            coverText.textContent = text;
            goBtn.textContent = label;
            cover.style.display = "flex";
            row.style.visibility = "hidden";
            pauseBtn.style.visibility = "hidden";
            setFrame(false);
          };
          let idleTimer = null;
          const armIdle = () => {
            if (idleTimer) clearTimeout(idleTimer);
            idleTimer = setTimeout(() => doPause("paused after 25 s without a click"), 25000);
          };
          const reveal = () => {
            setFrame(true);
            cover.style.display = "none";
            row.style.visibility = "visible";
            pauseBtn.style.visibility = "visible";
            revealed = true;
            pausedNow = false;
            t0 = performance.now();   // baseline re-initialized on EVERY reveal
            armIdle();
          };
          const doPause = (why) => {
            if (!revealed || pausedNow) return;
            pausedNow = true;
            pauses += 1;
            if (idleTimer) clearTimeout(idleTimer);
            setCover((why || "paused") + " \u2014 the stimulus is hidden; " +
              "the clock re-baselines when you resume", "resume");
          };
          goBtn.onclick = reveal;
          if (model.get("gate")) {
            setCover(model.get("gate_text"), "begin");
          } else {
            reveal();
          }
          pauseBtn.onclick = () => doPause("paused");
          const onVis = () => { if (document.hidden) doPause("paused while the tab was hidden"); };
          document.addEventListener("visibilitychange", onVis);
          let inputMethod = "mouse";
          const pick = (tid) => {
            if (!revealed || pausedNow) return;
            if (idleTimer) clearTimeout(idleTimer);
            model.set("clicks", model.get("clicks") + 1);
            model.set("choice", tid);
            model.set("pauses", pauses);
            model.set("t_render", t0);
            model.set("t_click", performance.now());
            model.set("input_method", inputMethod);
            model.save_changes();
          };
          // Arrow keys answer a duel, and that is a MEASUREMENT fix as much as a comfort:
          // clicking the left card on a 2560-or-wider screen is a different distance of
          // mouse travel than clicking the right one, so the reaction time the likelihood
          // reads as evidence of preference strength carried a systematic side component --
          // on top of the side bias already fitted. Two keys equidistant from the hand
          // remove it. Space reveals or pauses, so a whole sitting needs no mouse at all.
          // The method is recorded per response, so mouse and key trials stay separable.
          const onKey = (ev) => {
            if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
            const k = ev.key;
            if (isDuel && revealed && !pausedNow && (k === "ArrowLeft" || k === "ArrowRight")) {
              ev.preventDefault();
              inputMethod = "key";
              pick(k === "ArrowLeft" ? 0 : 1);
            } else if (k === " " || k === "Spacebar") {
              ev.preventDefault();
              if (!revealed || pausedNow) {
                reveal();
              } else {
                doPause("paused with the spacebar");
              }
            }
          };
          document.addEventListener("keydown", onKey);
          model.get("cards").forEach((c, i) => {
            const card = document.createElement("div");
            // The surface's own blocks (prose, code card, output) are SIBLINGS, so they go
            // inside one block-level child: a flex parent would otherwise lay them out as
            // a row and clip the code mid-line (measured -- it looked exactly as broken as
            // it sounds).
            const inner = document.createElement("div");
            // Capped so the pair straddles the centre rather than each block sprawling to
            // its own outer edge.
            inner.style.cssText = isDuel
              ? "width:100%;max-width:min(720px, 44vw);min-width:0"
              : "width:100%;max-width:100%;min-width:0";
            inner.innerHTML = c.html;
            card.appendChild(inner);
            card.style.cssText = isDuel
              // Grounds stay full-bleed (adaptation), but the CONTENT of each half hugs the
              // seam: left half right-aligned, right half left-aligned, so both code blocks
              // sit inside the middle half of the screen. On an 8K panel the outer edges
              // are viewed at an angle steep enough to skew the judgement, and code being
              // left-bound put the left candidate out there (Titus). Symmetric about the
              // centre, so neither candidate gains — the fairness property is preserved
              // while the optics stop biasing the answer.
              ? `background:${c.ground};padding:26px 34px;flex:1 1 0;min-width:0;` +
                `overflow:hidden;display:flex;align-items:center;` +
                `justify-content:${i === 0 ? "flex-end" : "flex-start"}`
              // A single-card trial centres on the screen, on the same ground the page
              // took, with no radius: card, band and page are one continuous field.
              : `background:${c.ground};padding:28px 32px;max-width:min(1100px, 92vw);` +
                `min-width:0;overflow:hidden;display:flex;align-items:center;` +
                `justify-content:center`;
            if (isDuel) {
              card.style.cursor = "pointer";
              card.onclick = () => pick(i);
            } else {
              card.onclick = (ev) => {
                const s = ev.target.closest("[data-tid]");
                if (s) pick(parseInt(s.dataset.tid));
              };
            }
            row.appendChild(card);
          });
          // The prompt sits ABOVE the covered area, never under it. The cover hides the
          // stimulus while he reads the instruction and again while paused; if it hid the
          // instruction too, reading the target name would fall inside the measured time
          // and every find would carry a word-recognition latency it did not before.
          const shade = document.createElement("div");
          // A duel fills the viewport edge to edge, so its stimulus box grows; a single
          // card is content-sized, so the box hugs it and the prompt above lands a fixed
          // distance from the code rather than a distance that depends on the window.
          shade.style.cssText = "position:relative;display:flex;min-height:0;width:100%;" +
            (isDuel ? "flex:1 1 auto" : "flex:0 0 auto;justify-content:center");
          shade.appendChild(row);
          shade.appendChild(cover);
          stage.appendChild(prompt);
          stage.appendChild(shade);
          wrap.appendChild(top);
          wrap.appendChild(stage);
          el.replaceChildren(wrap);
          return () => {
            if (idleTimer) clearTimeout(idleTimer);
            document.removeEventListener("visibilitychange", onVis);
            // Both of these matter per trial, not just at teardown: render() runs again for
            // every trial, so a listener left behind would fire once per past trial (one
            // keypress answering several) and a frame left parented to <body> would sit
            // over the next trial as an orphan.
            document.removeEventListener("keydown", onKey);
            // The stage is deliberately NOT removed here: leaving the last trial on screen
            // is what stops the loading placeholders from flashing between trials. The next
            // render replaces its contents; the inline path removes it when a trial is
            // gated and the page should be reachable again.
          };
        }
        export default { render };
        """
        mode = traitlets.Unicode("duel").tag(sync=True)
        strip_bg = traitlets.Unicode("#888888").tag(sync=True)
        page_bg = traitlets.Unicode("#888888").tag(sync=True)
        page_dark = traitlets.Bool(False).tag(sync=True)
        ink = traitlets.Unicode("#808080").tag(sync=True)
        prompt_html = traitlets.Unicode("").tag(sync=True)
        chip = traitlets.Unicode("").tag(sync=True)
        progress = traitlets.Unicode("").tag(sync=True)
        gate = traitlets.Bool(False).tag(sync=True)
        gate_text = traitlets.Unicode("").tag(sync=True)
        cards = traitlets.List([]).tag(sync=True)
        choice = traitlets.Int(-1).tag(sync=True)
        clicks = traitlets.Int(0).tag(sync=True)
        input_method = traitlets.Unicode("mouse").tag(sync=True)
        pauses = traitlets.Int(0).tag(sync=True)
        t_render = traitlets.Float(-1.0).tag(sync=True)
        t_click = traitlets.Float(-1.0).tag(sync=True)

    # The thing he is hunting for, set as a token rather than as running text: same
    # typeface it wears in the code, on a neutral tint so it reads as a quoted string
    # rather than as part of the sentence. Neutral on purpose -- a tinted chip in any
    # theme colour would cue the search, and the find-highlight hue is one of the axes
    # under test.
    _mono = (
        "font-family:'IosevkaLigated Nerd Font Mono',monospace;font-size:20px;"
        "background:color-mix(in srgb, currentColor 9%, transparent);"
        "padding:2px 9px;border-radius:6px;letter-spacing:.01em"
    )
    _chip = {"duel": "duel", "comprehension": "spot", "search": "find"}[_t["mode"]] + f" · {_pol} page"
    _progress = f"{_pos + 1} of {_len}"
    _gate_text = {
        "duel": (
            f"A run of {_len} duels on the {_pol} page: two pages render the same code — "
            "click the one you would rather read. Trust the first pull; a slow choice reads as a tie."
        ),
        "comprehension": (
            f"A run of {_len} probes on the {_pol} page: the line above names a function — "
            "click that name in the code as fast as you can find it."
        ),
        "search": (
            f"A run of {_len} find hunts on the {_pol} page: several matches are highlighted — "
            "click the current one, the strongest highlight, as fast as you can find it."
        ),
    }[_t["mode"]]
    if _t["mode"] == "duel":
        _cur = _rng.choice(_snip["ident_ids"]) if _snip["ident_ids"] else None
        _surface = _t.get("surface", "editor")
        _cards = [
            {
                "html": render_card(_t["theme_a"], _snip, _t["code_px"], find_current=_cur, surface=_surface),
                "ground": _t["theme_a"]["ground"],
            },
            {
                "html": render_card(_t["theme_b"], _snip, _t["code_px"], find_current=_cur, surface=_surface),
                "ground": _t["theme_b"]["ground"],
            },
        ]
        if _t["swap"]:
            _cards = _cards[::-1]
        _prompt = 'Which page would you rather read? <span style="opacity:.55">Click it.</span>'
    elif _t["mode"] == "comprehension":
        _target = _rng.choice(_snip["fn_ids"])
        _name = _snip["spans"][_target]["text"]
        _surface = _t.get("surface", "editor")
        _cards = [
            {
                "html": render_card(_t["theme_a"], _snip, _t["code_px"], task=True, prose=False),
                "ground": _t["theme_a"]["ground"],
            }
        ]
        _prompt = f'Click <code style="{_mono}">{_name}</code>'

    else:
        _cur = _rng.choice(_snip["ident_ids"])
        _surface = _t.get("surface", "editor")
        _cards = [
            {
                "html": render_card(_t["theme_a"], _snip, _t["code_px"], find_current=_cur, task=True, prose=False),
                "ground": _t["theme_a"]["ground"],
            }
        ]
        _prompt = 'Click the <b>current</b> match <span style="opacity:.55">— the strongest highlight.</span>'

    trial_widget = mo.ui.anywidget(
        _ThemeTrial(
            mode=_t["mode"],
            strip_bg=_strip,
            page_bg=_page_bg,
            page_dark=bool(_page_dark),
            ink=_ptxt,
            prompt_html=_prompt,
            chip=_chip,
            progress=_progress,
            gate=bool(_gate),
            gate_text=_gate_text,
            cards=_cards,
        )
    )
    trial_widget
    return (trial_widget,)


@app.cell(hide_code=True)
def _(LOG, datetime, get_responses, json, random, set_responses, snippet_for, timezone, trial_for, trial_widget):
    # Recording watches the widget's synced traits; the guard converts a stale surface's
    # click into a dropped click instead of a mis-record, and the trial is recomputed from
    # the log at event time — never read from a rendering's closure.
    _n = len(get_responses())

    def _record(v, n=_n):
        if n != len(get_responses()):
            return
        _t = trial_for(n, get_responses())
        _rng = random.Random(n * 48271 % (2**31))
        _snip = snippet_for(_t["snippet"], _t.get("snippet_width"), _t.get("target_kind"), _t.get("snippet_lines"))
        _entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n": n,
            "mode": _t["mode"],
            "kind": _t["kind"],
            "polarity": _t["polarity"],
            "snippet": _snip["id"],
            "snippet_hash": _snip.get("hash"),
            "snippet_kind": _snip.get("kind"),
            "snippet_fresh": bool(_snip.get("fresh", True)),
            "target_kind": _snip.get("target_kind"),
            "surface": _t.get("surface", "editor"),
            "input_method": _v.get("input_method", "mouse"),
            # Recomputed rather than shared: this is a different cell, and the widget
            # cell's underscore names are local to it.
            "page_bg": (
                {"day": "#d8d2cf", "night": "#14161c"}[_t["polarity"]]
                if _t["mode"] == "duel"
                else _t["theme_a"]["ground"]
            ),
            "code_px": _t["code_px"],
            "theta_a": _t["theta_a"],
            "theme_a": _t["theme_a"],
            # rt_ms runs from the LAST reveal (render, or resume after a pause); a trial
            # that was ever paused is flagged so its time is read as a near-tie downstream.
            "rt_ms": round(v["t_click"] - v["t_render"], 1),
            "t_render": round(v["t_render"], 1),
            "t_click": round(v["t_click"], 1),
            "paused": v.get("pauses", 0) > 0,
        }
        if _t["mode"] == "duel":
            _cur = _rng.choice(_snip["ident_ids"]) if _snip["ident_ids"] else None
            _shown = v["choice"]  # 0 = left card
            _choice = (1 - _shown) if _t["swap"] else _shown  # 0 = theme_a
            _entry.update(
                theta_b=_t["theta_b"],
                theme_b=_t["theme_b"],
                swap=_t["swap"],
                find_current=_cur,
                choice=_choice,
            )
        elif _t["mode"] == "comprehension":
            _target = _rng.choice(_snip["fn_ids"])
            _name = _snip["spans"][_target]["text"]
            _accept = [_i for _i, _s in enumerate(_snip["spans"]) if _s["role"] == "function" and _s["text"] == _name]
            _entry.update(target=_target, target_text=_name, clicked=v["choice"], correct=v["choice"] in _accept)
        else:
            _cur = _rng.choice(_snip["ident_ids"])
            _entry.update(
                target=_cur,
                clicked=v["choice"],
                correct=v["choice"] == _cur,
                salience=_t["theme_a"]["salience"],
                find_sal_theta=_t["theta_a"][8],
            )
        # Append-only, one record per line: concurrent sessions interleave, never overwrite.
        with LOG.open("a") as _f:
            _f.write(json.dumps(_entry) + "\n")
        set_responses([*get_responses(), _entry])

    _v = trial_widget.value
    if _v.get("clicks") == 1 and _v.get("choice", -1) >= 0 and _v.get("t_click", -1) > 0:
        _record(_v)
    return


@app.cell(hide_code=True)
def _(
    AXES,
    CHAMPION,
    DE_MIN,
    axis_consensus,
    factor_effect,
    POOL,
    THRESH_DETAIL,
    VISION_N,
    best_set,
    candidates,
    fitted,
    get_responses,
    json,
    mo,
    mu_at,
    np,
    pd,
    progress_report,
    render_card,
    rt_at,
    rt_exponent,
    rt_fit,
    rt_penalty,
    snippet_for,
    spread_out,
    surface_effect,
):
    # A stable page for the champion preview: the same generated code every time, so
    # what changes between renders is the theme and nothing else.
    _preview_snip = snippet_for(0)
    _log = get_responses()
    if not _log:
        _out = mo.md("*No responses yet — the analysis fills in as you answer.*")
    else:
        _frame = pd.DataFrame(_log)
        _n_duel = int((_frame["mode"] == "duel").sum())
        _fit = fitted(_log) if _n_duel >= 5 else None
        _blocks = [
            mo.hstack(
                [
                    mo.stat(f"{len(_frame):,}", label="responses", bordered=True),
                    mo.stat(str(_n_duel), label="duels", bordered=True),
                    mo.stat(
                        str(int((_frame["mode"] == "comprehension").sum())),
                        label="comprehension probes",
                        bordered=True,
                    ),
                    mo.stat(str(int((_frame["mode"] == "search").sum())), label="find hunts", bordered=True),
                    mo.stat(
                        f"{DE_MIN['day']:.1f} / {DE_MIN['night']:.1f}",
                        label=f"ΔE floors (day/night), {VISION_N} vision trials",
                        bordered=True,
                    ),
                ],
                justify="start",
                gap=1,
            )
        ]
        if _fit is not None:
            for _pol in ("day", "night"):
                # The verdict is computed over BRED candidates, not the frozen pool: the
                # answer should be the best theme the search can reach, not the best of 512
                # points fixed before the first click.
                _bred = candidates(_fit, _pol, np.random.default_rng(4242), n_trial=0)[0]
                _thetas = [_b[0] for _b in _bred]
                _themes = [_b[1] for _b in _bred]
                # The timed arms bind here. A page he likes but reads slowly is not a
                # winner, so candidates the legibility surface says are credibly slower
                # than the fastest are dropped BEFORE the preference verdict is computed --
                # constraint first, preference second, the same order the contrast floors
                # use. A thin or noisy RT log drops nothing, by construction.
                _rf = rt_fit(_log, _pol, _fit.get("ls"))
                _rt_note = ""
                if _rf is not None:
                    _excl, _secs = rt_penalty(_rf, _thetas, _pol)
                    _rvar = rt_at(_rf, _thetas, _pol)[1]
                    _keep_idx = [_i for _i in range(len(_thetas)) if not _excl[_i]]
                    if len(_keep_idx) >= 32:
                        # Whether his taste is costing him reading speed -- as a DIFFERENCE
                        # with an interval, never as two point estimates side by side. The
                        # old wording put "5.6 s" next to "3.4 s for the quickest page the
                        # model knows" and let the reader draw a conclusion the data does not
                        # support twice over: the posterior sd on either page is around 0.25
                        # to 0.38 in log time, so the difference of two is wider still; and
                        # the minimum over several hundred noisy predictions is an extreme
                        # order statistic, biased low, so the fast end of that comparison was
                        # partly a selection effect. Measured on the current log the gap is
                        # +0.03 [-0.80, +0.85] by day: no measurable cost to liking what he
                        # likes, which is the reassuring answer and also the honest one.
                        _lead_i = _keep_idx[0]
                        _fast_i = int(np.argmin(_secs))
                        _dmu = float(np.log(_secs[_lead_i]) - np.log(_secs[_fast_i]))
                        _dsd = float(np.sqrt(_rvar[_lead_i] + _rvar[_fast_i]))
                        _rt_note = (
                            f" {int(_excl.sum())} of {len(_thetas)} candidates were dropped first "
                            f"as credibly slower to read than the fastest ({_rf['n']} timed trials). "
                            f"The leader reads in about {_secs[_lead_i] / 1000:.1f} s"
                        )
                        if _dmu - 1.96 * _dsd > 0:
                            _rt_note += (
                                f", and that is credibly slower than the quickest page the model "
                                f"knows by {_dmu:+.2f} in log time (95% interval "
                                f"[{_dmu - 1.96 * _dsd:+.2f}, {_dmu + 1.96 * _dsd:+.2f}]) — taste "
                                f"and speed are pulling apart here, and the shelf is worth "
                                f"re-reading with that in mind."
                            )
                        else:
                            _rt_note += (
                                f", which is not measurably slower than the quickest page the model "
                                f"knows: the difference is {_dmu:+.2f} in log time with a 95% "
                                f"interval of [{_dmu - 1.96 * _dsd:+.2f}, {_dmu + 1.96 * _dsd:+.2f}], "
                                f"so liking these pages is costing you no measurable reading speed."
                            )
                        _thetas = [_thetas[_i] for _i in _keep_idx]
                        _themes = [_themes[_i] for _i in _keep_idx]
                _mu = mu_at(_fit, _thetas, _pol)
                _ci = int(np.argmax(_mu))
                _champ_theta, _champ = _thetas[_ci], _themes[_ci]
                _beats = float(np.mean(1.0 / (1.0 + np.exp(-(_mu[_ci] - _mu)))))
                # Is there ONE best theme or a plateau of equals? P(best) over the joint
                # posterior answers it as a distribution rather than a ranking: mass
                # concentrated on one page means a winner, mass spread means any member of
                # the credible set is a defensible choice -- and the ones shown are picked
                # for spread, since a plateau is only useful if its members look different.
                _bs = best_set(_fit, _pol, _thetas, seed=17)
                _cred = _bs["credible"]
                _reps = spread_out(_thetas, _cred, 4, _fit.get("ls"))
                # Each card carries its GROUP's probability, which is the number the verdict
                # above quotes. Its own p_best is the mass of one point in a continuum and is
                # always far smaller -- printing that next to "the leader holds 24%" made the
                # leader's card read 2%. Ordered by it too, so the leader is the first card
                # rather than wherever the spread-out walk happened to place it.
                _gp_of = dict(zip(_cred, _bs["credible_p"], strict=True))
                _reps = sorted(_reps, key=lambda _i: -_gp_of.get(_i, 0.0))
                _lead_pct = 100 * _bs["lead"]
                if _bs["verdict"] == "single":
                    _verdict = (
                        f"**one theme leads** — it holds {_lead_pct:.0f}% of the probability of "
                        f"being the best theme, so this is the one to apply"
                    )
                elif _bs["verdict"] == "plateau":
                    _verdict = (
                        f"**a plateau of {len(_cred)} distinct themes** — the leader holds "
                        f"{_lead_pct:.0f}%, and these together hold half the probability of being "
                        f"best. They are equally good by measurement, not merely acceptable: every "
                        f"one has already cleared the legibility floors, so pick by eye"
                    )
                else:
                    _verdict = (
                        f"**not yet decided** — the strongest theme holds only {_lead_pct:.0f}% of "
                        f"the probability of being best, which is what a thin log looks like rather "
                        f"than a real plateau. {len(_cred)} themes share half the mass; more duels "
                        f"on this polarity will separate them"
                    )
                _prog = progress_report(_log, _pol, _thetas)
                _prog_note = ""
                if _prog is not None:
                    _moved = 100 * (_prog["lead_now"] - _prog["lead_then"])
                    _shrunk = _prog["set_then"] - _prog["set_now"]
                    _head = (
                        f" Over the last {_prog['back']} duels the leader's share moved "
                        f"{_moved:+.0f} points and the credible set changed by {-_shrunk:+d} themes"
                    )
                    if _prog["duels_to_decide"] is not None:
                        # A leader gaining ground: extrapolate, and say plainly that it is
                        # a straight line through two points.
                        _prog_note = (
                            f"{_head}; at that rate roughly {_prog['duels_to_decide']} more duels "
                            f"would give one theme a majority — a naive straight-line estimate, "
                            f"worth reading as 'another sitting' or 'another ten'."
                        )
                    elif _shrunk > 0:
                        # The distinction that matters and that a two-case reading gets
                        # wrong: mass can move AWAY from the leader while the set shrinks.
                        # That is not stalling, it is the model resolving a real plateau --
                        # evidence still arriving, just not concentrating on one page.
                        _prog_note = (
                            f"{_head} — so evidence is still arriving and the field is narrowing, "
                            f"but the mass is spreading across the survivors rather than "
                            f"concentrating: what a genuine plateau looks like as it comes into "
                            f"focus. More duels sharpen WHICH themes are on the shelf, not which "
                            f"one wins."
                        )
                    else:
                        _prog_note = (
                            f"{_head} — flat on both counts, so more duels on this polarity are "
                            f"buying little and the shelf above is the answer rather than a "
                            f"waypoint."
                        )
                # Whether one theme is even the right SHAPE of answer. Cheap to ask and
                # expensive to get wrong: if the optimum moves between the editor, the chat
                # panel and a notebook, then converging one theme onto all three averages
                # over a real difference instead of resolving it.
                _sn, _sd_gain, _sp, _sv = surface_effect(_log, _pol)
                # Duels used to run at 12-13px, which is a size he never reads code at; they
                # now run at the size he reads each surface at (14 in editors, 16 in notebook
                # cells). That pools two stimulus regimes in one log, so the same test asks
                # whether they can be pooled: a code_px interaction means the older small-type
                # duels are answering a different question and should be discounted.
                _zn, _zd, _zp, _zv = factor_effect(_log, _pol, "code_px")
                if _sn < 24:
                    _surf_note = (
                        f" Surface (editor / panel / notebook) is logged but only {_sn} {_pol} duels "
                        f"carry a label so far, too few to ask whether the optimum moves between them."
                    )
                elif _sp < 0.02:
                    _surf_note = (
                        f" **Surface matters**: a per-surface tilt earns {_sd_gain:+.3f} nats/duel on "
                        f"held-out choices against its own permutation null (p = {_sp:.3f}, {_sn} duels). "
                        f"One theme is the wrong shape of answer here — the editor, the chat panel and "
                        f"the notebook want different pages."
                    )
                elif _sp < 0.10:
                    # Two stimulus factors are tested here (surface, type size) across two
                    # polarities, so about one reading in every two or three sittings lands
                    # this side of 0.10 with nothing real behind it. Saying so is the
                    # difference between a finding and a coincidence with a p-value.
                    _surf_note = (
                        f" Surface may matter: a per-surface tilt earns {_sd_gain:+.3f} nats/duel "
                        f"held out (p = {_sp:.3f} over {_sn} duels) — suggestive, not established, "
                        f"and one of four such factor readings, of which roughly one lands here by "
                        f"chance anyway. Duels are surface-balanced in groups of three, so this "
                        f"sharpens on its own; worth re-reading at about twice this many duels."
                    )
                else:
                    _surf_note = (
                        f" No surface effect this data can see (p = {_sp:.2f} over {_sn} duels), so "
                        f"one theme across editor, panel and notebook remains the right shape of answer."
                    )
                if _zn >= 24 and _zp < 0.10:
                    _surf_note += (
                        f" Type size also tilts the answer (p = {_zp:.3f} over {_zn} duels): the "
                        f"early duels judged at 12-13px are measuring a different question from "
                        f"those at his real 14 and 16, and should carry less weight."
                    )
                elif _zn >= 24:
                    _surf_note += (
                        f" Duels judged at different type sizes agree (p = {_zp:.2f}), so the early "
                        f"12-13px rounds pool safely with the ones at his real reading sizes."
                    )
                # What the plateau actually disagrees about. Without this, four pages that
                # share a ground and differ only in accent hue read as "four identical
                # themes" and the word "distinct" looks wrong -- when in fact one question
                # is answered and another is wide open.
                _cons = axis_consensus(_bs, _thetas)
                _settled = sorted([_c for _c in _cons if _c[1] < 0.55], key=lambda _c: _c[1])
                _open = sorted([_c for _c in _cons if _c[1] > 0.85], key=lambda _c: -_c[1])

                # One sentence per case: joining fragments produced "Your clicks have still
                # open on accent hue rotation" the first time night had nothing settled.
                def _names(_g):
                    return ", ".join(f"**{AXES[_a]}**" for _a, _r, _m in _g[:3])

                if _settled and _open:
                    _axis_note = (
                        f" Your clicks have settled {_names(_settled)}, and have not yet separated "
                        f"{_names(_open)} — so the themes on this shelf mostly differ in the second "
                        f"group, and that is what further duels decide."
                    )
                elif _settled:
                    _axis_note = (
                        f" Your clicks have settled {_names(_settled)}, and no axis is still wide "
                        f"open: what remains is fine separation rather than an open question."
                    )
                elif _open:
                    _verb = "are" if len(_open) > 1 else "is"
                    _axis_note = (
                        f" No axis has settled yet, and {_names(_open)} {_verb} still wide open — "
                        f"the shelf differs there, and that is what further duels decide."
                    )
                else:
                    _axis_note = ""
                _blocks.append(
                    mo.md(
                        f"### The {_pol} verdict\n\n{_verdict}.{_rt_note}{_prog_note}{_surf_note}{_axis_note}"
                        f" Shown below: the leader, then the "
                        f"most *different* members of the set holding half the probability mass — "
                        f"near-identical themes are grouped first, so what you see are choices "
                        f"rather than variations of one."
                    )
                )
                # Full-bleed, and each card wide enough for the page it holds. The prose
                # column is 610px because that is a good measure for READING; four theme
                # cards inside it are 306px each, which clips the page mid-token and leaves
                # him comparing palettes by the left two-thirds of every line. A page needs
                # about 520px to render whole at 12px, so the row steps out of the measure
                # and the cards wrap instead of shrinking -- on a full-screen 8k display all
                # four sit side by side at full page width, and on a narrow one they become
                # two rows of whole pages rather than one row of cropped ones.
                _cards = "".join(
                    f'<figure style="margin:0;flex:0 0 520px;max-width:100%">'
                    f'<figcaption style="font:600 13px/1.5 system-ui,sans-serif;margin:0 0 6px 2px">'
                    f"{100 * _gp_of.get(_i, 0.0):.0f}%"
                    f"{' · leads' if _i == _reps[0] else ''}"
                    f'<span style="font-weight:400;opacity:.65"> · utility {_mu[_i]:.2f}</span>'
                    f"</figcaption>"
                    f'<div style="background:{_themes[_i]["ground"]};border-radius:8px;'
                    f'padding:14px;overflow:hidden">'
                    + render_card(_themes[_i], _preview_snip, 12, prose=False)
                    + "</div></figure>"
                    for _i in _reps
                )
                _blocks.append(
                    mo.Html(
                        '<div style="width:94vw;margin-left:calc(-47vw + 50%);display:flex;'
                        'flex-wrap:wrap;gap:22px;justify-content:center">' + _cards + "</div>"
                    )
                )
                _sweep = []
                for _ax in range(9):
                    _lo_t = np.array(_champ_theta, dtype=float)
                    _hi_t = _lo_t.copy()
                    _lo_t[_ax], _hi_t[_ax] = 0.15, 0.85
                    _mm = mu_at(_fit, [_lo_t, _hi_t], _pol)
                    _sweep.append(
                        {
                            "axis": AXES[_ax],
                            "low (0.15)": round(float(_mm[0] - _mu[_ci]), 2),
                            "high (0.85)": round(float(_mm[1] - _mu[_ci]), 2),
                        }
                    )
                # Publish rather than print. The applier reads this file, so the palette
                # crosses from instrument to editor without a human copying hex codes -- the
                # step where a digit gets dropped and nobody notices for a week. Written on
                # every analysis pass, so it always reflects the current log; the applier is
                # what decides when his editor changes, and it is never this notebook.
                _pub = {}
                if CHAMPION.exists():
                    try:
                        _pub = json.loads(CHAMPION.read_text())
                    except Exception:
                        _pub = {}
                _pub[_pol] = {
                    "ground": _champ["ground"],
                    "find_fill": _champ["find_fill"],
                    "keyword": _champ["keyword"],
                    "function": _champ["function"],
                    "string": _champ["string"],
                    "comment": _champ["comment"],
                    "ink": _champ["ink"],
                    "punct": _champ["punct"],
                    "p_best": round(float(_bs["lead"]), 4),
                    "verdict": _bs["verdict"],
                    "n_duels": int(_frame[(_frame["mode"] == "duel") & (_frame["polarity"] == _pol)].shape[0]),
                }
                CHAMPION.write_text(json.dumps(_pub, indent=2, sort_keys=True) + "\n")
                _blocks += [
                    mo.md(
                        f"**Current best {_pol} page** — beats a random feasible theme with "
                        f"p ≈ {_beats:.2f}; utility marginals below are the posterior-mean change "
                        f"from the champion when one axis is pushed to its walls (negative = the "
                        f"champion's setting is better):"
                    ),
                    mo.Html(
                        f'<div style="background:{_champ["ground"]};border-radius:10px;padding:20px;max-width:620px">'
                        + render_card(
                            _champ,
                            _preview_snip,
                            16,
                            find_current=(_preview_snip["ident_ids"] or [None])[0],
                        )
                        + "</div>"
                    ),
                    mo.ui.table(pd.DataFrame(_sweep), selection=None),
                    mo.md(
                        f"Champion published to `{CHAMPION.name}` — apply it with "
                        f"`dotfiles/home/editors/vscode/apply-measured-theme.py`, which rewrites the "
                        f"marked block in settings.jsonc rather than asking you to paste. The same "
                        f"palette, for reading:"
                    ),
                    mo.md(
                        "```jsonc\n"
                        + "{\n"
                        + f"  // {_pol} · ground {_champ['ground']}\n"
                        + f'  "editor.background": "{_champ["ground"]}",\n'
                        + f'  "editor.findMatchBackground": "{_champ["find_fill"]}d9",\n'
                        + f'  "editor.findMatchHighlightBackground": "{_champ["find_fill"]}73",\n'
                        + '  "textMateRules": {\n'
                        + f'    "keyword": "{_champ["keyword"]}", "function": "{_champ["function"]}",\n'
                        + f'    "string|number": "{_champ["string"]}", "comment (italic)": "{_champ["comment"]}",\n'
                        + f'    "variables/ink": "{_champ["ink"]}", "punctuation": "{_champ["punct"]}"\n'
                        + "  }\n"
                        + "}\n```"
                    ),
                ]
        _tasks = _frame[_frame["mode"] == "comprehension"]
        if len(_tasks) >= 6:
            # Correct AND never-paused: a paused trial's clock measures the break, not the
            # eyes. Rows predating the pause affordance lack the field: they count unpaused.
            _np1 = ~_tasks.get("paused", pd.Series(False, index=_tasks.index)).fillna(False).astype(bool)
            _ok = _tasks[(_tasks["correct"] == True) & _np1]  # noqa: E712
            _blocks.append(
                mo.md(
                    f"**Comprehension**: {len(_tasks)} probes, {100 * _tasks['correct'].mean():.0f}% correct; "
                    f"median time-to-click {_ok['rt_ms'].median():.0f} ms "
                    f"(fastest quartile {_ok['rt_ms'].quantile(0.25):.0f} ms — the gap is what theming can win)."
                )
            )
        _rtp, _rtp_scores = rt_exponent(_log)
        if _rtp_scores and 0.0 in _rtp_scores:
            _gain = _rtp_scores[0.0] - _rtp_scores[_rtp]
            _blocks.append(
                mo.md(
                    f"**The clock's weight is fitted, not assumed**: duels are weighted by "
                    f"(median time / this time) to the power {_rtp}, chosen by held-out log-loss "
                    f"over {{0, ¼, ½, ¾}} and refit every 25 duels. Zero is in that set on purpose — "
                    f"it means ignoring the clock — and it currently loses by {_gain:.4f} nats per "
                    f"duel, so reading a fast click as strong evidence is "
                    + ("earning its keep." if _gain > 0.002 else "not earning much; watch it.")
                )
            )
        _hunts = _frame[_frame["mode"] == "search"]
        if len(_hunts) >= 6:
            _np2 = ~_hunts.get("paused", pd.Series(False, index=_hunts.index)).fillna(False).astype(bool)
            _hok = _hunts[(_hunts["correct"] == True) & _np2]  # noqa: E712
            if len(_hok) >= 4:
                _z = np.polyfit(_hok["salience"], np.log(_hok["rt_ms"]), 1)
                _blocks.append(
                    mo.md(
                        f"**Find hunts**: {len(_hunts)} trials, {100 * _hunts['correct'].mean():.0f}% correct; "
                        f"log time-to-find slope over salience {_z[0]:+.3f} per ΔE "
                        f"(negative = louder is genuinely faster; near zero = salience past this point buys nothing "
                        f"and beauty should take the wheel)."
                    )
                )
        if THRESH_DETAIL.get("day"):
            _blocks.append(
                mo.md(
                    "Constraint provenance — your fitted 75%-correct thresholds in CAM16-UCS ΔE (day / night): "
                    + ", ".join(
                        f"{_ax} {THRESH_DETAIL['day'][_ax]:.1f} / {THRESH_DETAIL['night'][_ax]:.1f}"
                        for _ax in THRESH_DETAIL["day"]
                    )
                    + " — the pairwise floor is 2× the minimum "
                    + f"({2 * DE_MIN['day']:.1f} day, {2 * DE_MIN['night']:.1f} night)."
                )
            )
        _out = mo.vstack(_blocks, gap=0.8)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the numbers, and what happens to them

    The utility is latent and relative — only differences mean anything, so the readout is
    given as *probabilities*: how often the current champion would beat a random feasible
    theme in a duel you haven't run. Marginals near zero on an axis mean your taste is flat
    there (let the constraints or the prior decide); a large one-sided marginal means the
    axis matters and the champion sits where you put it. Early sittings will look noisy —
    a preferential GP needs roughly forty duels before the Thompson arm stops wandering,
    and the 7% uniform probes *should* occasionally look strange: they are the insurance
    premium against a model that only ever asks questions it already believes.

    **One theme or several?** The verdict above is a distribution, not a ranking: sampling
    the joint posterior gives each candidate its probability of being *the* best, and the
    answer is read from how that mass sits. Two details make that honest. Near-identical
    candidates are **grouped** before counting, because eight hundred candidates contain
    many pages that differ by less than you could see and each sibling would otherwise steal
    argmax mass from the others. And the reading is of *cumulative* mass, not a fixed
    cutoff: one group holding over half of it is a winner; a handful sharing it is a real
    plateau, and the members shown are then chosen to be as *different* from each other as
    the set allows; and when even the strongest group is thin, the report says **not yet
    decided** rather than dressing a thin log up as a plateau. Nothing on that shelf is a
    compromise either way — every candidate has already cleared the legibility floors, so a
    plateau means genuinely equal, not merely acceptable.

    Two properties of the machinery were measured rather than assumed, and the tests live in
    `_model_tests.py` beside this file. The position of a card matters to you — over the
    first 79 duels the right-hand card won 61% of the time — so a side-advantage term is
    fitted and subtracted out instead of being left to land on the themes as noise. And the
    nine axes are not equally alive: their length-scales are learned, which shrinks the
    effective dimension the search has to cover, with the estimate held near isotropic until
    enough duels exist to identify relevance at all.

    Reaction time is doing quiet work throughout: a fast duel click steepens that duel's
    likelihood (drift-diffusion reading — big utility gaps decide quickly), a slow one
    flattens it toward a tie, so deliberating over a near-tie neither punishes nor rewards
    either side. That channel is only as clean as its baseline. The first trial of a sitting
    and the first of every run start hidden behind a **begin** button, because those are the
    moments you read an instruction; inside a run the click that produced a trial is its
    anchor, so the clock starts at render and no button stands between you and the next
    page. A **pause** button, the tab losing visibility, or 25 s without a click re-covers the
    stimulus — an exposed one lets a decision form off the clock — and resuming re-baselines;
    a trial paused after its first reveal is flagged in the log: its choice still counts, at
    the neutral slope, and it is excluded from the comprehension and find-hunt timing
    statistics.

    Comprehension probes and find hunts measure time directly, and that time now **binds**:
    a Gaussian process over log time-to-click gives a legibility surface across theme space,
    and candidates it says are credibly slower to read than the fastest are dropped before
    the preference verdict is computed. Constraint first, preference second — the same order
    the contrast floors use, one level deeper: a floor keeps a page readable in principle,
    this keeps it readable in fact. A page you like but read slowly is not a winner. The
    surface estimates its own signal and noise from your times rather than borrowing the
    preference kernel's, because reaction time is noisy enough that a loose prior invents
    differences, and with a thin or noisy log it drops nothing — the honest behaviour rather
    than the convenient one. These arms are also the glyph-scale ground truth that the 2×
    threshold safety margin (from the 104-px vision fit) stands in for until this instrument
    accumulates its own.

    Hard floors are never traded: every page shown clears WCAG 4.5:1 and APCA 60 on body
    tokens, and every pair of colored roles clears twice your measured CAM16-UCS threshold
    for its ground. Literals are one family by measurement, not taste: Horizon's own day
    string and number oranges sit within your threshold of each other. Plain variable reads
    render as ink by standing preference (figure-ground: definitions, literals, and control
    words carry the color).

    The winner's destination: the find-highlight pair lands in `editor.findMatchBackground`
    and `editor.findMatchHighlightBackground` (settings.jsonc already overrides those keys),
    the token colors in `editor.tokenColorCustomizations` per theme, the ground in the
    workbench block — via the champion snippet above, once its posterior stops moving
    between sittings. Trials accumulate in `aesthetics-responses.jsonl`, committed like any
    measurement; the trial generator is deterministic given that log, so any session
    resumes exactly where the last one stopped. Findings that outlive a sitting get written
    into this closing prose, next to the live numbers.
    """)
    return


if __name__ == "__main__":
    app.run()
