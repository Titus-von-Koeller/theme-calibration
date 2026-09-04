# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes -- correct for a notebook holding a model on a GPU, and wrong
# for a gallery, whose whole point is that the picture moves when the control does. Script
# metadata is merged over the project config at the highest precedence, so a notebook opts
# in on its own. `auto_instantiate` cannot be set here (marimo strips it from script
# metadata), so opening this file still runs nothing.
#
# Dependencies are deliberately NOT declared in that block. pixi.toml and pixi.lock are
# this project's one source of truth for the environment; a PEP 723 dependency list here
# would let `marimo edit --sandbox`, or an editor's uv integration, build a second and
# unlocked one behind the first -- the same wrong-environment failure with extra steps.
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
    *The looking half. The duels decide which theme wins and the vision trials decide what
    counts as legible; this page puts the candidates side by side so a human can see what
    the numbers are talking about.*

    # The candidate themes, on trial

    Every theme here is a real point in the search space, realised by the same `realize()`
    the instrument shows in a duel, and rendered by the same `render_card()`. Nothing on
    this page is a mock-up: what you see is the stimulus.

    Three instruments run over each candidate, and none of them is a verdict:

    1. **Contrast on the pixels that render.** WCAG 4.5:1 and APCA |Lc| 60 for body tokens,
       45 for context tokens, measured on the quantised hex a browser will actually write
       rather than on the unrounded value the search produced. A theme that fails here was
       never shown in a duel; the columns exist so the floor is visible rather than trusted.
    2. **Pairwise separation against the measured observer.** Every pair of coloured roles
       has to clear twice the CAM16-UCS threshold fitted in `notebooks/vision.py`. Twice,
       not once, because the threshold is measured on 104 px patches and read on glyphs, and
       contrast sensitivity falls with glyph scale. The column shows the worst pair and how
       much room it has.
    3. **Deuteranopia and grayscale.** Simulated with the Machado, Oliveira & Fernandes
       (2009) transform, the standard physiological model of red-green deficiency, and by
       isolating the luminance channel. These are *population* bounds and this repo has
       something better for one observer -- a fitted confusion axis and a fitted weight
       ellipse. They are kept because they answer a different question: whether a palette's
       distinctions survive for a reader who is not the one being measured, and whether they
       survive a printout.

    What no instrument settles is which of the survivors is worth reading all day. That is
    what the duels are for, and this page is where you check that the thing winning them is
    the thing you thought you were choosing.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json

    import altair as alt
    import numpy as np
    import pandas as pd

    try:
        import theme  # noqa: F401  -- an environment check, not a use; see below
    except ModuleNotFoundError as no_package:
        # theme-calibration is installed editable into this project's pixi environment and
        # into no other, so a missing `theme` means this notebook is running on the wrong
        # interpreter -- an editor kernel resolved for a parent folder is the way it
        # happens. Left unguarded that surfaces as NameError on every cell downstream of
        # this one, which names the symptom in a dozen places and the cause in none.
        raise ModuleNotFoundError(
            "gallery.py is running on an interpreter that does not have theme-calibration "
            "installed, so nothing below this cell can run. Start it with `pixi run gallery`, or "
            "select .pixi/envs/default/bin/python as this folder's interpreter. "
            "README, 'Running it', has both."
        ) from no_package

    from theme.color import apca_lc, composite, hex_to_rgb, rel_lum, ucs_dist, wcag
    from theme.observer import discriminability
    from theme.observer import fit as observer_fit
    from theme.paths import CHAMPION, VISION_LOG
    from theme.space import DE_MIN, POOL
    from theme.stimulus import render_card, snippet_for

    # Okabe & Ito published this eight-colour palette in 2002 as part of Japan's
    # universal-design movement, engineered so every pair stays distinguishable under the
    # common colour-vision deficiencies; it has since become the cross-field consensus for
    # categorical colour. It is on this page as a calibration bar and not as a candidate: an
    # editor theme is not a categorical palette, but if a candidate's roles separate less
    # well than these eight do, the shortfall is the candidate's.
    OKABE_ITO = {
        "black": "#000000",
        "orange": "#E69F00",
        "sky blue": "#56B4E9",
        "bluish green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish purple": "#CC79A7",
    }

    #: The roles a rendered page actually paints, in the order the tables list them.
    ROLES = ("keyword", "function", "string", "ink", "comment", "punct")
    #: Which of those carry meaning (body floor) rather than context (relaxed floor).
    BODY_ROLES = ("keyword", "function", "string", "ink")

    return (
        BODY_ROLES,
        CHAMPION,
        DE_MIN,
        OKABE_ITO,
        POOL,
        ROLES,
        VISION_LOG,
        alt,
        apca_lc,
        composite,
        discriminability,
        hex_to_rgb,
        json,
        np,
        observer_fit,
        pd,
        rel_lum,
        render_card,
        snippet_for,
        ucs_dist,
        wcag,
    )


@app.cell(hide_code=True)
def _(hex_to_rgb, np, rel_lum):
    # Machado, Oliveira & Fernandes (2009), IEEE TVCG: physiologically-based simulation of
    # dichromacy. This is their deuteranopia matrix at severity 1.0, applied in LINEAR RGB --
    # applying it to gamma-encoded values is the common mistake and it flatters the result.
    _DEUTAN = np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    )

    def _to_linear(c):
        c = np.asarray(c, dtype=float)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def _to_srgb(c):
        c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)

    def _as_hex(rgb):
        return "#" + "".join(f"{round(255 * float(v)):02x}" for v in np.clip(rgb, 0, 1))

    def luminance(hex_color):
        """WCAG relative luminance, 0..1 -- the quantity a grayscale printer keeps.

        Delegates the transfer function to theme.color so this page and the instrument agree
        on what luminance means; only the scalar unwrap is local.
        """
        return float(rel_lum(hex_to_rgb(hex_color))[0])

    def simulate(hex_color, mode):
        """One hex as a stated observer would meet it: as designed, deuteranope, or printed."""
        if mode == "as designed":
            return hex_color
        if mode == "deuteranopia":
            return _as_hex(_to_srgb(_DEUTAN @ _to_linear(hex_to_rgb(hex_color)[0])))
        _g = _to_srgb(luminance(hex_color))
        return _as_hex(np.array([_g, _g, _g]))

    return luminance, simulate


@app.cell(hide_code=True)
def _(CHAMPION, POOL, composite, json, np):
    # The candidate set. The published champion per polarity, the two Horizon originals it is
    # trying to beat, and a spread of the feasible pool so the page shows the range the
    # search is choosing from rather than only its current favourite.
    #
    # The champion file carries only the roles an applier needs to write. The two derived
    # roles and the find layer are reconstructed here exactly as space.py's `_assemble`
    # derives them -- number follows string, variable renders as ink, and the find pair is
    # the fill composited at 0.85 and 0.45 -- so a champion card renders identically to the
    # theme it was measured as. Deriving them differently would make this page a picture of a
    # theme nobody was ever shown.
    def _complete(theme):
        _t = dict(theme)
        _t.setdefault("number", _t["string"])
        _t.setdefault("variable", _t["ink"])
        if "find_fill" in _t:
            _t.setdefault("find_current", composite(_t["find_fill"], 0.85, _t["ground"]))
            _t.setdefault("find_other", composite(_t["find_fill"], 0.45, _t["ground"]))
        return _t

    # Horizon's own token colours, already alpha composited onto their page (night carries
    # 30% alpha on comments; measuring before compositing is the mistake that once made this
    # theme look AA-clean). Literal rather than scraped from `~/.vscode/extensions`, so this
    # page reads the same on a machine with no editor extension installed.
    HORIZON = {
        "day": {
            "ground": "#fdf0ed",
            "keyword": "#8a31b9",
            "function": "#1d8991",
            "string": "#f6661e",
            "number": "#f77d26",
            "variable": "#e84a72",
            "ink": "#3a3634",
            "comment": "#989190",
            "punct": "#6c6764",
        },
        "night": {
            "ground": "#1c1e26",
            "keyword": "#a96ec9",
            "function": "#24a2ad",
            "string": "#e4a88a",
            "number": "#e4a88a",
            "variable": "#e95378",
            "ink": "#cfcac6",
            "comment": "#4c4d53",
            "punct": "#9a958f",
        },
    }

    _published = {}
    if CHAMPION.exists():
        try:
            _published = json.loads(CHAMPION.read_text())
        except json.JSONDecodeError:
            _published = {}

    CANDIDATES = {"day": [], "night": []}
    for _pol in ("day", "night"):
        if _pol in _published:
            CANDIDATES[_pol].append((f"measured champion ({_pol})", _complete(_published[_pol])))
        CANDIDATES[_pol].append((f"Horizon ({_pol})", _complete(HORIZON[_pol])))
        # A deterministic spread of the feasible pool, ordered by ground lightness so the
        # rows read from the palest page to the deepest rather than in pool order.
        _pool = POOL[_pol]
        if _pool:
            _by_light = sorted(range(len(_pool)), key=lambda _j: _pool[_j][1]["ground"])
            _idx = np.linspace(0, len(_by_light) - 1, min(4, len(_by_light))).round().astype(int)
            for _i in sorted({int(_by_light[int(_j)]) for _j in _idx}):
                CANDIDATES[_pol].append((f"pool #{_i} ({_pol})", _pool[_i][1]))

    return (CANDIDATES,)


@app.cell(hide_code=True)
def _(mo):
    polarity = mo.ui.radio(options=["day", "night"], value="day", label="Page polarity", inline=True)
    mo.output.replace(polarity)
    return (polarity,)


@app.cell(hide_code=True)
def _(CANDIDATES, mo, polarity):
    # Its own cell, so that changing polarity rebuilds the list of candidates to choose from.
    # Built in one cell with the radio, the dropdown would keep offering the day candidates
    # after a switch to night and hand the renderer an index into the wrong list.
    _names = [_name for _name, _ in CANDIDATES[polarity.value]]
    candidate = mo.ui.dropdown(
        options={_name: _i for _i, _name in enumerate(_names)},
        value=_names[0] if _names else None,
        label="Card to render below",
    )
    mo.output.replace(candidate)
    return (candidate,)


@app.cell(hide_code=True)
def _(
    BODY_ROLES,
    CANDIDATES,
    DE_MIN,
    ROLES,
    VISION_LOG,
    apca_lc,
    discriminability,
    hex_to_rgb,
    mo,
    np,
    observer_fit,
    pd,
    polarity,
    ucs_dist,
    wcag,
):
    # The measurement table. One row per candidate, and every column is a floor the
    # instrument enforces rather than a score it optimises: the only question the search ever
    # asks is which of the already-legible candidates is better.
    _pol = polarity.value
    _floor = DE_MIN[_pol]
    _fit = observer_fit(VISION_LOG) if VISION_LOG.exists() else None

    def _role_measures(theme):
        """(worst WCAG, worst body Lc, worst context Lc) for one theme's roles."""
        _hexes = [theme[_r] for _r in ROLES]
        _fg = hex_to_rgb(_hexes)
        _bg = np.repeat(hex_to_rgb([theme["ground"]]), len(_hexes), axis=0)
        _lc = np.abs(apca_lc(_fg, _bg))
        _body = [_lc[_i] for _i, _r in enumerate(ROLES) if _r in BODY_ROLES]
        _context = [_lc[_i] for _i, _r in enumerate(ROLES) if _r not in BODY_ROLES]
        return float(wcag(_fg, _bg).min()), float(min(_body)), float(min(_context))

    def _worst_pair(theme):
        """(separation in dE, the pair) for the closest two coloured roles."""
        _worst, _which = float("inf"), None
        for _i, _a in enumerate(ROLES):
            for _b in ROLES[_i + 1 :]:
                _d = float(ucs_dist(theme[_a], theme[_b])[0])
                if _d < _worst:
                    _worst, _which = _d, f"{_a}/{_b}"
        return _worst, _which

    _rows = []
    for _name, _theme in CANDIDATES[_pol]:
        _min_wcag, _min_body, _min_ctx = _role_measures(_theme)
        _sep, _pair = _worst_pair(_theme)
        _row = {
            "candidate": _name,
            "ground": _theme["ground"],
            "worst WCAG": round(_min_wcag, 2),
            "worst body Lc": round(_min_body, 1),
            "worst context Lc": round(_min_ctx, 1),
            "closest roles": _pair,
            "their dE": round(_sep, 2),
            f"vs floor (2x{_floor:.1f})": round(_sep - 2 * _floor, 2),
        }
        if _fit is not None:
            _a, _b = _pair.split("/")
            _row["P(tell apart)"] = round(discriminability(_fit, _theme[_a], _theme[_b], _theme["ground"]), 3)
        _rows.append(_row)

    _note = (
        f"Floors for the **{_pol}** page: WCAG 4.5 on every role, APCA |Lc| 60 on body roles "
        f"and 45 on context roles, and {2 * _floor:.1f} dE between any two coloured roles "
        f"(twice the {_floor:.1f} dE measured at 104 px, because these are read at glyph "
        f"scale). A negative number in the last column is a theme the search would refuse to "
        f"show."
    )
    if _fit is not None:
        _note += (
            f" `P(tell apart)` is the fitted observer's probability of separating that "
            f"closest pair in the four-square task, from {_fit.n:,} calibration trials -- "
            f"0.25 is chance."
        )
    else:
        _note += " No calibration log on this machine, so the observer column is absent."
    mo.output.replace(mo.vstack([mo.md(_note), mo.ui.table(pd.DataFrame(_rows), selection=None)], gap=0.6))
    return


@app.cell(hide_code=True)
def _(CANDIDATES, candidate, mo, polarity, render_card, snippet_for):
    # A real page, not a swatch row. Two things only a rendered page shows: how much of the
    # screen each role actually occupies (punctuation and ink are most of the pixels, and a
    # swatch strip gives them the same weight as a keyword), and whether the comment colour
    # recedes or disappears.
    #
    # One fixed snippet for every candidate, so what changes between renders is the theme and
    # nothing else.
    _pol = polarity.value
    _pairs = CANDIDATES[_pol]
    _i = candidate.value if isinstance(candidate.value, int) and candidate.value < len(_pairs) else 0
    _name, _theme = _pairs[_i]
    _snip = snippet_for(0)
    mo.output.replace(
        mo.vstack(
            [
                mo.md(f"### {_name}"),
                mo.Html(
                    f'<div style="background:{_theme["ground"]};border-radius:10px;padding:20px;'
                    f'max-width:640px;overflow:hidden">' + render_card(_theme, _snip, 14, prose=False) + "</div>"
                ),
            ],
            gap=0.5,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same palette, as three observers meet it

    Each candidate's roles under the three instruments. The number on every swatch is its
    relative luminance (0-100), which is the quantity that survives a grayscale printer and
    the channel fine spatial detail rides on (Ware): if two roles have the same luminance,
    they are the same colour to the part of the visual system that resolves glyph shapes,
    whatever their hue does.

    Read the deuteranopia row for *collapses*, not for prettiness. Two roles that separate by
    hue alone converge there, and a reader with the deficiency then has one colour where the
    theme intended two. The grayscale row is the same test with hue removed entirely.
    """)
    return


@app.cell(hide_code=True)
def _(CANDIDATES, ROLES, alt, luminance, mo, pd, polarity, simulate):
    _pol = polarity.value
    _modes = ("as designed", "deuteranopia", "grayscale")
    _rows = [
        {
            "candidate": _name,
            "mode": _mode,
            "i": _i,
            "role": _role,
            "fill": simulate(_theme[_role], _mode),
            "page": simulate(_theme["ground"], _mode),
            "y": round(100 * luminance(simulate(_theme[_role], _mode))),
        }
        for _name, _theme in CANDIDATES[_pol]
        for _mode in _modes
        for _i, _role in enumerate(ROLES)
    ]
    _frame = pd.DataFrame(_rows)
    _order = [_name for _name, _ in CANDIDATES[_pol]]
    _at = {
        "x": alt.X("role:N", axis=alt.Axis(title=None, labelAngle=-40), sort=list(ROLES)),
        "y": alt.Y("candidate:N", axis=alt.Axis(title=None, domain=False, ticks=False), sort=_order),
    }
    _base = alt.Chart(_frame)
    # Ink flips on the swatch's own luminance rather than on a fixed threshold, so the number
    # stays readable on both a near-white string and a near-black keyword. That is the same
    # rule the themes themselves are held to, applied to this page's own labels.
    _chart = (
        _base.mark_rect().encode(**_at, color=alt.Color("fill:N", scale=None))
        + _base.mark_text(fontSize=10, fontWeight=600).encode(
            **_at,
            text="y:Q",
            color=alt.condition("datum.y > 45", alt.value("#111111"), alt.value("#f4f4f4")),
        )
    ).properties(width=52 * len(ROLES), height=30 * len(_order))
    mo.output.replace(
        mo.ui.altair_chart(
            _chart.facet(column=alt.Column("mode:N", title=None, sort=list(_modes))),
            chart_selection=False,
            legend_selection=False,
        )
    )
    return


@app.cell(hide_code=True)
def _(OKABE_ITO, ROLES, alt, luminance, mo, pd, simulate):
    # The calibration bar: a palette engineered for exactly this test, run through exactly
    # this test. Its deuteranopia row should show no collapse, which is what makes it a bar
    # rather than another candidate. If this row ever collapses, the bug is on this page.
    _modes = ("as designed", "deuteranopia", "grayscale")
    _frame = pd.DataFrame(
        [
            {
                "mode": _mode,
                "name": _name,
                "i": _i,
                "fill": simulate(_hex, _mode),
                "y": round(100 * luminance(simulate(_hex, _mode))),
            }
            for _mode in _modes
            for _i, (_name, _hex) in enumerate(OKABE_ITO.items())
        ]
    )
    _at = {
        "x": alt.X("i:O", axis=None, scale=alt.Scale(paddingInner=0.1)),
        "y": alt.Y("mode:N", axis=alt.Axis(title=None, domain=False, ticks=False), sort=list(_modes)),
    }
    _base = alt.Chart(_frame)
    _chart = (
        _base.mark_rect().encode(**_at, color=alt.Color("fill:N", scale=None), tooltip=["name:N", "y:Q"])
        + _base.mark_text(fontSize=10, fontWeight=600).encode(
            **_at,
            text="y:Q",
            color=alt.condition("datum.y > 45", alt.value("#111111"), alt.value("#f4f4f4")),
        )
    ).properties(width=52 * len(ROLES), height=30 * len(_modes))
    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    "**Okabe-Ito, as the bar.** Eight colours engineered in 2002 to stay "
                    "pairwise distinguishable under the common deficiencies. A categorical "
                    "palette has no order, so these luminance numbers should *not* be "
                    "monotonic; they show which labels survive a printer and how much "
                    "luminance variety the palette spends telling neighbours apart."
                ),
                mo.ui.altair_chart(_chart, chart_selection=False, legend_selection=False),
            ],
            gap=0.5,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the rules on this page come from

    The floors and the reading rules are assembled from results, not tastes:

    - **Jacques Bertin**, *Semiologie graphique* (1967) -- the first theory of visual
      variables; established that lightness reads as ordered while hue does not. The root of
      "magnitude is carried by lightness and only by lightness", and of why a token palette
      may use hue for identity but must not use it for emphasis.
    - **Colin Ware**, *Information Visualization* (2000) -- fine spatial detail rides the
      luminance channel, so text and shape contrast must be luminance contrast. This is why
      the contrast floors are APCA and WCAG rather than a colour-difference metric: a glyph
      is spatial detail before it is a colour.
    - **Cynthia Brewer** -- ColorBrewer (2003): palettes as engineered instruments with
      declared roles. The idea that a palette can be specified and tested rather than
      admired.
    - **Masataka Okabe & Kei Ito** (2002) -- the colour-universal-design palette used as the
      bar above.
    - **Machado, Oliveira & Fernandes** (2009, IEEE TVCG) -- the physiological dichromacy
      simulation this page runs, and the reason it runs in linear light.
    - **Li, Luo et al.** (2017) -- CAM16-UCS, the appearance space every threshold and every
      separation in this repo is stated in.
    - **Somers**, APCA (W3C/SAPC work, 0.1.9 4g here) -- perceptual lightness contrast, which
      unlike the WCAG ratio does not treat a dark page and a light page as the same problem.
      Both are enforced, because APCA is not yet a normative standard and WCAG 4.5:1 is.
    - **Fabio Crameri, Grace Shephard & Philip Heron**, *The misuse of colour in science
      communication* (Nature Communications, 2020) -- the test battery of uniformity, CVD and
      grayscale that this page's three columns are a specialisation of.
    - **Tamara Munzner**, *Visualization Analysis & Design* (2014) -- identity to hue,
      magnitude to lightness/position. The one-sentence version of the whole page.

    ## What this page is not

    It is not the verdict. The verdict is in `notebooks/analysis.py`, it is a distribution
    rather than a ranking, and it is computed over candidates bred fresh by the search rather
    than over the handful shown here. This page exists so that the thing winning the duels
    can be looked at, and so a floor that is silently doing nothing can be caught by eye.

    It is also not a chart-palette gallery. Sequential and diverging colour maps -- viridis,
    cividis, batlow and the rest -- are a different problem with a different literature, and
    they used to fill this notebook because it began life beside a plotting library. They are
    gone, along with the plotting stack that drew them: what is left is the palette this repo
    is actually measuring.
    """)
    return


if __name__ == "__main__":
    app.run()
