# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes -- correct for a notebook holding a model on the GPU, and
# fatal for a slider, whose whole point is that the picture moves while you drag. Script
# metadata is merged over the project config at the highest precedence, so a notebook
# opts in on its own. `auto_instantiate` cannot be set here (marimo strips it from script
# metadata), so opening this file still runs nothing.
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
    *A sidecar to the PyTorch basics series — the viewing vocabulary of `_viz.py`, drawn, tested,
    and set beside the world's alternatives so the choice of theme is a decision, not a habit.*

    # The theme, on trial

    "Theme" means two independent layers here. The **reading theme** is VSCode's: this machine
    flips between Horizon Bright and Horizon Bold with the OS, so every exhibit is read on a
    near-white page by day and a near-black one at night. The **exhibit color system** is ours,
    in `_viz.py`: squares own their fills and inks, the page owns the gaps, so one drawing
    survives both pages. This notebook puts that system on trial: what it looks like, what the
    field's alternatives look like on the same data, and the three tests that separate a color
    scheme from a decoration —

    1. **Lightness is monotonic in the data** — magnitude must be readable from luminance alone.
    2. **It survives deuteranopia** — simulated below with the Machado, Oliveira & Fernandes
       (2009) transform, the standard model of red-green color-vision deficiency.
    3. **It survives grayscale** — the printout test, which is the luminance channel isolated.

    Every swatch on this page is computed at runtime from the palette's own definition; the
    number on each swatch is its relative luminance (0–100). If the numbers rise monotonically,
    test 1 passes in front of you.

    The tests are instruments, not verdicts. The galleries below are deliberately wide — loved
    classics and known failures included — because two other things matter that no simulation
    settles: which rows *your* eyes separate cleanly (the deuteranopia column is a worst-case
    bound; your own comparison across the "as designed" column is the real measurement), and
    which of the survivors you actually love. Beauty is allowed to vote; it just does not get
    to overrule the instruments.
    """)
    return


@app.cell(hide_code=True)
def _():
    import altair as alt
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    import torch
    from _viz import INK_DARK, INK_LIGHT, OKABE_ITO, POLARITY, RAMP, show
    from cmcrameri import cm as cmc

    return INK_DARK, INK_LIGHT, OKABE_ITO, POLARITY, RAMP, alt, cmc, mpl, np, pd, show, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The system as it ships

    `show()` with the two ramps `_viz.py` defines: sequential (unsigned data) and diverging
    (signed data, hue carries the sign and nothing else). Self-identifying data, so every square
    names its own position.
    """)
    return


@app.cell(hide_code=True)
def _(mo, show, torch):
    mo.hstack(
        [
            show(torch.arange(12).reshape(3, 4), "sequential: RAMP"),
            show(torch.arange(-6, 6).reshape(3, 4), "diverging: POLARITY"),
        ],
        justify="start",
        gap=3,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(INK_DARK, INK_LIGHT, alt, mpl, np, pd):
    def _srgb_to_linear(c):
        c = np.asarray(c, dtype=float)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def _linear_to_srgb(c):
        c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)

    # Machado, Oliveira & Fernandes (2009), IEEE TVCG: physiologically-based simulation of
    # dichromacy. This is their deuteranopia matrix at severity 1.0, applied in linear RGB.
    _DEUTAN = np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    )

    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])

    def _rgb_to_hex(rgb):
        return "#" + "".join(f"{round(255 * v):02x}" for v in np.clip(rgb, 0, 1))

    def luminance(hex_color):
        """WCAG relative luminance, 0..1 — the quantity grayscale printing keeps."""
        r, g, b = _srgb_to_linear(_hex_to_rgb(hex_color))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def simulate(hex_color, mode):
        if mode == "as designed":
            return hex_color
        if mode == "deuteranopia":
            linear = _srgb_to_linear(_hex_to_rgb(hex_color))
            return _rgb_to_hex(_linear_to_srgb(_DEUTAN @ linear))
        gray = _linear_to_srgb(luminance(hex_color))
        return _rgb_to_hex(np.array([gray, gray, gray]))

    def sample_cmap(name, n=7):
        """n hexes from a colormap (matplotlib name or Colormap object) — its definition."""
        cmap = mpl.colormaps[name] if isinstance(name, str) else name
        if hasattr(cmap, "colors") and len(cmap.colors) < 30:
            return [_rgb_to_hex(np.asarray(c)[:3]) for c in cmap.colors[:n]]
        return [_rgb_to_hex(np.asarray(cmap(t))[:3]) for t in np.linspace(0, 1, n)]

    def gallery(palettes, modes=("as designed", "deuteranopia", "grayscale")):
        """Each palette under each simulation, luminance printed on every swatch."""
        _rows = [
            {
                "palette": name,
                "mode": mode,
                "i": i,
                "fill": simulate(h, mode),
                "y": round(100 * luminance(simulate(h, mode))),
            }
            for name, hexes in palettes.items()
            for mode in modes
            for i, h in enumerate(hexes)
        ]
        _frame = pd.DataFrame(_rows)
        _order = list(palettes)
        _at = {
            "x": alt.X("i:O", axis=None, scale=alt.Scale(paddingInner=0.08)),
            "y": alt.Y("palette:N", axis=alt.Axis(title=None, domain=False, ticks=False), sort=_order),
        }
        _base = alt.Chart(_frame)
        return (
            (
                _base.mark_rect().encode(**_at, color=alt.Color("fill:N", scale=None))
                + _base.mark_text(fontSize=11, fontWeight=500).encode(
                    **_at,
                    text="y:Q",
                    color=alt.condition("datum.y > 42", alt.value(INK_DARK), alt.value(INK_LIGHT)),
                )
            )
            .properties(width=34 * 9, height=30 * len(palettes))
            .facet(column=alt.Column("mode:N", title=None, sort=list(modes)))
        )

    return gallery, luminance, sample_cmap, simulate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sequential: one direction of magnitude

    The house `RAMP` beside the field, canon and contested alike. The canon: `viridis` (Smith &
    van der Walt, 2015) ended matplotlib's rainbow default with uniformity by construction;
    `cividis` (Nuñez, Anderton & Renslow, 2018) re-derived it so a deuteranope and a trichromat
    see *the same* map; `plasma`, `inferno` and `magma` are its higher-drama siblings;
    `cubehelix` (Dave Green, 2011, radio astronomy) is built to survive grayscale by
    construction; `Blues` and `YlGnBu` are ColorBrewer's takes. The contested: `turbo`
    (Google, 2019) is the rainbow rebuilt to fix jet's worst artifacts — widely loved, still
    non-monotonic in luminance — and `jet` itself, MATLAB's default until 2014 and matplotlib's
    until 2017, is kept here as the specimen: watch its luminance numbers rise *and fall*,
    which is the mechanism by which it paints ridges into smooth data. Fabio Crameri's
    `batlow` — the current geoscience reference — comes from his own `cmcrameri` package, so
    its row below is sampled from the map's actual definition, not a transcription.

    The uniform maps differ mostly in *local contrast* — multi-hue maps spend more perceptual
    distance per step than a single-hue ramp — and in temperament. Compare rows in the first
    column with your own eyes before reading anyone's verdict, including mine.
    """)
    return


@app.cell(hide_code=True)
def _(RAMP, cmc, gallery, mo, sample_cmap):
    mo.ui.altair_chart(
        gallery(
            {
                "RAMP (house)": RAMP,
                "cividis": sample_cmap("cividis"),
                "batlow (Crameri)": sample_cmap(cmc.batlow),
                "viridis": sample_cmap("viridis"),
                "plasma": sample_cmap("plasma"),
                "inferno": sample_cmap("inferno"),
                "magma": sample_cmap("magma"),
                "cubehelix": sample_cmap("cubehelix"),
                "Blues (Brewer)": sample_cmap("Blues"),
                "YlGnBu (Brewer)": sample_cmap("YlGnBu"),
                "turbo — contested": sample_cmap("turbo"),
                "jet — the specimen": sample_cmap("jet"),
            }
        ),
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diverging: sign carried by hue, magnitude by lightness

    The house `POLARITY` (blue-orange) beside the field's spread: ColorBrewer's `RdBu`, `PuOr`,
    `BrBG`, `PRGn` and `PiYG`, Moreland's `coolwarm` (the CFD staple), `seismic` (geophysics),
    and the two beloved red-through-green classics, `RdYlBu` and `RdYlGn`. The deuteranopia
    column is where diverging maps live or die: a diverging map's whole job is keeping its two
    arms apart, and red-against-green collapses there into a single mud axis — the *sign* of
    the data disappears. Blue-orange and purple-orange survive for every reader, which is why
    they are the recommended pairs; the rest is temperament, and several rows here are
    genuinely lovely on the first column. Check which ones still work for you on the second.
    """)
    return


@app.cell(hide_code=True)
def _(POLARITY, gallery, mo, sample_cmap):
    mo.ui.altair_chart(
        gallery(
            {
                "POLARITY (house)": POLARITY,
                "RdBu (Brewer)": sample_cmap("RdBu"),
                "PuOr (Brewer)": sample_cmap("PuOr"),
                "BrBG (Brewer)": sample_cmap("BrBG"),
                "PRGn (Brewer)": sample_cmap("PRGn"),
                "PiYG (Brewer)": sample_cmap("PiYG"),
                "coolwarm (Moreland)": sample_cmap("coolwarm"),
                "seismic": sample_cmap("seismic"),
                "RdYlBu": sample_cmap("RdYlBu"),
                "RdYlGn": sample_cmap("RdYlGn"),
            }
        ),
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical: identity carried by hue

    Okabe & Ito published this eight-color palette in 2002 as part of Japan's universal-design
    movement, engineered so every pair stays distinguishable under the common color-vision
    deficiencies; it has since become the cross-field consensus for categorical color.
    The field it competes with, from Tableau's `tab10` (today's most common default) through
    Brewer's qualitative sets to `tab20`, which trades safety for headcount. Watch each row's
    reds against its greens in the deuteranopia column — and note how the pastel sets fare on
    a grayscale printer. A categorical palette has no order, so the luminance numbers should
    *not* be monotonic here; they only show which labels survive printing, and how much
    luminance variety the palette spends on telling neighbors apart.
    """)
    return


@app.cell(hide_code=True)
def _(OKABE_ITO, gallery, mo, sample_cmap):
    mo.ui.altair_chart(
        gallery(
            {
                "Okabe-Ito (house)": list(OKABE_ITO.values())[:8],
                "tab10 (Tableau)": sample_cmap("tab10", 8),
                "Dark2 (Brewer)": sample_cmap("Dark2", 8),
                "Set1 (Brewer)": sample_cmap("Set1", 8),
                "Set2 (Brewer)": sample_cmap("Set2", 8),
                "Paired (Brewer)": sample_cmap("Paired", 8),
                "Accent (Brewer)": sample_cmap("Accent", 8),
                "tab20": sample_cmap("tab20", 8),
            }
        ),
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same ramp on both pages

    The reading theme flips daily, so an exhibit must sit on both grounds. Squares own their
    fills; the frame around each row below simulates the two Horizon pages. Nothing in the
    swatches changes between the rows — only the page does, which is the point: the system
    delegates the gaps to the theme and keeps every color-on-color contrast internal, where it
    was measured.
    """)
    return


@app.cell(hide_code=True)
def _(RAMP, alt, mo, pd):
    _grounds = {"on Horizon Bright's page (day)": "#fdf0ed", "on Horizon Bold's page (night)": "#1c1e26"}
    _frame = pd.DataFrame(
        [
            {"ground": g, "i": i, "fill": c, "kind": "swatch"}
            for g, _hex in _grounds.items()
            for i, c in enumerate(RAMP)
        ]
        + [{"ground": g, "i": -1, "fill": hex_, "kind": "page"} for g, hex_ in _grounds.items()]
    )
    _page = (
        alt.Chart(_frame[_frame.kind == "page"])
        .mark_rect()
        .encode(
            x=alt.value(0),
            x2=alt.value(34 * 7),
            y=alt.Y("ground:N", axis=alt.Axis(title=None, domain=False, ticks=False)),
            color=alt.Color("fill:N", scale=None),
        )
    )
    _swatches = (
        alt.Chart(_frame[_frame.kind == "swatch"])
        .mark_rect(width=24, height=24)
        .encode(
            x=alt.X("i:O", axis=None, scale=alt.Scale(paddingInner=0.35)),
            y=alt.Y("ground:N", axis=alt.Axis(title=None, domain=False, ticks=False)),
            color=alt.Color("fill:N", scale=None),
        )
    )
    mo.ui.altair_chart(
        (_page + _swatches).properties(width=34 * 7, height=110), chart_selection=False, legend_selection=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The editor theme, under the same instruments

    The reading theme deserves the trial it hosts. Below, the two Horizon variants actually
    installed on this machine — sampled from the extension's own JSON, alpha composited onto
    each theme's page before measuring — with the WCAG contrast ratio printed under every
    accent, on that theme's own ground, as designed and under deuteranopia. AA for body-size
    text is 4.5:1; ratios below it are flagged.

    As measured on `horizon-theme-vscode 1.0.1`, alpha included: the **day variant fails AA on
    four of six roles** — comments and strings at 2.8:1, links at 3.3:1, functions at 3.7:1
    against the warm page — and the **night variant on two**: comments at 2.0:1 (they carry 30%
    alpha) and variables at 4.1:1. An earlier draft of this paragraph, written from a probe
    that skipped alpha compositing, claimed the night variant cleared AA everywhere; the
    exhibit below corrected its own caption. The palette's two pink-family roles, links (345°)
    and variables (346°), sit one hue degree apart — a distinction riding the weakest axis of
    red-green vision even before simulation; watch them converge in the deuteranopia rows. The
    warm paper ground and the day/night pairing are genuine strengths; accent contrast, on both
    pages once alpha is honest, is where an override would earn its keep.
    """)
    return


@app.cell(hide_code=True)
def _(OKABE_ITO, alt, luminance, mo, pd, simulate):
    import json as _json
    import re as _re
    from pathlib import Path as _Path

    _theme_files = {
        "Horizon Bright Bold — day": "horizon-bright-bold.json",
        "Horizon Bold — night": "horizon-bold.json",
    }
    _roots = list(_Path.home().glob(".vscode/extensions/*horizon*/themes/"))

    def _load(name):
        raw = (_roots[0] / name).read_text()
        return _json.loads(_re.sub(r"//[^\n\"]*$", "", raw, flags=_re.M))

    def _composite(hex_color, page):
        """An 8-digit hex carries alpha; what the eye meets is the blend onto the page."""
        raw = hex_color.lstrip("#")
        rgb = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
        alpha = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1.0
        bg = [int(page.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
        return "#" + "".join(f"{round(alpha * c + (1 - alpha) * b):02x}" for c, b in zip(rgb, bg, strict=True))

    def _contrast(fg, bg):
        _hi, _lo = sorted([luminance(fg), luminance(bg)], reverse=True)
        return (_hi + 0.05) / (_lo + 0.05)

    _wanted = ["keyword", "string", "variable", "entity.name.function", "comment"]
    _rows = []
    if _roots:
        for _label, _file in _theme_files.items():
            _t = _load(_file)
            _page_color = _t["colors"]["editor.background"]
            _accents = {"link": _t["colors"].get("textLink.foreground")}
            for _entry in _t.get("tokenColors", []):
                _scopes = _entry.get("scope", [])
                _scopes = [_scopes] if isinstance(_scopes, str) else _scopes
                _color = _entry.get("settings", {}).get("foreground")
                if _color:
                    for _want in _wanted:
                        _short = _want.split(".")[-1]
                        if _short not in _accents and any(_s == _want or _s.startswith(_want) for _s in _scopes):
                            _accents[_short] = _color
            for _mode in ("as designed", "deuteranopia"):
                for _i, (_role, _color) in enumerate(_accents.items()):
                    _flat = _composite(_color, _page_color)
                    _rows.append(
                        {
                            "theme": _label,
                            "mode": _mode,
                            "i": _i,
                            "role": _role,
                            "fill": simulate(_flat, _mode),
                            "page": simulate(_page_color, _mode),
                            "ratio": round(_contrast(_flat, _page_color), 1),
                            "low": _contrast(_flat, _page_color) < 4.5,
                        }
                    )

    if not _rows:
        _out = mo.md("*No Horizon theme found under `~/.vscode/extensions` on this machine.*")
    else:
        _frame = pd.DataFrame(_rows)
        _at = {
            "x": alt.X("i:O", axis=None, scale=alt.Scale(paddingInner=0.25)),
            "y": alt.Y("mode:N", axis=alt.Axis(title=None, domain=False, ticks=False)),
        }
        _charts = []
        for _label in _theme_files:
            _sub = _frame[_frame.theme == _label]
            _ground = (
                alt.Chart(_sub)
                .mark_rect()
                .encode(x=alt.value(0), x2=alt.value(6 * 78), **{"y": _at["y"]}, color=alt.Color("page:N", scale=None))
            )
            _swatch = (
                alt.Chart(_sub).mark_rect(width=58, height=24).encode(**_at, color=alt.Color("fill:N", scale=None))
            )
            # Role names and ratios are set in the accent's own color on the theme's own
            # page: their legibility right here is the measurement, re-performed by your eyes.
            _names = (
                alt.Chart(_sub)
                .mark_text(fontSize=10, dy=22, fontWeight=500)
                .encode(**_at, text="role:N", color=alt.Color("fill:N", scale=None))
            )
            _ratios = (
                alt.Chart(_sub)
                .mark_text(fontSize=11, dy=36, fontWeight=600)
                .encode(
                    **_at,
                    text=alt.Text("ratio:Q", format=".1f"),
                    color=alt.condition(
                        "datum.low", alt.value(OKABE_ITO["vermillion"]), alt.Color("fill:N", scale=None)
                    ),
                )
            )
            _charts.append((_ground + _swatch + _names + _ratios).properties(width=6 * 78, height=180, title=_label))
        _out = mo.vstack([mo.ui.altair_chart(_c, chart_selection=False, legend_selection=False) for _c in _charts])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the rules come from

    The system in `_viz.py` is assembled from results, not tastes. The lineage, and what each
    contributor settled:

    - **Jacques Bertin**, *Sémiologie graphique* (1967) — the first theory of visual variables;
      established that lightness (*value*) reads as ordered while hue does not. The root of
      "magnitude is carried by lightness and only by lightness."
    - **Edward Tufte**, *The Visual Display of Quantitative Information* (1983) — data-ink ratio
      and chartjunk. Why `show()` has no axes, no legend, no chart title: the numbers are in the
      squares and the caption says what the object is.
    - **William Cleveland & Robert McGill** (1984, JASA) — measured, experimentally, which
      encodings people decode accurately: position on a common scale first, then length, angle,
      area, and color last. Why the digits sit *in* the squares: color reinforces; it is never
      the only channel.
    - **Colin Ware**, *Information Visualization* (2000) — the perception textbook: fine spatial
      detail rides the luminance channel, so text and shape contrast must be luminance contrast.
      The ink-crossover machinery in `show()` is this, made executable.
    - **Cynthia Brewer** — ColorBrewer (2003): palettes as engineered instruments with declared
      roles — sequential, diverging, qualitative. The taxonomy this page is organized by.
    - **Leland Wilkinson**, *The Grammar of Graphics* (1999) → **Hadley Wickham** (ggplot2) →
      **Jeffrey Heer & Mike Bostock** (D3, then Vega/Vega-Lite) — the declarative lineage that
      Altair, and therefore every chart in these notebooks, compiles into.
    - **Masataka Okabe & Kei Ito** (2002) — the color-universal-design palette used for every
      categorical hue here.
    - **Stéfan van der Walt & Nathaniel Smith** — viridis (2015): perceptual uniformity as a
      constructed property, ending matplotlib's rainbow default.
    - **Peter Kovesi** (2015) — the test images and metrics that make "perceptually uniform" a
      measurable claim rather than a compliment.
    - **Jamie Nuñez, Christopher Anderton & Ryan Renslow** (2018, PLOS ONE) — cividis: the
      uniform map re-derived so color-vision-deficient and typical readers see the same thing.
    - **Fabio Crameri** — the Scientific Colour Maps (batlow and family); with Grace Shephard
      and Philip Heron, *The misuse of colour in science communication* (Nature Communications,
      2020), and the 2024 *Current Protocols* guide that is the field's current how-to. Their
      test battery — uniformity, CVD, grayscale — is the one this page runs.
    - **Tamara Munzner**, *Visualization Analysis & Design* (2014) — the channel-effectiveness
      synthesis: identity → hue, magnitude → lightness/position. The one-sentence version of
      this entire page.

    (The interaction side of these notebooks — prediction before reveal, guided exploration —
    has its own canon and its own file: Victor and Case, in pedagogy.md.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verdict: is the house theme the best available?

    What the columns above establish, honestly stated:

    - **Okabe-Ito for categories** is the world-class consensus, full stop. Keep.
    - **Blue-orange for divergence** is one of the two recommended hue pairs (the other being
      purple-orange). `POLARITY`'s lightness is monotonic per arm and its arms survive
      deuteranopia. Keep.
    - **The sequential `RAMP` is the genuine judgment call.** As a single-hue blue ramp it
      passes all three tests trivially — but it is not perceptually uniform in the measured
      CAM02 sense, and it spends less contrast per step than cividis or batlow, so two nearby
      values sit closer together than they need to. The counterweight, per Cleveland & McGill:
      in `show()`, color is never the only channel — every value is printed in its square — so
      uniformity matters less here than in a dense, legend-read heatmap. Where an exhibit ever
      drops the digits (large tensors, loss surfaces), the standards-grade move is cividis or
      batlow, not a prettier hand ramp.

    A reasonable policy, should you want it: RAMP stays for digit-bearing grids where color is
    reinforcement; cividis (already `_viz.py`'s named scheme for charts) for anything read by
    color alone; batlow via `cmcrameri` if a future exhibit wants the current geoscience
    reference. That is a proposal — the trade between consistency and uniformity is yours to
    adjudicate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try the alternatives on real squares

    The gallery shows swatches; this shows a tensor. Same data, your choice of ramp — the
    digits stay, so what changes is exactly the reinforcement layer the verdict above weighs.

    The dropdown is ordered as a ranking for *this* use — digit-bearing grids, read on both
    Horizon pages — by **your measured observer** (`_observer.py`, fit from your odd-one-out
    responses), not by a population simulation: each ramp is scored by its worst adjacent-pair
    discriminability, the minimum over both grounds of the fitted probability that you tell
    neighboring steps apart at exhibit scale. (An earlier ordering here leaned on an assumed
    mild red-green deficiency; the fitted observer shows no deficiency signal, so the
    measurement retired that assumption.) The two luminance-non-monotonic rainbows stay last
    regardless of score — luminance monotonicity is a house rule, and their adjacent-step
    contrast is exactly the property that makes them lie about magnitude — present for
    comparison, not candidacy.
    """)
    return


@app.cell(hide_code=True)
def _(RAMP, cmc, luminance, mo, sample_cmap):
    from pathlib import Path as _P

    from _observer import discriminability as _disc
    from _observer import fit as _obs_fit

    RAMPS = {
        "cividis": sample_cmap("cividis", 5),
        "batlow (Crameri)": sample_cmap(cmc.batlow, 5),
        "viridis": sample_cmap("viridis", 5),
        "RAMP (house)": RAMP,
        "YlGnBu (Brewer)": sample_cmap("YlGnBu", 5),
        "magma": sample_cmap("magma", 5),
        "plasma": sample_cmap("plasma", 5),
        "inferno": sample_cmap("inferno", 5),
        "Blues (Brewer)": sample_cmap("Blues", 5),
        "cubehelix": sample_cmap("cubehelix", 5),
        "turbo": sample_cmap("turbo", 5),
        "jet": sample_cmap("jet", 5),
    }
    # Measured ranking: worst adjacent-pair discriminability under the fitted observer,
    # minimum over both Horizon grounds — replacing the earlier hand order that leaned on
    # an assumed (and since falsified) red-green deficiency. Luminance-non-monotonic
    # rainbows sort last regardless: monotonicity is a house rule the score cannot buy back.
    _log = _P(__file__).parent / "calibration-responses.jsonl"

    def _monotone(hexes):
        from itertools import pairwise as _pw

        _l = [luminance(_h) for _h in hexes]
        return all(_a < _b for _a, _b in _pw(_l)) or all(_a > _b for _a, _b in _pw(_l))

    if _log.exists():
        _fit = _obs_fit(_log)

        def _score(hexes):
            from itertools import pairwise as _pw2

            return min(_disc(_fit, _a, _b, _g) for _a, _b in _pw2(hexes) for _g in ("#fdf0ed", "#1c1e26"))

        _ranked = sorted(RAMPS, key=lambda k: (not _monotone(RAMPS[k]), -_score(RAMPS[k]), k))
        _label = f"sequential ramp, ranked by your measured eyes ({_fit.n} trials)"
    else:
        _ranked = list(RAMPS)
        _label = "sequential ramp, ranked (no calibration log on this machine: hand order)"
    ramp_choice = mo.ui.dropdown(options=_ranked, value="RAMP (house)", label=_label)
    ramp_choice
    return RAMPS, ramp_choice


@app.cell(hide_code=True)
def _(INK_DARK, INK_LIGHT, RAMPS, alt, luminance, mo, pd, ramp_choice, torch):
    _ramp = RAMPS[ramp_choice.value]
    _values = torch.arange(48).reshape(6, 8)
    _limit = float(_values.max())
    _frame = pd.DataFrame(
        [{"row": i, "col": j, "v": int(v)} for i, row in enumerate(_values.tolist()) for j, v in enumerate(row)]
    )
    # Ink flips where the chosen ramp's own luminance drops below mid-gray, computed per ramp
    # rather than hard-coded, since each candidate darkens at a different point.
    _dark_from = next((k / (len(_ramp) - 1) for k in range(len(_ramp)) if luminance(_ramp[k]) < 0.2), 1.1)
    _at = {
        "x": alt.X("col:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
        "y": alt.Y("row:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
    }
    mo.ui.altair_chart(
        (
            alt.Chart(_frame)
            .mark_rect()
            .encode(
                **_at,
                color=alt.Color("v:Q", scale=alt.Scale(range=_ramp, domain=[0, _limit]), legend=None),
                tooltip=[alt.Tooltip("v:Q", title="value")],
            )
            + alt.Chart(_frame)
            .mark_text(fontSize=13, fontWeight=500)
            .encode(
                **_at,
                text=alt.Text("v:Q"),
                color=alt.condition(f"datum.v > {_dark_from * _limit}", alt.value(INK_LIGHT), alt.value(INK_DARK)),
            )
        ).properties(width=8 * 46, height=6 * 46),
        chart_selection=False,
        legend_selection=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go next

    - Crameri's [Scientific Colour Maps](https://www.fabiocrameri.ch/colourmaps/) and the 2024
      guide, [Choosing Suitable Color Palettes for Accessible and Accurate Science
      Figures](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.1126) — the
      field's current instructions.
    - [Crameri, Shephard & Heron 2020](https://www.nature.com/articles/s41467-020-19160-7),
      *The misuse of colour in science communication* — why this notebook exists.
    - [The cividis paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239)
      (Nuñez, Anderton & Renslow 2018) — a colormap derived from a stated objective function.
    - [ColorBrewer](https://colorbrewer2.org) — Brewer's original instrument, still the fastest
      way to feel the sequential/diverging/qualitative taxonomy.
    - Munzner's *Visualization Analysis & Design* — the textbook behind "identity → hue,
      magnitude → lightness."
    """)
    return


if __name__ == "__main__":
    app.run()
