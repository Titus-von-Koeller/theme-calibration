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
    *A sidecar to the theme gallery — the instrument that replaces its simulations with you.*

    # Calibrating the palettes against your eyes

    The gallery's deuteranopia column is a population model; what decides legibility is what
    *your* eyes distinguish on *this* screen, in the theme and light you actually read in. This
    notebook measures that directly: each trial shows four squares on one of seven candidate
    page grounds — the two Horizon pages and the best-in-class field (Selenized, Modus, GitHub
    dark) — at one of three patch sizes, from the exhibit scale (104 px) down to editor-token
    scale (10 px), where color discrimination is known to collapse and no large-patch result
    can be trusted. Three squares share one color, one differs; both are drawn from the
    palettes under evaluation (the editor theme's own accents included). Click the odd one
    out. Chance is 25%; a pair you can no longer beat chance on is, for you, at that size, on
    that page, one color.

    What is being optimized is not your score but your **observer model** (v2, shared with
    every other instrument through `_observer.py`): thresholds live in CAM16-UCS — the same
    space the aesthetics search runs in — with a fitted psychometric slope, a fitted lapse, a
    chromatic weight ellipse whose orientation is free to find any red–green confusion axis,
    threshold as a *smooth function of ground lightness* (so the fit generalizes to pages
    never shown), and a small-field exponent that the 10–16 px trials identify. Each trial is
    *generated* to be maximally informative about that model, which parks it near your
    ~75%-correct zone — **feeling hard means it is working**, and every answer moves the whole
    surface, not one pair's tally. A fraction of trials stay easy palette pairs, as anchors
    and breathers.

    Protocol: glance, decide within about a second, click — hesitation measures reasoning, not
    perception. Sixty-plus trials make a sitting; every response appends to
    `calibration-responses.jsonl` beside this file, so sittings accumulate across days, themes,
    and ambient light. Grounds and sizes run in sixteen-trial blocks (adaptation is part of the
    measurement). The screen itself is uncalibrated for now (parked in the queue) — that
    limits absolute claims, not relative ones: which pairs and which palettes fail *you* on
    *this* screen is exactly what accumulates below.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import random
    import re
    from datetime import datetime, timezone
    from pathlib import Path

    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    from _palette import OKABE_ITO, POLARITY, RAMP
    from cmcrameri import cm as cmc

    def _hexes(cmap, n=7):
        cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        if hasattr(cmap, "colors") and len(cmap.colors) < 30:
            picks = list(cmap.colors)[:n]
        else:
            picks = [cmap(i / (n - 1)) for i in range(n)]
        return ["#" + "".join(f"{round(255 * v):02x}" for v in c[:3]) for c in picks]

    def _horizon_accents():
        """The editor theme's own accents, alpha composited onto their page."""
        found = {}
        for label, name in (("horizon-day", "horizon-bright-bold.json"), ("horizon-night", "horizon-bold.json")):
            candidates = list(Path.home().glob(f".vscode/extensions/*horizon*/themes/{name}"))
            if not candidates:
                continue
            theme = json.loads(re.sub(r"//[^\n\"]*$", "", candidates[0].read_text(), flags=re.M))
            page = theme["colors"]["editor.background"]
            accents = [theme["colors"].get("textLink.foreground")]
            wanted = ["keyword", "string", "variable", "entity.name.function", "comment"]
            for entry in theme.get("tokenColors", []):
                scopes = entry.get("scope", [])
                scopes = [scopes] if isinstance(scopes, str) else scopes
                color = entry.get("settings", {}).get("foreground")
                if color and any(s == w or s.startswith(w) for w in list(wanted) for s in scopes):
                    accents.append(color)
                    wanted = [w for w in wanted if not any(s == w or s.startswith(w) for s in scopes)]
            bg = [int(page.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]

            def _flat(c, bg=bg):
                raw = c.lstrip("#")
                a = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1.0
                rgb = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
                return "#" + "".join(f"{round(a * v + (1 - a) * b):02x}" for v, b in zip(rgb, bg, strict=True))

            found[label] = [_flat(c) for c in accents if c]
        return found

    PALETTES = {
        "RAMP": RAMP,
        "POLARITY": POLARITY,
        "okabe-ito": list(OKABE_ITO.values())[:8],
        "cividis": _hexes("cividis"),
        "batlow": _hexes(cmc.batlow),
        "viridis": _hexes("viridis"),
        "tab10": _hexes("tab10", 8),
        "Set1": _hexes("Set1", 8),
        **_horizon_accents(),
    }

    # Every within-palette pair is a candidate trial; neighbors in sequential ramps measure
    # exactly the local contrast the gallery could only assert.
    PAIRS = [
        (name, a, b) for name, hexes in PALETTES.items() for i, a in enumerate(hexes) for b in hexes[i + 1 :] if a != b
    ]

    # The ground family spans the theme program's whole candidate field in lightness (and,
    # weakly for now, warmth): threshold is modeled as a smooth function of ground, so every
    # page here sharpens the prediction for pages never shown. Sources: the two Horizon
    # pages; Selenized light/dark (jan-warchol/selenized, the-values.md); Modus operandi/
    # vivendi (pure white/black by design); GitHub dark default.
    GROUND_LIST = [
        ("horizon-day", "#fdf0ed"),
        ("horizon-night", "#1c1e26"),
        ("selenized-light", "#fbf3db"),
        ("selenized-dark", "#103c48"),
        ("modus-light", "#ffffff"),
        ("modus-dark", "#000000"),
        ("github-dark", "#0d1117"),
    ]
    # Patch sizes: exhibit scale, and the glyph scale that actually decides the editor theme.
    # Gap scales with size (near-abutting, like adjacent glyphs); both are logged per trial.
    SIZES = (104, 16, 10)
    GAPS = {104: 12, 16: 2, 10: 1}
    LOG = Path(__file__).parent / "calibration-responses.jsonl"

    return GAPS, GROUND_LIST, LOG, PAIRS, SIZES, datetime, json, np, pd, random, timezone


@app.cell(hide_code=True)
def _(LOG, json, mo):
    _existing = [json.loads(_line) for _line in LOG.read_text().splitlines() if _line.strip()] if LOG.exists() else []
    get_responses, set_responses = mo.state(_existing)
    return get_responses, set_responses


@app.cell(hide_code=True)
def _(GAPS, GROUND_LIST, LOG, PAIRS, SIZES, np, random):
    # The observer model lives in _observer.py — one model, one fit, shared with the
    # aesthetics instrument so constraints never fork from measurements. This cell only
    # maintains a live posterior over its parameter grid and generates informative trials.
    import _observer as obs

    _COLS = obs.grid_columns()
    _N_CELLS = len(_COLS[0])
    # Dense log-posterior, bootstrapped from a binary sidecar so a fresh kernel does not
    # pay the full ~35 s refit; per-response updates are incremental (~60 ms) and equal to
    # a from-scratch fit up to float accumulation noise (~1e-4 log units, measured).
    _LOGP_NPY = LOG.parent / f"observer-logp-{obs.MODEL_VERSION}.npy"
    _LOGP_META = LOG.parent / f"observer-logp-{obs.MODEL_VERSION}.json"
    _PSTATE = {"n": -1, "logp": None}

    def _posterior_logp(responses):
        import json as _json

        if _PSTATE["n"] == len(responses):
            return _PSTATE["logp"]
        if _PSTATE["logp"] is None:
            _logp, _base = None, 0
            if _LOGP_NPY.exists() and _LOGP_META.exists():
                _meta = _json.loads(_LOGP_META.read_text())
                if _meta.get("n", 0) <= len(responses) and _meta.get("cells") == _N_CELLS:
                    _logp, _base = np.load(_LOGP_NPY), _meta["n"]
            if _logp is None:
                _logp, _base = np.zeros(_N_CELLS), 0
            if responses[_base:]:
                obs.add_loglik(_logp, responses[_base:], chunk=40_000)
            np.save(_LOGP_NPY, _logp)
            _LOGP_META.write_text(_json.dumps({"n": len(responses), "cells": _N_CELLS}))
        elif _PSTATE["n"] < len(responses):
            _logp = obs.add_loglik(_PSTATE["logp"], responses[_PSTATE["n"] :])
        else:  # log shrank (external edit): refit from scratch
            _logp = obs.add_loglik(np.zeros(_N_CELLS), responses, chunk=40_000)
        _PSTATE["n"], _PSTATE["logp"] = len(responses), _logp
        return _logp

    def posterior_for(responses):
        """(condensed posterior, condensed grid columns) — the trial generator's view."""
        _logp = _posterior_logp(responses)
        if len(responses) < 16:
            # A near-flat posterior makes top-k selection an arbitrary corner of the grid;
            # a strided subset spans it evenly until the data has an opinion.
            _idx = np.arange(0, _N_CELLS, max(1, _N_CELLS // 25_000))
            _p = np.exp(_logp[_idx] - _logp[_idx].max())
            return _p / _p.sum(), [c[_idx] for c in _COLS]
        return obs.condense(_logp, _COLS)

    def full_posterior(responses):
        """The full-grid normalized posterior — the analysis cell's view."""
        _logp = _posterior_logp(responses)
        _p = np.exp(_logp - _logp.max())
        return _p / _p.sum()

    def _entropy(q):
        return -(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12))

    # Deterministic given the log; the memo only spares two sibling cells the recompute.
    _TRIAL_MEMO = {}

    def trial_for(n, responses):
        """The nth trial, generated to maximize expected information about the observer.

        Grounds and sizes run in 16-trial blocks (adaptation stays part of the measurement;
        blocks cycle all grounds, then rotate patch size). Within a block, candidate stimuli
        are built from a palette color plus offsets along the CAM16-UCS axes, the diagonals,
        and the current confusion-axis estimate, at magnitudes swept coarse-then-fine; the
        winner maximizes mutual information between the response and the posterior. 5% stay
        plain palette pairs as anchors against model misspecification."""
        if n in _TRIAL_MEMO:
            return _TRIAL_MEMO[n]
        _rng = random.Random(n * 2654435761 % (2**31))
        _b = n // 16
        _glabel, _ghex = GROUND_LIST[_b % len(GROUND_LIST)]
        _size = SIZES[(_b // len(GROUND_LIST)) % len(SIZES)]
        _pal, _a, _b2 = _rng.choice(PAIRS)
        if _rng.random() < 0.05:
            if _rng.random() < 0.5:
                _a, _b2 = _b2, _a
            _base, _odd, _kind = _a, _b2, _pal
        else:
            _post, _cols = posterior_for(responses[:n])
            _base_u = obs.hex_to_ucs(_a)[0]
            _gj = np.array([obs.hex_to_ucs(_ghex)[0, 0] / 100.0], dtype=np.float32)
            _sz = np.array([float(_size)], dtype=np.float32)
            _s2 = 1 / np.sqrt(2)
            _phi_hat = float((_post * _cols[0]).sum())
            _rad = np.radians(_phi_hat)
            _dirs = [
                np.array(_v)
                for _v in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                    (0, _s2, _s2),
                    (0, -_s2, _s2),
                    (0, np.cos(_rad), np.sin(_rad)),
                    (0, -np.cos(_rad), np.sin(_rad)),
                ]
            ]

            def _eig_batch(_dv, _mags):
                """(best_eig, best_mag, best_hex) along one direction at several magnitudes."""
                _cands = np.array([_base_u + _dv * _m for _m in _mags])
                _hexes = obs.ucs_to_hex(_cands)
                _du = (_base_u[None, :] - obs.hex_to_ucs(_hexes)).astype(np.float32)
                _p = obs.p_correct_cells(_cols, _du, np.repeat(_gj, len(_mags)), np.repeat(_sz, len(_mags)))
                _pbar = _post @ _p
                _eig = _entropy(_pbar) - _post @ _entropy(_p)
                for _i, _h in enumerate(_hexes):
                    if _h == _a:
                        _eig[_i] = -1.0
                _j = int(np.argmax(_eig))
                return float(_eig[_j]), float(_mags[_j]), _hexes[_j]

            # Two-stage magnitude search per direction (a coarse-only grid measured ~28%
            # information lost when the threshold falls between its steps).
            _best, _best_hex = -1.0, None
            for _dv in _dirs:
                _e1, _m1, _h1 = _eig_batch(_dv, np.geomspace(0.3, 90.0, 7))
                if _e1 > _best:
                    _best, _best_hex = _e1, _h1
                _e2, _, _h2 = _eig_batch(_dv, np.geomspace(_m1 / 2.5, min(_m1 * 2.5, 110.0), 8))
                if _e2 > _best:
                    _best, _best_hex = _e2, _h2
            _base, _odd, _kind = _a, _best_hex, "probe"
        _trial = {
            "palette": _kind,
            "base": _base,
            "odd_color": _odd,
            "ground": _glabel,
            "ground_hex": _ghex,
            "size_px": _size,
            "gap_px": GAPS[_size],
            "odd_position": _rng.randrange(4),
        }
        _TRIAL_MEMO[n] = _trial
        return _trial

    return full_posterior, obs, trial_for


@app.cell(hide_code=True)
def _(get_responses, mo, trial_for):
    # The trial number doubles as a staleness indicator: if it ever disagrees with the
    # squares below, the surface lagged and clicks are being dropped by the guard.
    _n = len(get_responses())
    _t = trial_for(_n, get_responses())
    mo.hstack(
        [mo.md(f"**Trial {_n + 1}** · {_t['ground']} · {_t['size_px']} px — click the odd square.")],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(get_responses, mo, trial_for):
    _n = len(get_responses())
    _t = trial_for(_n, get_responses())
    _colors = [_t["base"]] * 4
    _colors[_t["odd_position"]] = _t["odd_color"]

    # A real widget instead of styled buttons: the squares are plain clickable divs on one
    # ground, so nothing of a button's chrome shows. anywidget syncs the click back as the
    # chosen index; a fresh widget renders per trial and the guard drops stale clicks.
    # At glyph scale the squares are near-unclickable targets, so the wrap resolves any
    # click to the nearest square within a generous radius — the click is the answer, not
    # the motor test.
    import anywidget
    import traitlets

    class _OddOneOut(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          el.style.cssText = `display:block;width:100%`;
          const wrap = document.createElement("div");
          wrap.style.cssText = `background:${model.get("ground")};padding:22px;` +
            `border-radius:10px;display:flex;justify-content:center;align-items:center;` +
            `gap:${model.get("gap")}px;width:100%;box-sizing:border-box;aspect-ratio:1.618/1`;
          const size = model.get("size");
          const squares = [];
          model.get("colors").forEach((c, i) => {
            const sq = document.createElement("div");
            // Fixed pixels on purpose: patch size AND separation are stimulus parameters
            // (spatial summation; near-abutting fields give the most sensitive
            // comparison, and match how adjacent glyphs and chart marks are read).
            sq.style.cssText = `width:${size}px;height:${size}px;` +
              `border-radius:${Math.max(1, Math.round(size / 10))}px;background:${c}`;
            squares.push(sq);
            wrap.appendChild(sq);
          });
          wrap.style.cursor = "pointer";
          wrap.onclick = (ev) => {
            let best = -1, bestD = Infinity;
            squares.forEach((sq, i) => {
              const r = sq.getBoundingClientRect();
              const dx = ev.clientX - (r.x + r.width / 2), dy = ev.clientY - (r.y + r.height / 2);
              const d = Math.hypot(dx, dy);
              if (d < bestD) { bestD = d; best = i; }
            });
            if (bestD > Math.max(size, 30)) return;  // a click far from every square is no answer
            model.set("clicks", model.get("clicks") + 1);
            model.set("choice", best);
            model.save_changes();
          };
          el.replaceChildren(wrap);
        }
        export default { render };
        """
        colors = traitlets.List([]).tag(sync=True)
        ground = traitlets.Unicode("#ffffff").tag(sync=True)
        size = traitlets.Int(104).tag(sync=True)
        gap = traitlets.Int(12).tag(sync=True)
        choice = traitlets.Int(-1).tag(sync=True)
        clicks = traitlets.Int(0).tag(sync=True)

    answer_squares = mo.ui.anywidget(
        _OddOneOut(colors=_colors, ground=_t["ground_hex"], size=_t["size_px"], gap=_t["gap_px"])
    )
    answer_squares
    return (answer_squares,)


@app.cell(hide_code=True)
def _(LOG, answer_squares, datetime, get_responses, json, set_responses, timezone, trial_for):
    # Recording watches the widget's synced traits. Only the FIRST click of a fresh widget
    # counts (clicks == 1): later clicks on the same trial, and clicks on an orphaned stale
    # widget, record nothing — the guard below double-checks against the response count.
    _n = len(get_responses())

    def _record(choice, n=_n):
        # The squares are the buttons, so they re-render per trial — which reintroduces the
        # stale-surface risk. The guard converts it from data corruption into a dropped
        # click: a rendering whose trial is no longer current records nothing.
        if n != len(get_responses()):
            return
        _now = trial_for(n, get_responses())
        _entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "palette": _now["palette"],
            "base": _now["base"],
            "odd_color": _now["odd_color"],
            "ground": _now["ground"],
            "ground_hex": _now["ground_hex"],
            "odd_position": _now["odd_position"],
            "choice": choice,
            "correct": choice == _now["odd_position"],
            # Size, gap and ground are stimulus parameters; the observer model fits
            # threshold as a function of them, so they ride with every response.
            "size_px": _now["size_px"],
            "gap_px": _now["gap_px"],
        }
        # Append-only, one record per line: concurrent sessions interleave instead of
        # overwriting each other's history.
        with LOG.open("a") as _f:
            _f.write(json.dumps(_entry) + "\n")
        set_responses([*get_responses(), _entry])

    _v = answer_squares.value
    if _v.get("clicks") == 1 and _v.get("choice", -1) >= 0:
        _record(_v["choice"])
    return


@app.cell(hide_code=True)
def _(full_posterior, get_responses, mo, np, obs, pd):
    _log = get_responses()
    if not _log:
        _out = mo.md("*No responses yet — the analysis fills in as you answer.*")
    else:
        _frame = pd.DataFrame(_log)
        _frame["pair"] = _frame.apply(lambda r: " / ".join(sorted([r.base, r.odd_color])), axis=1)
        _acc = _frame.correct.mean()
        _by_pair = (
            _frame.groupby(["palette", "pair"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
        )
        _tested = _by_pair[_by_pair.n >= 3].sort_values("accuracy")
        _by_palette = (
            _frame.groupby("palette")
            .agg(trials=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
            .sort_values("accuracy")
        )
        _by_ground = _frame.groupby("ground").correct.mean()
        _post = full_posterior(_log)
        _cols = obs.grid_columns()
        _dirs = {
            "lightness": np.array([1.0, 0, 0]),
            "red–green (a')": np.array([0.0, 1, 0]),
            "blue–yellow (b')": np.array([0.0, 0, 1]),
        }
        _gj = {"day": 0.966, "night": 0.147}  # Horizon pages, J'/100
        _tiles = [
            mo.stat(
                f"{obs.threshold_de(_post, _cols, _v, _gj[_g]):.2f}",
                label=f"{_name} · {_g} (ΔE)",
                bordered=True,
            )
            for _name, _v in _dirs.items()
            for _g in ("day", "night")
        ]
        _beta = float((obs.marginal(_post, "beta")[1] * obs.marginal(_post, "beta")[0]).sum())
        _lam = float((obs.marginal(_post, "lam")[1] * obs.marginal(_post, "lam")[0]).sum())
        _phi_v, _phi_p = obs.marginal(_post, "phi")
        _phi = float((_phi_p * _phi_v).sum())
        _phi_sd = float(np.sqrt(max((_phi_p * _phi_v**2).sum() - _phi**2, 0.0)))
        _gl = float((obs.marginal(_post, "gL")[1] * obs.marginal(_post, "gL")[0]).sum())
        _gam_v, _gam_p = obs.marginal(_post, "gamma")
        _gam_sd = float(np.sqrt(max((_gam_p * _gam_v**2).sum() - ((_gam_p * _gam_v).sum()) ** 2, 0.0)))
        _model_tiles = [
            mo.stat(f"{_beta:.2f}", label="psychometric slope β", bordered=True),
            mo.stat(f"{100 * _lam:.1f}%", label="your fitted slip rate", bordered=True),
            mo.stat(f"{_phi:.0f}° ± {_phi_sd:.0f}°", label="confusion-axis angle", bordered=True),
            mo.stat(f"{np.exp(_gl * (0.147 - 0.966)):.2f}×", label="dark-page threshold ratio", bordered=True),
            mo.stat(
                f"{(_gam_p * _gam_v).sum():.2f} ± {_gam_sd:.2f}",
                label="small-field exponent γ (needs glyph trials)",
                bordered=True,
            ),
        ]
        _out = mo.vstack(
            [
                mo.md(
                    "**What the model has learned about your eyes** — CAM16-UCS distance per "
                    "direction and Horizon ground at which you reach 75% correct, at 104 px "
                    "(smaller is finer discrimination):"
                ),
                mo.hstack(_tiles, justify="start", gap=1),
                mo.hstack(_model_tiles, justify="start", gap=1),
                mo.hstack(
                    [
                        mo.stat(f"{len(_frame):,}", label="responses", bordered=True),
                        mo.stat(f"{100 * _acc:.0f}%", label="overall accuracy (chance 25%)", bordered=True),
                        *[mo.stat(f"{100 * v:.0f}%", label=f"on {g}", bordered=True) for g, v in _by_ground.items()],
                    ],
                    justify="start",
                    gap=1,
                ),
                mo.md("**Palettes, hardest first for your eyes** (accuracy over all their tested pairs):"),
                mo.ui.table(_by_palette, selection=None),
                mo.md("**Most confused pairs so far** (at least three trials each):"),
                mo.ui.table(_tested.head(12), selection=None) if len(_tested) else mo.md("*none with n ≥ 3 yet*"),
            ],
            gap=0.8,
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the numbers, and what happens to them

    A pair at 25% is invisible to you; at 100% it is trivially yours; sequential ramps live or
    die by their *adjacent* pairs, categorical palettes by their worst pair anywhere. The tiles
    report 75%-correct thresholds as CAM16-UCS distances (ΔE) — one perceptual currency for
    every direction, ground, and instrument in the theme program, fit by the shared observer
    model in `_observer.py` (v2: fitted slope and lapse, free confusion-axis orientation,
    threshold smooth in ground lightness, small-field exponent). The v1 findings survive the
    re-derivation at 748 trials, most of them sharpened:

    - **Slip rate ~0.6%** — unchanged; your motor errors are negligible.
    - **Dark pages read finer, ~0.76× the light-page threshold** — v1's 25–30% night
      advantage, re-derived as a smooth lightness slope (gL ≈ 0.33) that now predicts
      *every* candidate ground, not just the two Horizon pages.
    - **Thresholds at 104 px, Horizon day / night**: lightness 3.2 / 2.5 ΔE, blue–yellow
      3.6 / 2.8, red–green 5.3 / 4.1. v1's per-axis numbers were conditioned on an assumed
      psychometric slope of 2; the fitted slope is **β ≈ 1.2** (shallower — misses fade in
      gradually rather than cliff), so these are the better-calibrated figures.
    - **Are you colorblind? The data says no.** The model lets a red–green confusion axis
      emerge freely (orientation φ and a weight ellipse in the chromatic plane). What it
      finds: your weakest chromatic direction sits at φ ≈ 1° ± 20° — too uncertain to even
      call an axis — and your red–green threshold is 1.5× your blue–yellow one. Anomalous
      trichromats show *several-fold* red–green elevation in comparable discrimination
      tasks (Boehm, MacLeod & Bosten 2014, JOV; Bosten 2021, Vision Research reviews the
      2–10× range), and a deficiency would pin φ near the protan/deutan line with a
      strongly depressed weight — your weight ratio is a mild 0.39 with an unconstrained
      angle. Mild red–green coarseness at this magnitude is within the normal range for a
      4AFC patch task on an uncalibrated display. Every candidate palette's worst pair
      still sits at the lapse-limited ceiling at 104 px on both Horizon grounds.
    - **What 104 px does not settle, this instrument now measures**: blocks cycle three
      patch sizes (104/16/10 px) and seven grounds (Horizon, Selenized, Modus, GitHub
      dark). The small-field exponent γ starts flat by design and tightens as 10–16 px
      trials accumulate — that number, not the exhibit-scale one, decides evolve-vs-switch
      for the editor theme. Ground *warmth* stays out of the model until the ground family
      decouples it from lightness; the hexes ride with every response either way.

    A note on how trials *feel*: an information-optimal trial sits near your threshold, so
    most should look nearly indistinguishable — a run of "I'm mostly guessing" is the
    instrument working, not failing; the model expects and absorbs those misses. At 10 px
    everything will feel hard: that is the point. Trials accumulate in
    `calibration-responses.jsonl`, committed like any measurement; the fit is cached beside
    it (`observer-fit.json`, plus an uncommitted binary posterior sidecar) and every
    instrument — the aesthetics duels' hard constraints included — reads the same fit, so a
    sitting here immediately retunes what the preference optimizer is allowed to show you.
    """)
    return


if __name__ == "__main__":
    app.run()
