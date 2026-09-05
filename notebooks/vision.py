# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes -- correct for a notebook holding a model on a GPU, and fatal
# for a trial loop, whose whole point is that the next stimulus appears on the answer.
# Script metadata is merged over the project config at the highest precedence, so a
# notebook opts in on its own.
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
    *The measurement the rest of the instrument stands on. Every contrast floor and every
    pairwise colour separation the theme search enforces is a number fitted here.*

    # Calibrating the palette against your eyes

    A population model of colour vision predicts what an average observer distinguishes.
    What decides whether a theme is legible is what *your* eyes distinguish on *this*
    screen, in the light you actually read in. This notebook measures that directly.

    Each trial shows four squares on one of seven candidate page grounds -- the two Horizon
    pages and a best-in-class field (Selenized, Modus, GitHub dark) -- at one of three patch
    sizes, from exhibit scale (104 px) down to editor-token scale (10 px), where colour
    discrimination is known to collapse and no large-patch result can be trusted. Three
    squares share one colour and one differs. Press the key under the odd one out, or click
    its button. Chance is 25%; a pair you can no longer beat chance on is, for you, at that
    size, on that page, one colour.

    What is being optimised is not your score but your **observer model**, which lives in
    `theme/observer.py` and is shared with every other instrument in this repo. Thresholds
    live in CAM16-UCS -- the same space the aesthetics search runs in -- with a fitted
    psychometric slope, a fitted lapse, a chromatic weight ellipse whose orientation is free
    to find any red-green confusion axis, threshold as a *smooth function of ground
    lightness* (so the fit generalises to pages never shown), and a small-field exponent
    that the 10-16 px trials identify. Each probe trial is *generated* to maximise expected
    information about that model, so every answer moves the whole surface rather than one
    pair's tally. A declared 5% of trials stay easy palette pairs, as anchors against model
    misspecification -- and they are counted separately below, because an easy trial carries
    almost no threshold information and pooling it into an accuracy figure flatters the
    number without sharpening the fit.

    Protocol: glance, decide, answer. Sixty-plus trials make a sitting; every response
    appends to `data/calibration-responses.jsonl`, so sittings accumulate across days,
    themes and ambient light. Grounds and sizes run in sixteen-trial blocks, because
    adaptation to a page is part of what is being measured rather than noise in it.

    **Where the clock lives, and why it is not here.** The same trials are served with a
    reaction-time clock by the web app (`pixi run serve`), as the fourth arm of every
    32-trial block: eight colour trials after the find hunts, at the glyph sizes only,
    stamped generator `v4`, surface `app`, with `rt_ms` and the input method on every row.
    A notebook cannot give that clock honestly: it rebuilds its widgets between answers, and
    a rebuild between reveal and click lands directly on the reaction time. This loop
    therefore stays clockless -- it scores accuracy only, so a rebuild costs a frame and not
    a measurement -- and its rows carry no `rt_ms` rather than a fake one. Both surfaces
    append to the same log in one numbering, so a sitting here and a sitting there are one
    series; the input asymmetry that corrupts a time cannot corrupt an accuracy, but it can
    still bias a *guess*, so the answer keys are equidistant on both surfaces and the
    diagnostics below test whether a positional preference survived anyway.
    """)
    return


@app.cell(hide_code=True)
def _():
    import random

    import numpy as np
    import pandas as pd
    from scipy import stats

    try:
        import theme  # noqa: F401  -- an environment check, not a use; see below
    except ModuleNotFoundError as no_package:
        # theme-calibration is installed editable into this project's pixi environment and
        # into no other, so a missing `theme` means this notebook is running on the wrong
        # interpreter -- an editor kernel resolved for a parent folder is the way it
        # happens. Left unguarded that surfaces as NameError on every cell downstream of
        # this one, which names the symptom in a dozen places and the cause in none.
        raise ModuleNotFoundError(
            "vision.py is running on an interpreter that does not have theme-calibration "
            "installed, so nothing below this cell can run. Start it with `pixi run vision`, or "
            "select .pixi/envs/default/bin/python as this folder's interpreter. "
            "README, 'Running it', has both."
        ) from no_package

    from theme import observer as obs
    from theme import vision
    from theme.paths import VISION_LOG
    from theme.responses import ResponseLog
    from theme.vision import (
        ANCHOR_SHARE,
        BLOCK,
        GROUND_LIST,
        NOTEBOOK_GENERATOR,
        PALETTES,
        POSTERIOR,
        SIZES,
    )

    # The palettes, the ground family, the sizes and the generator itself live in
    # theme/vision.py, extracted from this notebook so the web app can serve the same trials
    # with a clock. This notebook keeps the reading half and a clockless trial loop; every
    # constant it quotes is the module's.
    LOG = ResponseLog(VISION_LOG)
    GENERATOR = NOTEBOOK_GENERATOR

    return (
        ANCHOR_SHARE,
        BLOCK,
        GENERATOR,
        GROUND_LIST,
        LOG,
        PALETTES,
        POSTERIOR,
        SIZES,
        np,
        obs,
        pd,
        random,
        stats,
        vision,
        VISION_LOG,
    )


@app.cell(hide_code=True)
def _(LOG, mo):
    _existing = LOG.read()
    get_responses, set_responses = mo.state(_existing)
    return get_responses, set_responses


@app.cell(hide_code=True)
def _(POSTERIOR, vision):
    # The observer model lives in theme/observer.py and its trial generator in
    # theme/vision.py -- one model, one fit, one generator, shared with the web app so the
    # trials a notebook sitting and an app sitting append to the same log are the same
    # trials. Bound here under the names the cells below have always used.
    def trial_for(n, responses):
        return vision.trial_for(n, responses)

    def dense_posterior(responses):
        return POSTERIOR.dense_for(responses)

    return dense_posterior, trial_for


@app.cell(hide_code=True)
def _(mo):
    # A begin gate, for the same reason the aesthetics runs have one and for one more.
    # Generating an information-optimal trial needs the dense posterior over 3.57M grid
    # cells, which costs minutes from cold; opening this notebook to READ the report should
    # not pay for a trial nobody is about to answer. Past the gate the sidecar is warm and
    # each subsequent trial costs tens of milliseconds.
    sitting = mo.ui.run_button(label="Begin a sitting")
    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    "### Take a sitting\n\nThe first trial builds the posterior from the whole "
                    "log and takes a few minutes; every trial after it is immediate. The report "
                    "below reads the cached fit and needs no sitting at all."
                ),
                sitting,
            ],
            gap=0.5,
        )
    )
    return (sitting,)


@app.cell(hide_code=True)
def _(get_responses, mo, sitting, trial_for):
    # The trial number doubles as a staleness indicator: if it ever disagrees with the
    # squares below, the surface lagged and answers are being dropped by the guard.
    if sitting.value:
        _n = len(get_responses())
        _t = trial_for(_n, get_responses())
        _head = mo.md(
            f"**Trial {_n + 1}** &middot; {_t['ground']} &middot; {_t['size_px']} px "
            f"&middot; {_t['kind']} --- answer the odd square."
        )
    else:
        _head = mo.md("")
    mo.output.replace(mo.hstack([_head], justify="center"))
    return


@app.cell(hide_code=True)
def _(get_responses, mo, sitting, trial_for):
    # Until the gate opens there is no stimulus and nothing downstream to record: mo.stop
    # halts this cell and every cell that depends on it, which is exactly the trial loop.
    mo.stop(not sitting.value, mo.md(""))
    _n = len(get_responses())
    _t = trial_for(_n, get_responses())
    _colors = [_t["base"]] * 4
    _colors[_t["odd_position"]] = _t["odd_color"]
    _size, _gap = _t["size_px"], _t["gap_px"]

    # Plain HTML on one ground, so nothing of a button's chrome shows anywhere near the
    # stimulus. This used to be an anywidget custom element; the widget is gone because the
    # dependency is gone from the environment, and with it the last thing in this notebook
    # that could not be rendered by marimo itself.
    #
    # Fixed pixels on purpose: patch size AND separation are stimulus parameters (spatial
    # summation; near-abutting fields give the most sensitive comparison, and match how
    # adjacent glyphs and chart marks are read).
    _squares = "".join(
        f'<div style="width:{_size}px;height:{_size}px;'
        f"border-radius:{max(1, round(_size / 10))}px;background:{_c}"
        '"></div>'
        for _c in _colors
    )
    _stage = mo.Html(
        f'<div style="background:{_t["ground_hex"]};padding:22px;border-radius:10px;'
        f"display:flex;justify-content:center;align-items:center;gap:{_gap}px;"
        f'width:100%;box-sizing:border-box;aspect-ratio:1.618/1">{_squares}</div>'
    )

    # Four keys, one per slot, equidistant from the hand. At glyph scale the squares
    # themselves are near-unclickable targets, and the old surface resolved a click to the
    # nearest square within a generous radius -- which turned every answer into a motor task
    # with a different cost per slot. The keys remove that: answering slot 1 and slot 4 take
    # the same effort. That matters even without a clock, because the cheapest slot is the
    # one a guess drifts toward, and a guess is exactly what a threshold trial elicits.
    answer_keys = mo.ui.array(
        [
            mo.ui.button(
                label=f"{_i + 1}",
                value=0,
                on_click=lambda _count: _count + 1,
                keyboard_shortcut=f"{_i + 1}",
                full_width=True,
            )
            for _i in range(4)
        ]
    )
    mo.output.replace(
        mo.vstack([_stage, mo.hstack(list(answer_keys), justify="center", gap=1, widths="equal")], gap=0.6)
    )
    return (answer_keys,)


@app.cell(hide_code=True)
def _(LOG, answer_keys, get_responses, set_responses, trial_for, vision):
    from datetime import UTC, datetime

    # Recording watches the four buttons' click counters. The array is rebuilt per trial, so
    # every counter starts at zero: exactly one counter at one is an answer, and anything
    # else -- a double press, a stale rendering answered twice -- records nothing. The guard
    # against the response count turns a lagging surface into a dropped answer rather than
    # into a row written onto somebody else's trial.
    _n = len(get_responses())
    _presses = list(answer_keys.value)

    def _record(choice, n=_n):
        if n != len(get_responses()):
            return
        # The row is built by the module the app records with, so a notebook sitting and
        # an app sitting write the same record; the notebook's rows simply carry no clock.
        _entry = vision.build_entry(
            n, trial_for(n, get_responses()), choice, datetime.now(UTC).isoformat(timespec="seconds")
        )
        LOG.append(_entry)
        set_responses([*get_responses(), _entry])

    if sum(1 for _p in _presses if _p) == 1 and max(_presses) == 1:
        _record(_presses.index(1))
    return


@app.cell(hide_code=True)
def _(LOG, VISION_LOG, mo, np, obs):
    # The fast readout: the cached fit, never a refit. theme.observer.fit stores its
    # per-axis posterior marginals in observer-fit.json alongside the point estimates, so
    # every number below arrives with the spread the data actually supports and the page
    # still opens in milliseconds. The joint quantities -- a threshold interval, the
    # red-green/blue-yellow ratio -- need the dense grid and are in the next cell, behind a
    # button, because that grid costs minutes.
    _n_log = len(LOG)

    def _q(values, probs, qs=(0.1, 0.5, 0.9)):
        """Weighted quantiles of one posterior marginal, by inverse CDF.

        The step definition and NOT linear interpolation of the CDF, for a reason that
        showed up as a 24% disagreement between two ways of computing the same interval.
        This posterior lives on a coarse grid -- gL has nine values, the lapse three -- so
        an interpolated quantile invents a value the model cannot represent, and worse, it
        is not consistent under a change of variable: interpolating the quantiles of
        exp(gL * dJ) gave 0.54 where transforming the quantiles of gL gave 0.61. The
        inverse CDF returns an actual grid value, so it commutes with any monotone
        transform and two routes to the same number agree exactly.

        The price is honest coarseness: an interval is quoted to grid resolution, which is
        what the data can actually distinguish.
        """
        _v = np.asarray(values, dtype=float)
        _o = np.argsort(_v)
        _c = np.cumsum(np.asarray(probs, dtype=float)[_o])
        _c = _c / _c[-1]
        return np.array([_v[_o][min(int(np.searchsorted(_c, _t, side="left")), len(_v) - 1)] for _t in qs])

    def _edge_note(values, probs, mass=0.1):
        """Warn when the posterior is pressed against an end of its own grid.

        A point estimate hides this; a distribution shows it. If a tenth of the mass sits on
        the first or last grid value, the true value may lie outside the grid and the
        reported spread is a property of the grid rather than of the data.
        """
        _p = np.asarray(probs, dtype=float)
        if _p[0] >= mass:
            return " (at the grid's low end -- may lie below it)"
        if _p[-1] >= mass:
            return " (at the grid's high end -- may lie above it)"
        return ""

    if _n_log == 0:
        _out = mo.md(f"*`{VISION_LOG.name}` is empty -- the readout fills in as you answer.*")
    else:
        _fit = obs.fit(VISION_LOG)
        _marg = _fit.marginals

        def _tile(axis, label, fmt="{:.2f}", scale=1.0):
            _v, _p = _marg[axis]["values"], _marg[axis]["p"]
            _lo, _mid, _hi = _q(_v, _p)
            return mo.stat(
                fmt.format(_mid * scale),
                label=label,
                caption=f"80% CI {fmt.format(_lo * scale)} to {fmt.format(_hi * scale)}" + _edge_note(_v, _p),
                bordered=True,
            )

        # Threshold ratio between the darkest and lightest measured page. A monotone
        # function of gL alone, so its own marginal gives an honest interval.
        _gl_v, _gl_p = _marg["gL"]["values"], _marg["gL"]["p"]
        _dj = float(obs.hex_to_ucs("#1c1e26")[0, 0] - obs.hex_to_ucs("#fdf0ed")[0, 0]) / 100.0
        _ratio = np.exp(np.asarray(_gl_v, dtype=float) * _dj)
        _rl, _rm, _rh = _q(_ratio, _gl_p)
        _de = [
            mo.stat(
                f"{_fit.de_dir_day[_ax]:.2f} / {_fit.de_dir_night[_ax]:.2f}",
                label=f"{_ax} (dE)",
                caption="Horizon day / night, 104 px",
                bordered=True,
            )
            for _ax in _fit.de_dir_day
        ]
        _model = [
            _tile("beta", "psychometric slope"),
            _tile("lam", "lapse rate", fmt="{:.1f}%", scale=100.0),
            _tile("phi", "confusion-axis angle", fmt="{:.0f} deg"),
            mo.stat(
                f"{_rm:.2f}x",
                label="dark/light threshold ratio",
                caption=f"80% CI {_rl:.2f}x to {_rh:.2f}x"
                + (" -- includes 1.00, i.e. no advantage" if _rh >= 1.0 else ""),
                bordered=True,
            ),
            _tile("gamma", "small-field exponent"),
        ]
        _out = mo.vstack(
            [
                mo.md(
                    f"**What the model has learned about your eyes**, from {_fit.n:,} trials. "
                    "The dE tiles are posterior-mean CAM16-UCS distances at which you reach "
                    "75% correct on each Horizon page at 104 px (smaller is finer "
                    "discrimination); every other tile carries the middle 80% of its "
                    "posterior, because a parameter reported as one number is a parameter "
                    "whose uncertainty someone else has to guess at."
                ),
                mo.hstack(_de, justify="start", gap=1),
                mo.hstack(_model, justify="start", gap=1),
            ],
            gap=0.8,
        )
    mo.output.replace(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    # Gated on purpose. The dense posterior is 3.57M grid cells over every response;
    # measured at 227 s from cold on a loaded machine, then cached to a binary sidecar
    # beside the log (gitignored -- it is derived data). A report that opens in four minutes
    # is a report nobody reads, so the page opens on the cached point fit above and computes
    # the joint intervals when asked.
    dense_button = mo.ui.run_button(label="Compute credible intervals (dense posterior; minutes from cold)")
    mo.output.replace(dense_button)
    return (dense_button,)


@app.cell(hide_code=True)
def _(dense_button, dense_posterior, get_responses, mo, np, obs):
    _log = get_responses()
    if not dense_button.value or not _log:
        _out = mo.md(
            "*The interval report and the colour-deficiency power check run on the dense "
            "posterior; press the button above.*"
        )
    else:
        _post, _cols = dense_posterior(_log)

        def _wq(values, weights, qs):
            """Inverse-CDF quantiles, for the reason spelled out in the fast readout above:
            every quantity here is a function of a coarse grid, so interpolating the CDF
            invents values the model cannot hold and stops commuting with the transform.
            The step definition makes these numbers and the marginal tiles agree exactly."""
            _v = np.asarray(values, dtype=float)
            _o = np.argsort(_v)
            _c = np.cumsum(np.asarray(weights, dtype=float)[_o])
            _c = _c / _c[-1]
            return np.array([_v[_o][min(int(np.searchsorted(_c, _t, side="left")), len(_v) - 1)] for _t in qs])

        # The threshold along one direction, per grid cell, read off a swept dE ladder
        # through theme.observer's own p_correct_cells rather than re-deriving its Weibull
        # inversion here. Two reasons: a second copy of the psychometric algebra is a second
        # place for it to drift from the model every other instrument uses, and the ladder
        # generalises to any p_target without new algebra. The condensed grid carries
        # 99.96% of the posterior mass (measured), so quantiles off it are exact to three
        # figures at a fraction of the memory.
        _ladder = np.geomspace(0.05, 80.0, 96)
        _k = min(25_000, len(_post))
        _top = np.argpartition(_post, -_k)[-_k:]
        _cpost = _post[_top] / _post[_top].sum()
        _ccols = [_c[_top] for _c in _cols]

        def _threshold_q(direction, g_j, size=104.0):
            _du = np.outer(_ladder, direction).astype(np.float32)
            _p = obs.p_correct_cells(
                _ccols,
                _du,
                np.full(len(_ladder), g_j, dtype=np.float32),
                np.full(len(_ladder), size, dtype=np.float32),
            )
            _reached = _p[:, -1] >= 0.75
            _de = _ladder[np.argmax(_p >= 0.75, axis=1)]
            if not _reached.any():
                return None
            return _wq(_de[_reached], _cpost[_reached], (0.1, 0.5, 0.9))

        _dirs = {
            "lightness": np.array([1.0, 0.0, 0.0]),
            "red-green (a')": np.array([0.0, 1.0, 0.0]),
            "blue-yellow (b')": np.array([0.0, 0.0, 1.0]),
        }
        _rows = []
        for _gname, _ghex in (("day", "#fdf0ed"), ("night", "#1c1e26")):
            _gj = float(obs.hex_to_ucs(_ghex)[0, 0] / 100.0)
            for _dname, _dv in _dirs.items():
                _qs = _threshold_q(_dv, _gj)
                if _qs is None:
                    continue
                _rows.append(
                    {
                        "ground": _gname,
                        "direction": _dname,
                        "10%": round(float(_qs[0]), 2),
                        "median dE": round(float(_qs[1]), 2),
                        "90%": round(float(_qs[2]), 2),
                        "point (posterior mean)": round(obs.threshold_de(_post, _cols, _dv, _gj), 2),
                    }
                )

        # The colour-deficiency question, asked as a distribution over the ONE quantity that
        # answers it: how many times coarser the red-green direction is than blue-yellow.
        # This needs the joint over (phi, w1, w2) -- three per-axis marginals cannot give
        # it, because the angle decides which weight the red-green direction is reading.
        _phi, _w1, _w2 = _cols[0], _cols[1], _cols[2]
        _rad = np.deg2rad(_phi)

        def _w_dir(_d):
            _u1 = np.cos(_rad) * _d[1] + np.sin(_rad) * _d[2]
            _u2 = -np.sin(_rad) * _d[1] + np.cos(_rad) * _d[2]
            return _d[0] ** 2 + _w1 * _u1**2 + _w2 * _u2**2

        _rg_by = np.sqrt(_w_dir(np.array([0.0, 0.0, 1.0])) / _w_dir(np.array([0.0, 1.0, 0.0])))
        _rl, _rm, _rh = _wq(_rg_by, _post, (0.05, 0.5, 0.95))
        _excl = {_k2: float(_post[_rg_by >= _k2].sum()) for _k2 in (1.5, 2.0, 3.0)}
        _out = mo.vstack(
            [
                mo.md(
                    "**Thresholds as distributions.** The point column is what the cached fit "
                    "publishes and what the aesthetics constraints consume; the interval is "
                    "what the data actually pins down. Where the two disagree, the interval is "
                    "the honest number -- and the width of the red-green rows is the reason "
                    "this table exists."
                ),
                mo.ui.table(_rows, selection=None),
                mo.md(
                    f"**Is there a red-green deficiency? Stated with its power.** The model lets "
                    f"a confusion axis emerge freely (an orientation and a weight ellipse in the "
                    f"chromatic plane), so the question reduces to one ratio: your red-green "
                    f"threshold divided by your blue-yellow one. Posterior median "
                    f"**{_rm:.2f}x**, 90% credible interval **[{_rl:.2f}x, {_rh:.2f}x]**.\n\n"
                    f"That interval is the whole answer, and it says two different things at once. "
                    f"The elevation an anomalous trichromat shows in a comparable discrimination "
                    f"task is several-fold (Boehm, MacLeod & Bosten 2014, JOV; Bosten 2021, "
                    f"Vision Research, reviews a 2-10x range), and this data excludes that: "
                    f"P(ratio >= 3) = {_excl[3.0]:.4f}. But it does **not** establish a normal "
                    f"observer either -- P(ratio >= 2) = {_excl[2.0]:.3f}, and the interval "
                    + (
                        "includes 1.0, so equal red-green and blue-yellow acuity is also consistent with these trials."
                        if _rl <= 1.0
                        else "excludes 1.0, so some real elevation is established."
                    )
                    + " A quiet answer is not an absent effect; it is this much power and no more."
                ),
            ],
            gap=0.8,
        )
    mo.output.replace(_out)
    return


@app.cell(hide_code=True)
def _(get_responses, mo, np, pd, stats):
    # Instrument diagnostics: what the log says about the TASK rather than about the eyes.
    # Every number here is a check that the thing being measured is the thing intended, and
    # each one is cheap enough to run on every page load.
    _log = get_responses()
    if not _log:
        _out = mo.md("*No responses yet -- the diagnostics fill in as you answer.*")
    else:
        _frame = pd.DataFrame(_log)
        # Rows predating the `kind` field encoded it by writing "probe" into `palette`.
        _kind = (
            _frame["kind"]
            if "kind" in _frame
            else pd.Series(np.where(_frame.palette == "probe", "probe", "anchor"), index=_frame.index)
        )
        _kind = _kind.fillna(pd.Series(np.where(_frame.palette == "probe", "probe", "anchor"), index=_frame.index))
        _frame = _frame.assign(kind=_kind)
        _probe = _frame[_frame.kind == "probe"]
        _anchor = _frame[_frame.kind == "anchor"]

        # 1. Anchors and probes are not one population. An anchor is a deliberately easy
        #    palette pair; a probe sits wherever the information search put it. Pooling them
        #    reports a task difficulty nobody set.
        _split = mo.hstack(
            [
                mo.stat(f"{len(_frame):,}", label="responses", bordered=True),
                mo.stat(
                    f"{100 * _probe.correct.mean():.1f}%" if len(_probe) else "--",
                    label="probe accuracy",
                    caption=f"{len(_probe):,} trials, chance 25%",
                    bordered=True,
                ),
                mo.stat(
                    f"{100 * _anchor.correct.mean():.1f}%" if len(_anchor) else "--",
                    label="anchor accuracy",
                    caption=f"{len(_anchor):,} trials, easy by design",
                    bordered=True,
                ),
                mo.stat(
                    f"{100 * _frame.correct.mean():.1f}%",
                    label="pooled",
                    caption="reported for completeness; not a threshold statistic",
                    bordered=True,
                ),
            ],
            justify="start",
            gap=1,
        )

        # 2. Where the information search actually parks the observer. An information-optimal
        #    4AFC trial with a fitted lapse does NOT sit at 75%; it sits wherever the
        #    posterior learns fastest, which for this model is well above it. Worth printing,
        #    because "the trials should feel hard" is a claim about this number and the log
        #    is the only thing entitled to make it.
        _sits = float(_probe.correct.mean()) if len(_probe) else float("nan")

        # 3. Position. Randomised AND tested: the guess that lands when the odd square was
        #    not found is the cleanest view of a positional preference, because it is free of
        #    the odd item's own visibility. The null is uniform over the three slots that
        #    were NOT the target.
        _wrong = _frame[~_frame.correct.astype(bool)]
        _pos_note = "Too few errors yet to test a positional preference."
        if len(_wrong) >= 20:
            _seen = np.array([int((_wrong.choice == _i).sum()) for _i in range(4)])
            _expect = np.zeros(4)
            for _op in _wrong.odd_position:
                for _i in range(4):
                    if _i != _op:
                        _expect[_i] += 1 / 3
            _chi = stats.chisquare(_seen, _expect)
            _worst = int(np.argmax(_seen - _expect))
            _pos_note = (
                f"Over {len(_wrong)} errors the four slots were chosen "
                f"{'/'.join(str(_v) for _v in _seen)} against {'/'.join(f'{_v:.0f}' for _v in _expect)} "
                f"expected under no preference (chi-square p = {_chi.pvalue:.4f}). "
                + (
                    f"**Slot {_worst + 1} attracts guesses.** A biased guess is not neutral: the "
                    "observer model's chance floor is a flat 25% per slot, so a preference makes "
                    "the observer beat chance when the target sits in the favoured slot and lose "
                    "to it elsewhere. Pooled, that is close to a wash in the mean and pure "
                    "overdispersion in the spread, which the fit absorbs as a shallower slope and "
                    "a larger lapse than the eyes deserve. The aesthetics duels hit the same wall "
                    "and answered it by fitting the bias as a term; that fix is not available "
                    "here without a change to the shared likelihood, so it is written down "
                    "instead."
                    if _chi.pvalue < 0.05
                    else "No positional preference this data can see."
                )
            )

        # 4. Task difficulty by slot. In a row of four near-abutting squares an end slot
        #    abuts one same-coloured neighbour and a middle slot abuts two, so the middle
        #    slots offer an extra edge comparison. That is the same defect the aesthetics
        #    probes had when they mixed `def` sites with call sites: if one level of a factor
        #    is systematically easier, reaction time and accuracy measure which level was
        #    drawn rather than what the stimulus was for.
        _end = _frame[_frame.odd_position.isin((0, 3))].correct
        _mid = _frame[_frame.odd_position.isin((1, 2))].correct
        _diff_note = "Too few trials yet to compare end and middle slots."
        if len(_end) >= 30 and len(_mid) >= 30:
            _tab = [
                [int((~_end.astype(bool)).sum()), int(_end.sum())],
                [int((~_mid.astype(bool)).sum()), int(_mid.sum())],
            ]
            _p_em = stats.chi2_contingency(_tab)[1]
            _diff_note = (
                f"End slots (1 and 4) are answered correctly {100 * _end.mean():.1f}% of the time "
                f"and middle slots (2 and 3) {100 * _mid.mean():.1f}% (p = {_p_em:.3f}), over "
                f"{len(_end)} and {len(_mid)} trials."
            )

        # 5. Freshness. A re-shown pair measures memory as well as perception.
        _pairs = _frame.apply(lambda _r: tuple(sorted((_r.base, _r.odd_color))), axis=1)
        _p_pairs = _pairs[_frame.kind == "probe"]
        _fresh_note = (
            f"{_p_pairs.nunique():,} distinct colour pairs over {len(_p_pairs):,} probe trials "
            f"({100 * (1 - _p_pairs.nunique() / max(len(_p_pairs), 1)):.0f}% re-shows)."
            if len(_p_pairs)
            else "No probe trials yet."
        )

        # 6. What the size and ground schedule has actually delivered, as opposed to what it
        #    offers. A parameter the log cannot identify should be visible as such here
        #    rather than inferred from a suspiciously round posterior.
        _by_size = _frame.get("size_px", pd.Series(dtype=float)).fillna(104).value_counts().sort_index()
        _cover = ", ".join(f"{int(_k)} px: {int(_v):,}" for _k, _v in _by_size.items())
        _grounds = ", ".join(f"{_g}: {int(_c):,}" for _g, _c in _frame.ground.value_counts().items())

        _out = mo.vstack(
            [
                mo.md("## Does the instrument measure what it means to?"),
                _split,
                mo.md(
                    f"**Where the search parks you.** Probe accuracy is {100 * _sits:.1f}%. An "
                    "information-optimal 4AFC trial is not a 75%-correct trial: with a fitted "
                    "slope and a fitted lapse the fastest-learning stimulus sits above "
                    "threshold, so this number belongs in the high eighties or nineties and a "
                    "run of trials that feel guessable is the generator working. If it ever "
                    "sits near 25% the generator has lost the plot; near 100% and the trials "
                    "have stopped carrying information."
                ),
                mo.md(f"**Position.** {_pos_note}"),
                mo.md(f"**Equal difficulty across slots.** {_diff_note}"),
                mo.md(f"**Fresh stimuli.** {_fresh_note}"),
                mo.md(f"**Coverage.** Patch sizes -- {_cover}. Grounds -- {_grounds}."),
            ],
            gap=0.7,
        )
    mo.output.replace(_out)
    return


@app.cell(hide_code=True)
def _(PALETTES, VISION_LOG, mo, obs, pd):
    # Palettes ranked by the fitted observer's PREDICTED worst adjacent pair, over both
    # Horizon grounds. This replaces a per-palette accuracy table, which could not answer the
    # question it was labelled with: the information search drives every probe toward the
    # same accuracy, so accuracy ranked the sampler. The prediction ranks the palette.
    if not VISION_LOG.exists() or not len(VISION_LOG.read_text().strip()):
        _out = mo.md("*Palette ranking needs a calibration log.*")
    else:
        _fit = obs.fit(VISION_LOG)
        _rows = []
        for _name, _hexes in PALETTES.items():
            _worst, _pair = 1.0, None
            for _i in range(len(_hexes)):
                for _j in range(_i + 1, len(_hexes)):
                    for _g in ("#fdf0ed", "#1c1e26"):
                        _p = obs.discriminability(_fit, _hexes[_i], _hexes[_j], _g)
                        if _p < _worst:
                            _worst, _pair = _p, f"{_hexes[_i]} / {_hexes[_j]}"
            _rows.append(
                {
                    "palette": _name,
                    "worst pair P(correct)": round(_worst, 3),
                    "that pair": _pair,
                    "colours": len(_hexes),
                }
            )
        _out = mo.vstack(
            [
                mo.md(
                    "**Palettes, hardest pair first, as the fitted observer predicts them.** "
                    "Each row is the *minimum* over every within-palette pair and both Horizon "
                    "grounds of the modelled probability of telling the pair apart in this task "
                    "at 104 px. A categorical palette lives or dies by its worst pair anywhere; "
                    "a sequential ramp by its adjacent ones. 0.25 is chance -- for you, at this "
                    "size, on that page, those two colours are one colour."
                ),
                mo.ui.table(pd.DataFrame(_rows).sort_values("worst pair P(correct)"), selection=None),
            ],
            gap=0.7,
        )
    mo.output.replace(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the numbers, and what happens to them

    A pair at 25% is invisible to you; at 100% it is trivially yours. Sequential ramps live
    or die by their *adjacent* pairs, categorical palettes by their worst pair anywhere. The
    tiles report 75%-correct thresholds as CAM16-UCS distances (dE) -- one perceptual
    currency for every direction, ground and instrument in this repo, fit by the shared
    observer model in `theme/observer.py` (v2: fitted slope and lapse, free confusion-axis
    orientation, threshold smooth in ground lightness, small-field exponent).

    What 748 trials have established, and how firmly:

    - **Dark pages read finer, about 0.78x the light-page threshold** -- but the 80%
      credible interval on that ratio runs to 1.00. The direction is consistent and the
      magnitude is not yet pinned; read it as "probably a real advantage of up to a quarter"
      and not as a measured 22%.
    - **Thresholds at 104 px, Horizon day / night**: lightness 3.2 / 2.5 dE, blue-yellow
      3.6 / 2.8, red-green 5.3 / 4.1, as posterior means. The intervals matter more than the
      means. Blue-yellow and lightness are tight (day blue-yellow 3.3 to 4.5); day red-green
      runs 3.9 to 8.4, a factor of 2.2, so any decision that turns on the red-green figure
      is turning on a number this data has barely constrained.
    - **The psychometric slope is shallow and pressed against its grid.** Posterior mean
      1.18, median 1.2, and a fifth of the mass sitting on the grid's own floor of 0.8 -- so
      the true slope may be shallower still, and the grid, not the data, is setting that
      bound. Same for the lapse: 94% of its posterior is on the smallest value the grid
      offers, so "slip rate 0.6%" is properly read as "at or below 0.5%, which is the least
      this grid can say".
    - **Colour deficiency: excluded at the magnitude the literature describes, not excluded
      at a mild one.** See the interval report above for the numbers and the power. The
      earlier reading of this log said flatly that the data says no; that overstated a
      posterior which still puts about 15% of its mass above a two-fold red-green elevation.
    - **The small-field exponent is not measured at all.** Its marginal is its prior, because
      every trial in the log so far is at 104 px. That number, not the exhibit-scale one,
      decides whether the editor theme can evolve or has to change, and the block schedule
      reaches 16 px and 10 px trials as the log grows. Until it does, the aesthetics
      constraints use a 2x safety margin in place of a measurement, and say so.
    - **Ground warmth stays out of the model** until the ground family decouples it from
      lightness. The hexes ride with every response either way, so the axis can be added to
      the model later and fitted on data already collected.

    ## Stated limitations

    Each of these is a known bias with a known direction. A documented bias can be corrected
    for, argued with, or fixed; a silent one becomes a year of data quietly answering a
    different question.

    1. **The position bias is measured but not modelled.** The diagnostics above test it and
       report it. The observer likelihood in `theme/observer.py` has a flat 25% chance floor
       with no positional term, so a preference among slots enters the fit as
       overdispersion, and the fit pays for it with a shallower slope and a larger lapse than
       the eyes deserve. Fitting it as a term -- which is exactly what the aesthetics duels
       do with their 61% right-hand bias -- is a change to a likelihood shared by three
       instruments, so it is a deliberate, serialised change and not a notebook edit. The
       randomisation was strengthened in the meantime: slots are now assigned by a shuffled
       permutation per group of four rather than by an independent draw, so position is
       balanced exactly and decorrelated from everything else the schedule does.
    2. **The four slots are not equally difficult.** In a row of four near-abutting squares
       an end slot has one same-coloured neighbour and a middle slot has two, so the middle
       offers an extra edge comparison. The diagnostics measure the size of that; a 2x2
       arrangement would remove it, since every cell would then have the same two
       orthogonal neighbours. It has deliberately NOT been changed. Switching the layout
       changes the stimulus, the model carries no layout term to absorb the step, and the
       change would land on the small-field exponent that the next few hundred trials are
       supposed to identify. The `layout` field is now recorded on every response so that a
       future switch has a baseline to be separated from rather than a discontinuity nobody
       can find.
    3. **Patch size and gap are perfectly confounded.** Gap is a fixed function of size
       (12 px at 104, 2 px at 16, 1 px at 10), so the fitted small-field exponent absorbs
       both spatial summation and edge separation and cannot be read as either alone.
       Varying gap independently would identify them, at the cost of a stimulus parameter
       the model cannot fit today, which would land as pure noise. Recorded, confounded,
       documented.
    4. **The reaction-time channel is the app's, not this notebook's.** Rows answered here
       are accuracy only, which is what lets this trial loop live in a notebook at all, and
       is also why "decide within about a second" is advice rather than a measurement for
       them: a slow, reasoned answer and a fast, perceptual one are the same row. Rows
       answered in the app's colour arm carry `rt_ms`, `input_method` and `paused`, and are
       told apart by `surface == "app"`; nothing below reads the clock yet, so the first
       reading of what reaction time adds to a threshold is still to be written.
    5. **A third of the historical probe trials re-show a pair.** 332 distinct pairs over 489
       probe trials, because the information search kept converging on the same offsets from
       the same finite set of palette colours. Probe bases are now jittered off their palette
       colour, which makes new probes fresh by construction; the older rows stay in the fit,
       since a re-shown pair is still a valid trial, just a less informative one.
    6. **The screen is uncalibrated.** That limits absolute claims, not relative ones. Which
       pairs and which palettes fail *you* on *this* screen is exactly what accumulates in
       the log.
    7. **One observer.** Nothing here generalises to anybody else, and it is not meant to.
       The population models are what a gallery uses; this is the instrument that replaces
       them for one pair of eyes.
    8. **Every interval is quoted to grid resolution.** The posterior is exact but discrete:
       nine values for the ground slope, five for the small-field exponent, three for the
       lapse. An interval is therefore reported as actual grid values by inverse CDF rather
       than smoothed between them, which is honest but coarse -- an 80% interval on the
       ground slope can only be [0.0, 0.6] and not [-0.03, 0.46]. Smoothing it looked
       sharper and was wrong twice over: it named values the model cannot hold, and it
       stopped agreeing with itself under a change of variable, which is how the problem was
       found. A finer grid is the fix; it costs time in proportion.

    ## Where these numbers go

    Trials accumulate in `data/calibration-responses.jsonl`, committed like any measurement.
    The fit is cached beside it in `observer-fit.json`, plus an uncommitted binary posterior
    sidecar, and every instrument reads the same fit -- the aesthetics duels' hard
    constraints included -- so a sitting here immediately retunes what the preference
    optimiser is allowed to show. The trial generator is deterministic in the trial number,
    so a sitting resumes exactly where the last one stopped, and past rows are never
    regenerated: every response carries its own complete stimulus, which is what makes a
    generator change safe to land mid-experiment as long as it is stamped, and it is.
    """)
    return


if __name__ == "__main__":
    app.run()
