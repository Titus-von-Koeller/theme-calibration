# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

# The repository default is lazy, which marks a cell stale rather than running it when
# something upstream changes. That is right for a notebook holding a model on a GPU and
# wrong for a report: every cell here reads the response log and renders a number, so a
# stale cell is a wrong number wearing the same typeface as a right one. Script metadata
# is merged over the project config at the highest precedence, so a notebook opts in on
# its own.
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
    *The reading half of the theme instrument. The trials are taken in the app served by
    `pixi run serve`; this notebook reads what they recorded and reports what the model
    believes.*

    # What the duels have decided

    Legibility floors are measurable and measured; above them, theming has been guesswork.
    The instrument replaces the guess with a model: a latent **aesthetic utility** over a
    CAM16-UCS-parametrised theme space — page lightness and warmth, the accent set's hue,
    chroma, contrast and spread, how far comments recede, and the editor's find-highlight as
    its own salience-versus-beauty axis. Three kinds of trial feed it:

    - **A duel**: two candidate pages render the same generated code at the pixel size that
      surface is really read at. Click the one you would rather live in.
    - **A comprehension probe**: one page, one instruction — *click the function name*. Time
      to land measures what is genuinely easy to grasp, not what merely looks tidy.
    - **A find hunt**: several matches are highlighted; click the current one. Time-to-find
      calibrates how loud the find highlight has to be before it stops earning its salience.

    Under the trials sits preferential Bayesian optimisation: a Gaussian-process posterior
    over utility, a Bradley-Terry likelihood over the choices sharpened by reaction time (a
    fast click is strong evidence; a slow one reads as a near-tie, the way drift-diffusion
    models read decision time), and each duel *generated* to be maximally informative — the
    model's best guess against the challenger that would teach it most, with a small share of
    uniform probes as insurance against a model that only asks questions it already believes.
    Candidates are **bred fresh every trial** rather than drawn from a fixed list, so the
    search can sit between any two themes it has shown and can still reach ground it has
    never visited. Every page is **code never seen before** — generated, or lifted from a
    corner of the standard library — because a reused page turns time-to-find into a memory
    test. Measured discrimination thresholds (from `calibration-responses.jsonl`, via
    `theme.observer`) and APCA/WCAG contrast floors are **hard constraints, never
    objectives**: every candidate shown is already legible, and the only question ever asked
    is which is better.

    Nothing is asked of the observer but clicks. Which colours they love is **inferred, never
    declared**: the prior mean carries only the field's general harmony models, and the hues
    emerge from the duels — which is why the search keeps exploring hue rather than settling
    on lightness alone, and why a stated favourite would be worth less than a measured one
    anyway.

    **This notebook defines nothing of its own.** Every model, every colour transform and
    every trial rule lives in the `theme` package, so the numbers below are the same numbers
    the running instrument acts on. A notebook that re-implemented any of it would eventually
    disagree with the app, and the disagreement would look like a result.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json

    import numpy as np
    import pandas as pd

    from theme.model import (
        axis_consensus,
        best_set,
        candidates,
        factor_effect,
        fitted,
        mu_at,
        progress_report,
        rt_at,
        rt_exponent,
        rt_fit,
        rt_penalty,
        spread_out,
        surface_effect,
    )
    from theme.paths import CHAMPION, RESPONSE_LOG
    from theme.responses import ResponseLog
    from theme.space import AXES, DE_MIN, THRESH_DETAIL, VISION_N
    from theme.stimulus import render_card, snippet_for

    return (
        AXES,
        CHAMPION,
        DE_MIN,
        RESPONSE_LOG,
        ResponseLog,
        THRESH_DETAIL,
        VISION_N,
        axis_consensus,
        best_set,
        candidates,
        factor_effect,
        fitted,
        json,
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
    )


@app.cell(hide_code=True)
def _(RESPONSE_LOG, ResponseLog, mo):
    # Read through the same object the server appends with, so there is exactly one
    # definition of what a response log is. A refresh re-reads the file, which is how a
    # sitting in the app and this page open beside it stay in step.
    _responses = ResponseLog(RESPONSE_LOG).read()
    get_responses, _set_responses = mo.state(_responses)
    mo.md(
        f"Read **{len(_responses):,}** responses from `{RESPONSE_LOG.name}`."
        if _responses
        else f"`{RESPONSE_LOG.name}` is empty or absent — take a sitting with `pixi run serve` first."
    )
    return (get_responses,)


@app.cell(hide_code=True)
def _(
    AXES,
    CHAMPION,
    DE_MIN,
    THRESH_DETAIL,
    VISION_N,
    axis_consensus,
    best_set,
    candidates,
    factor_effect,
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
                # The timed arms bind here. A page you like but read slowly is not a
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
                        # Whether taste is costing reading speed -- as a DIFFERENCE
                        # with an interval, never as two point estimates side by side. The
                        # old wording put "5.6 s" next to "3.4 s for the quickest page the
                        # model knows" and let the reader draw a conclusion the data does not
                        # support twice over: the posterior sd on either page is around 0.25
                        # to 0.38 in log time, so the difference of two is wider still; and
                        # the minimum over several hundred noisy predictions is an extreme
                        # order statistic, biased low, so the fast end of that comparison was
                        # partly a selection effect. Measured on the current log the gap is
                        # +0.03 [-0.80, +0.85] by day: no measurable cost to the preferred
                        # pages, which is the reassuring answer and also the honest one.
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
                # Duels used to run at 12-13px, a size nobody reads code at; they now run at
                # each surface's real reading size (14 in editors, 16 in notebook cells).
                # That pools two stimulus regimes in one log, so the same test asks
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
                        f"those at the real 14 and 16, and should carry less weight."
                    )
                elif _zn >= 24:
                    _surf_note += (
                        f" Duels judged at different type sizes agree (p = {_zp:.2f}), so the early "
                        f"12-13px rounds pool safely with the ones at the real reading sizes."
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
                # palettes compared by the left two-thirds of every line. A page needs
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
                # what decides when the editor changes, and it is never this notebook.
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
                        f"whatever applier owns the editor settings: it reads this file, so the palette "
                        f"crosses from instrument to editor without anyone retyping a hex code. "
                        f"The same palette, for reading:"
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
                    + " — the pairwise floor is 2x the minimum "
                    + f"({2 * DE_MIN['day']:.1f} day, {2 * DE_MIN['night']:.1f} night)."
                )
            )
        _out = mo.vstack(_blocks, gap=0.8)
    mo.output.replace(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the numbers, and what happens to them

    The utility is latent and relative — only differences mean anything, so the readout is
    given as *probabilities*: how often the current champion would beat a random feasible
    theme in a duel that has not been run. Marginals near zero on an axis mean taste is flat
    there (let the constraints or the prior decide); a large one-sided marginal means the axis
    matters and the champion sits where the clicks put it. Early sittings look noisy — a
    preferential GP needs roughly forty duels before the Thompson arm stops wandering — and
    the 7% uniform probes *should* occasionally look strange: they are the insurance premium
    against a model that only ever asks questions it already believes.

    **One theme or several?** The verdict above is a distribution, not a ranking: sampling the
    joint posterior gives each candidate its probability of being *the* best, and the answer
    is read from how that mass sits. Two details make that honest. Near-identical candidates
    are **grouped** before counting, because eight hundred candidates contain many pages that
    differ by less than anyone could see, and each sibling would otherwise steal argmax mass
    from the others. And the reading is of *cumulative* mass, not a fixed cutoff: one group
    holding over half of it is a winner; a handful sharing it is a real plateau, and the
    members shown are then chosen to be as *different* from each other as the set allows; and
    when even the strongest group is thin, the report says **not yet decided** rather than
    dressing a thin log up as a plateau. Nothing on that shelf is a compromise either way —
    every candidate has already cleared the legibility floors, so a plateau means genuinely
    equal, not merely acceptable.

    Two properties of the machinery were measured rather than assumed, and the tests live
    under `tests/`. Card position matters: over the first 79 duels the right-hand card won 61%
    of the time, so a side-advantage term is fitted and subtracted out instead of being left
    to land on the themes as noise. And the nine axes are not equally alive: their
    length-scales are learned, which shrinks the effective dimension the search has to cover,
    with the estimate held near isotropic until enough duels exist to identify relevance at
    all.

    Reaction time is doing quiet work throughout: a fast duel click steepens that duel's
    likelihood (drift-diffusion reading — big utility gaps decide quickly), a slow one
    flattens it toward a tie, so deliberating over a near-tie neither punishes nor rewards
    either side. That channel is only as clean as its baseline, which is why the trial surface
    is a page that owns its own DOM rather than a notebook cell. The first trial of a sitting
    and the first of every run start hidden behind a **begin** button, because those are the
    moments an instruction is read; inside a run the click that produced a trial is its
    anchor, so the clock starts at render and no button stands between the reader and the next
    page. Arrow keys answer a duel, which is a measurement fix and not a comfort: reaching the
    left card is a different distance of mouse travel than the right, so reaction time carried
    a systematic side component on top of the fitted side bias. The input method is recorded
    per response, so mouse and key trials stay separable. A **pause** button, the tab losing
    visibility, or 25 s without a click re-covers the stimulus — an exposed one lets a
    decision form off the clock — and resuming re-baselines; a trial paused after its first
    reveal is flagged in the log: its choice still counts, at the neutral slope, and it is
    excluded from the comprehension and find-hunt timing statistics.

    Comprehension probes and find hunts measure time directly, and that time now **binds**: a
    Gaussian process over log time-to-click gives a legibility surface across theme space, and
    candidates it says are credibly slower to read than the fastest are dropped before the
    preference verdict is computed. Constraint first, preference second — the same order the
    contrast floors use, one level deeper: a floor keeps a page readable in principle, this
    keeps it readable in fact. A page that is liked but read slowly is not a winner. The
    surface estimates its own signal and noise from the recorded times rather than borrowing
    the preference kernel's, because reaction time is noisy enough that a loose prior invents
    differences, and with a thin or noisy log it drops nothing — the honest behaviour rather
    than the convenient one. It also carries a **per-arm and per-size baseline**, because a
    find hunt and a comprehension probe are not equally hard and the timed arms have not
    always run at one type size; without those baselines a change of task or of glyph scale
    would land on the theme surface as if some region of theme space had got slower on the day
    the stimulus changed. These arms are also the glyph-scale ground truth that the 2x
    threshold safety margin (from the 104-px vision fit) stands in for until this instrument
    accumulates its own.

    Hard floors are never traded: every page shown clears WCAG 4.5:1 and APCA 60 on body
    tokens, and every pair of coloured roles clears twice the measured CAM16-UCS threshold for
    its ground. Literals are one family by measurement, not taste: Horizon's own day string
    and number oranges sit within threshold of each other. Plain variable reads render as ink
    by standing preference (figure-ground: definitions, literals and control words carry the
    colour).

    The winner's destination: the find-highlight pair lands in `editor.findMatchBackground`
    and `editor.findMatchHighlightBackground`, the token colours in
    `editor.tokenColorCustomizations` per theme, the ground in the workbench block — via the
    champion file above, once its posterior stops moving between sittings. Trials accumulate
    in `aesthetics-responses.jsonl`, committed like any measurement; the trial generator is
    deterministic given that log, so any session resumes exactly where the last one stopped.
    Findings that outlive a sitting get written into this closing prose, next to the live
    numbers.
    """)
    return


if __name__ == "__main__":
    app.run()
