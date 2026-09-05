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
            "analysis.py is running on an interpreter that does not have theme-calibration "
            "installed, so nothing below this cell can run. Start it with `pixi run analyse`, or "
            "select .pixi/envs/default/bin/python as this folder's interpreter. "
            "README, 'Running it', has both."
        ) from no_package

    from theme.conspicuity import conspicuity_of, find_time_knee, highlight_baseline
    from theme.model import rt_exponent
    from theme.paths import CHAMPION, LIVED_LOG, RESPONSE_LOG
    from theme.responses import ResponseLog
    from theme.space import AXES, DE_MIN, THRESH_DETAIL, VISION_N
    from theme.stimulus import render_card, snippet_for
    from theme.verdict import publish, verdict_for

    return (
        AXES,
        CHAMPION,
        DE_MIN,
        LIVED_LOG,
        RESPONSE_LOG,
        ResponseLog,
        THRESH_DETAIL,
        VISION_N,
        conspicuity_of,
        find_time_knee,
        highlight_baseline,
        np,
        pd,
        publish,
        render_card,
        rt_exponent,
        snippet_for,
        verdict_for,
    )


@app.cell(hide_code=True)
def _(LIVED_LOG, RESPONSE_LOG, ResponseLog, mo):
    # Read through the same object the server appends with, so there is exactly one
    # definition of what a response log is. The lived duels -- themes compared by living in
    # them, recorded by `pixi run lived` -- are a second file the fit reads as part of the
    # same log; they stay apart on disk because the trial generator is a pure function of
    # the instrument's own log and a row it never generated would shift every trial after
    # it. A refresh re-reads both, which is how a sitting in the app and this page open
    # beside it stay in step.
    _instrument = ResponseLog(RESPONSE_LOG).read()
    _lived = ResponseLog(LIVED_LOG).read()
    get_responses, _set_responses = mo.state(_instrument + _lived)
    mo.md(
        f"Read **{len(_instrument):,}** responses from `{RESPONSE_LOG.name}`"
        + (f" and **{len(_lived)}** lived duels from `{LIVED_LOG.name}`." if _lived else ".")
        if _instrument
        else f"`{RESPONSE_LOG.name}` is empty or absent -- take a sitting with `pixi run serve` first."
    )
    return (get_responses,)


@app.cell(hide_code=True)
def _(AXES, mo):
    # The sentences of the verdict, each a function of the verdict object and nothing else,
    # so the wording lives in one place and the numbers in another (theme.verdict). One
    # sentence per case throughout: joining fragments once produced "Your clicks have still
    # open on accent hue rotation" the first time night had nothing settled.

    def headline(verdict):
        lead_pct = 100 * verdict.lead
        if verdict.verdict == "single":
            return (
                f"**one theme leads** -- it holds {lead_pct:.0f}% of the probability of being the "
                f"best theme, so this is the one to apply"
            )
        if verdict.verdict == "plateau":
            return (
                f"**a plateau of {len(verdict.credible)} distinct themes** -- the leader holds "
                f"{lead_pct:.0f}%, and these together hold half the probability of being best. They "
                f"are equally good by measurement, not merely acceptable: every one has already "
                f"cleared the legibility floors, so pick by eye"
            )
        return (
            f"**not yet decided** -- the strongest theme holds only {lead_pct:.0f}% of the "
            f"probability of being best, which is what a thin log looks like rather than a real "
            f"plateau. {len(verdict.credible)} themes share half the mass; more duels on this "
            f"polarity will separate them"
        )

    def legibility_sentence(verdict):
        # Whether taste is costing reading speed -- as a DIFFERENCE with an interval, never
        # two point estimates side by side. The posterior sd on either page is around 0.3 in
        # log time, so the difference of two is wider still, and the minimum over several
        # hundred noisy predictions is an extreme order statistic, biased low.
        note = verdict.legibility
        if note is None:
            return ""
        low, high = note.gap_interval
        head = (
            f" {note.n_excluded} of {note.n_candidates} candidates were dropped first as credibly "
            f"slower to read than the fastest ({note.n_timed} timed trials). The leader reads in "
            f"about {note.champion_seconds / 1000:.1f} s"
        )
        if note.champion_credibly_slower:
            return head + (
                f", and that is credibly slower than the quickest page the model knows by "
                f"{note.gap_log_time:+.2f} in log time (95% interval [{low:+.2f}, {high:+.2f}]) -- "
                f"taste and speed are pulling apart here, and the shelf is worth re-reading with "
                f"that in mind."
            )
        return head + (
            f", which is not measurably slower than the quickest page the model knows: the "
            f"difference is {note.gap_log_time:+.2f} in log time with a 95% interval of "
            f"[{low:+.2f}, {high:+.2f}], so liking these pages is costing you no measurable "
            f"reading speed."
        )

    def progress_sentence(verdict):
        progress = verdict.progress
        if progress is None:
            return ""
        moved = 100 * (progress["lead_now"] - progress["lead_then"])
        shrunk = progress["set_then"] - progress["set_now"]
        head = (
            f" Over the last {progress['back']} duels the leader's share moved {moved:+.0f} points "
            f"and the credible set changed by {-shrunk:+d} themes"
        )
        if progress["duels_to_decide"] is not None:
            # A leader gaining ground: extrapolate, and say plainly that it is a straight
            # line through two points.
            return head + (
                f"; at that rate roughly {progress['duels_to_decide']} more duels would give one "
                f"theme a majority -- a naive straight-line estimate, worth reading as 'another "
                f"sitting' or 'another ten'."
            )
        if shrunk > 0:
            # Mass can move AWAY from the leader while the set shrinks. That is not stalling,
            # it is the model resolving a real plateau.
            return head + (
                " -- so evidence is still arriving and the field is narrowing, but the mass is "
                "spreading across the survivors rather than concentrating: what a genuine plateau "
                "looks like as it comes into focus. More duels sharpen WHICH themes are on the "
                "shelf, not which one wins."
            )
        return head + (
            " -- flat on both counts, so more duels on this polarity are buying little and the "
            "shelf above is the answer rather than a waypoint."
        )

    def factor_sentence(verdict):
        # Whether one theme is even the right SHAPE of answer: if the optimum moves between
        # the editor, the chat panel and a notebook, converging one theme onto all three
        # averages over a real difference instead of resolving it. Two factors are tested per
        # polarity, so about one reading in every two or three sittings lands under 0.10
        # with nothing real behind it; the sentence says so.
        n, gain, p_value, _wording = verdict.factors["surface"]
        polarity = verdict.polarity
        if n < 24:
            text = (
                f" Surface (editor / panel / notebook) is logged but only {n} {polarity} duels carry "
                f"a label so far, too few to ask whether the optimum moves between them."
            )
        elif p_value < 0.02:
            text = (
                f" **Surface matters**: a per-surface tilt earns {gain:+.3f} nats/duel on held-out "
                f"choices against its own permutation null (p = {p_value:.3f}, {n} duels). One theme "
                f"is the wrong shape of answer here -- the editor, the chat panel and the notebook "
                f"want different pages."
            )
        elif p_value < 0.10:
            text = (
                f" Surface may matter: a per-surface tilt earns {gain:+.3f} nats/duel held out "
                f"(p = {p_value:.3f} over {n} duels) -- suggestive, not established, and one of four "
                f"such factor readings, of which roughly one lands here by chance anyway. Duels are "
                f"surface-balanced in groups of three, so this sharpens on its own; worth re-reading "
                f"at about twice this many duels."
            )
        else:
            text = (
                f" No surface effect this data can see (p = {p_value:.2f} over {n} duels), so one "
                f"theme across editor, panel and notebook remains the right shape of answer."
            )
        n, _gain, p_value, _wording = verdict.factors["code_px"]
        if n >= 24 and p_value < 0.10:
            text += (
                f" Type size also tilts the answer (p = {p_value:.3f} over {n} duels): the early "
                f"duels judged at 12-13px are measuring a different question from those at the "
                f"real 14 and 16, and should carry less weight."
            )
        elif n >= 24:
            text += (
                f" Duels judged at different type sizes agree (p = {p_value:.2f}), so the early "
                f"12-13px rounds pool safely with the ones at the real reading sizes."
            )
        return text

    def consensus_sentence(verdict):
        # What the plateau actually disagrees about: four pages that share a ground and
        # differ only in accent hue read as "four identical themes" unless the reader is
        # told the ground is decided and the hue is not.
        settled = sorted((c for c in verdict.consensus if c[1] < 0.55), key=lambda c: c[1])
        open_axes = sorted((c for c in verdict.consensus if c[1] > 0.85), key=lambda c: -c[1])

        def names(group):
            return ", ".join(f"**{AXES[axis]}**" for axis, _spread, _mean in group[:3])

        if settled and open_axes:
            return (
                f" Your clicks have settled {names(settled)}, and have not yet separated "
                f"{names(open_axes)} -- so the themes on this shelf mostly differ in the second "
                f"group, and that is what further duels decide."
            )
        if settled:
            return (
                f" Your clicks have settled {names(settled)}, and no axis is still wide open: what "
                f"remains is fine separation rather than an open question."
            )
        if open_axes:
            verb = "are" if len(open_axes) > 1 else "is"
            return (
                f" No axis has settled yet, and {names(open_axes)} {verb} still wide open -- the "
                f"shelf differs there, and that is what further duels decide."
            )
        return ""

    def verdict_prose(verdict):
        return mo.md(
            f"### The {verdict.polarity} verdict\n\n{headline(verdict)}.{legibility_sentence(verdict)}"
            f"{progress_sentence(verdict)}{factor_sentence(verdict)}{consensus_sentence(verdict)}"
            " Shown below: the leader, then the most *different* members of the set holding half "
            "the probability mass -- near-identical themes are grouped first, so what you see are "
            "choices rather than variations of one."
        )

    return (verdict_prose,)


@app.cell(hide_code=True)
def _(
    AXES,
    CHAMPION,
    DE_MIN,
    THRESH_DETAIL,
    VISION_N,
    conspicuity_of,
    find_time_knee,
    get_responses,
    highlight_baseline,
    mo,
    np,
    pd,
    publish,
    render_card,
    rt_exponent,
    snippet_for,
    verdict_for,
    verdict_prose,
):
    # A stable page for the champion preview: the same generated code every time, so
    # what changes between renders is the theme and nothing else.
    _preview_snip = snippet_for(0)
    _log = get_responses()
    if not _log:
        _out = mo.md("*No responses yet -- the analysis fills in as you answer.*")
    else:
        _frame = pd.DataFrame(_log)
        _n_duel = int((_frame["mode"] == "duel").sum())
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
                        label=f"dE floors (day/night), {VISION_N} vision trials",
                        bordered=True,
                    ),
                ],
                justify="start",
                gap=1,
            )
        ]
        for _pol in ("day", "night"):
            # Everything below is read off one object, computed once in theme.verdict: the
            # candidates are BRED rather than taken from the frozen pool (the answer should be
            # the best theme the search can reach, not the best of 512 points fixed before
            # the first click), the timed arms bind before the preference posterior picks a
            # winner, and P(best) is sampled from the joint posterior and grouped.
            _v = verdict_for(_log, _pol)
            if _v is None:
                continue
            _blocks.append(verdict_prose(_v))
            # Full-bleed, and each card wide enough for the page it holds. The prose column
            # is 610px because that is a good measure for READING; four theme cards inside it
            # are 306px each, which clips the page mid-token. A page needs about 520px to
            # render whole at 12px, so the row steps out of the measure and the cards wrap
            # instead of shrinking.
            _cards = "".join(
                f'<figure style="margin:0;flex:0 0 520px;max-width:100%">'
                f'<figcaption style="font:600 13px/1.5 system-ui,sans-serif;margin:0 0 6px 2px">'
                f"{100 * _v.shown_probability[_i]:.0f}%"
                f"{' · leads' if _i == _v.shown[0] else ''}"
                f'<span style="font-weight:400;opacity:.65"> · utility {_v.mean_utility[_i]:.2f}</span>'
                f"</figcaption>"
                f'<div style="background:{_v.themes[_i]["ground"]};border-radius:8px;'
                f'padding:14px;overflow:hidden">'
                + render_card(_v.themes[_i], _preview_snip, 12, prose=False)
                + "</div></figure>"
                for _i in _v.shown
            )
            _blocks.append(
                mo.Html(
                    '<div style="width:94vw;margin-left:calc(-47vw + 50%);display:flex;'
                    'flex-wrap:wrap;gap:22px;justify-content:center">' + _cards + "</div>"
                )
            )
            _champ = _v.champion_theme
            _blocks += [
                mo.md(
                    f"**Current best {_pol} page** -- beats a random feasible theme with "
                    f"p = {_v.beats_random:.2f}; utility marginals below are the posterior-mean change "
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
                mo.ui.table(pd.DataFrame(_v.axis_sweep(AXES)), selection=None),
            ]
        # Publish rather than print. The applier reads this file, so the palette crosses
        # from instrument to editor without a human copying hex codes -- the step where a
        # digit gets dropped and nobody notices for a week. Written on every analysis pass,
        # so it always reflects the current log; the applier is what decides when the editor
        # changes, and it is never this notebook. `pixi run publish` writes the same file
        # without a browser.
        if _n_duel >= 5:
            _published = publish(_log)
            _blocks.append(
                mo.md(
                    f"Champions published to `{CHAMPION.name}` for {', '.join(_published)} -- apply with "
                    f"`apply-measured-theme` (dotfiles), which reads this file and rewrites the marked "
                    f"regions of settings.jsonc, and record what living in it was like with "
                    f"`pixi run lived`. The palettes, for reading:"
                )
            )
            for _pol, _palette in _published.items():
                _blocks.append(
                    mo.md(
                        "```jsonc\n"
                        + "{\n"
                        + f"  // {_pol} · ground {_palette['ground']} · page {_palette['page']}\n"
                        + f'  "editor.background": "{_palette["ground"]}",\n'
                        + f'  "editor.findMatchBackground": "{_palette["find_fill"]}d9",\n'
                        + f'  "editor.findMatchHighlightBackground": "{_palette["find_fill"]}73",\n'
                        + '  "textMateRules": {\n'
                        + f'    "keyword": "{_palette["keyword"]}", "function": "{_palette["function"]}",\n'
                        + f'    "string|number": "{_palette["string"]}", '
                        + f'"comment (italic)": "{_palette["comment"]}",\n'
                        + f'    "variables/ink": "{_palette["ink"]}", "punctuation": "{_palette["punct"]}"\n'
                        + "  }\n"
                        + "}\n```"
                    )
                )
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
                    f"(fastest quartile {_ok['rt_ms'].quantile(0.25):.0f} ms -- the gap is what theming can win)."
                )
            )
        _rtp, _rtp_scores = rt_exponent(_log)
        if _rtp_scores and 0.0 in _rtp_scores:
            _gain = _rtp_scores[0.0] - _rtp_scores[_rtp]
            _blocks.append(
                mo.md(
                    f"**The clock's weight is fitted, not assumed**: duels are weighted by "
                    f"(median time / this time) to the power {_rtp}, chosen by held-out log-loss "
                    f"over {{0, 1/4, 1/2, 3/4}} and refit every 25 duels. Zero is in that set on purpose -- "
                    f"it means ignoring the clock -- and it currently loses by {_gain:.4f} nats per "
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
                        f"log time-to-find slope over salience {_z[0]:+.3f} per dE "
                        f"(negative = louder is genuinely faster; near zero = salience past this point buys nothing "
                        f"and beauty should take the wheel)."
                    )
                )
                # The baseline is a constant until this knee identifies. Read per polarity, in
                # the observer's own steps, from the hexes that were on screen.
                for _pol in ("day", "night"):
                    _hp = _hok[_hok["polarity"] == _pol]
                    _knee = find_time_knee(
                        [conspicuity_of(_t, _pol)[0] for _t in _hp["theme_a"]], _hp["rt_ms"].to_numpy()
                    )
                    _floor = highlight_baseline(_pol)[0]
                    if _knee is None:
                        _blocks.append(mo.md(f"*{_pol} hunts: {len(_hp)} correct so far; a knee needs 12.*"))
                        continue
                    _blocks.append(
                        mo.md(
                            f"**Where loudness stops buying time, {_pol}**: a hinge on log find-time over "
                            f"the highlight's conspicuity puts the knee at {_knee.knee_jnd:.1f} observer steps "
                            f"({_knee.n} hunts, {_knee.slope_per_step:+.2f} log-time per step below it, "
                            f"explaining {100 * _knee.gain:.0f}% of the variance over a flat line). The "
                            f"baseline every highlight owes the page is {_floor:g} steps, a constant; "
                            + (
                                "the knee has not identified it, so the constant stands."
                                if _knee.gain < 0.3 or _knee.n < 40
                                else "the knee is identifying -- read it against the constant."
                            )
                        )
                    )
        if THRESH_DETAIL.get("day"):
            _blocks.append(
                mo.md(
                    "Constraint provenance -- your fitted 75%-correct thresholds in CAM16-UCS dE (day / night): "
                    + ", ".join(
                        f"{_ax} {THRESH_DETAIL['day'][_ax]:.1f} / {THRESH_DETAIL['night'][_ax]:.1f}"
                        for _ax in THRESH_DETAIL["day"]
                    )
                    + " -- the pairwise floor is 2x the minimum "
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
