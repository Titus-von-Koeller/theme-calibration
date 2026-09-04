"""What the next trial should be.

The polarity blocks, the arm schedule inside a run, the surface assignment, and the trial
generator that decides which two themes to show and on what page. Extracted verbatim from
calibrate-aesthetics.py on 2026-09-04.
"""

import random

import numpy as np

from .model import (
    GH_W,
    GH_X,
    candidates,
    coords,
    fitted,
    h2,
    kmat,
    posterior_over,
    rt_at,
    rt_fit,
)
from .space import POOL, realize
from .stimulus import DUEL_WIDTH, READING_PX, SURFACES


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
TRIAL_MEMO = {}


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
    if n in TRIAL_MEMO:
        return TRIAL_MEMO[n]
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
            _mu, _var, _ks, _A = posterior_over(_fit, _cand, _pol)
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
                _cross = kmat(
                    np.array([coords(_t, _pol) for _t in _cand]),
                    np.array([coords(_cand[_i1], _pol)]),
                    _fit.get("ls"),
                )[:, 0] - np.einsum("ij,jk,k->i", _ks, _A, _ks[_i1])
                _mud = _mu - _mu[_i1]
                _s2 = np.maximum(_var + _var[_i1] - 2 * _cross, 1e-9)
                _pbar = 1.0 / (1.0 + np.exp(-_mud / np.sqrt(1 + np.pi * _s2 / 8)))
                _cond = h2(1.0 / (1.0 + np.exp(-(_mud[:, None] + np.sqrt(_s2)[:, None] * GH_X[None, :])))) @ GH_W
                _eig = h2(_pbar) - _cond
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
            _mu, _var, _ks, _A = posterior_over(_fit, [_b[0] for _b in _bred], _pol)
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
            _mu = posterior_over(_fit, [_b[0] for _b in _bred], _pol)[0]
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
    TRIAL_MEMO[n] = _trial
    return _trial
