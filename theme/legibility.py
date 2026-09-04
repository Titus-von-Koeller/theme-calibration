"""The legibility surface: how fast a theme can actually be read.

A Gaussian process regression on log time-to-click over the same theme space the
preference model searches, turning the timed arms into a constraint on the verdict.
"""

import math

import numpy as np

from .kernel import SF2, coords, kmat


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
        _X.append(coords(_r["theta_a"], polarity))
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
    _K = (_sf2 / SF2) * kmat(_X, _X, ls) + _noise * np.eye(len(_X))
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
    _Xs = np.array([coords(_t, polarity) for _t in thetas])
    _scale = rf["sf2"] / SF2
    _ks = _scale * kmat(_Xs, rf["X"], rf.get("ls"))
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
