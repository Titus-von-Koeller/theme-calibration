"""The preference model: which theme is liked better, from duels.

A Gaussian process over theme space with a Bradley-Terry likelihood on duels, fit by
Laplace approximation -- Chu & Ghahramani's preferential GP -- plus the held-out
cross-validation that decides how much the clock is allowed to weight a duel, and the
expected-information-gain helpers the trial chooser acquires against.
"""

import numpy as np

from .kernel import SF2, ard_scales, coords, kmat


def realized_space():
    """The realized-theme layer (`POOL`, `prior_mean`, `realize_many`), resolved at call
    time through `theme.model`.

    `theme.model` owns those three bindings and this module reaches them through it rather
    than importing them from `theme.space` directly, because that is where they are
    SUBSTITUTED: tests/conftest.py replaces them on `theme.model` so the search can be
    exercised without the contrast floors deciding which candidates exist. A module-level
    `from .space import prior_mean` here would bind past the substitution and the stub
    would silently do nothing -- the failure this suite has already met once, when a stub
    of the older per-theme `realize` kept passing after the search moved to batches. One
    sys.modules lookup per call, against a Cholesky.
    """
    from . import model

    return model


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
                _pts.append(coords(_th, _r["polarity"]))
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
    _prior_mean = realized_space().prior_mean
    _m = np.array([_prior_mean(_x[:9], "night" if _x[9] > 0.5 else "day") for _x in _X])
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
    _K = kmat(X, X, ls) + 1e-6 * np.eye(_n)
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
    _ks = kmat(Xs, X, ls)
    _mu = ms + _ks @ (Ki @ (f - m))
    _A = Ki - Ki @ cov @ Ki
    _var = np.maximum(SF2 - np.einsum("ij,jk,ik->i", _ks, _A, _ks), 1e-9)
    return _mu, _var, _ks, _A


def posterior_joint(fit, thetas, polarity):
    """Mean and FULL covariance over candidates -- what P(best) needs.

    Marginal variances cannot answer "which of these is the best theme": candidates
    near each other in theme space share almost all their uncertainty, and ignoring
    that correlation would scatter the probability of being best across a cluster of
    effectively identical pages.
    """
    _prior_mean = realized_space().prior_mean
    _Xs = np.array([coords(_t, polarity) for _t in thetas])
    _ms = np.array([_prior_mean(_t, polarity) for _t in thetas])
    _ls = fit.get("ls")
    _ks = kmat(_Xs, fit["X"], _ls)
    _mu = _ms + _ks @ (fit["Ki"] @ (fit["f"] - fit["m"]))
    _A = fit["Ki"] - fit["Ki"] @ fit["cov"] @ fit["Ki"]
    _cov = kmat(_Xs, _Xs, _ls) - _ks @ _A @ _ks.T
    _cov = 0.5 * (_cov + _cov.T) + 1e-8 * np.eye(len(thetas))
    return _mu, _cov


def h2(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9)
GH_W = GH_W / GH_W.sum()


def posterior_over(fit, thetas, polarity):
    _X, _duels, _lam, _m = fit["X"], fit["duels"], fit["lam"], fit["m"]
    _prior_mean = realized_space().prior_mean
    _Xs = np.array([coords(_t, polarity) for _t in thetas])
    _ms = np.array([_prior_mean(_t, polarity) for _t in thetas])
    return predict(_X, fit["f"], _m, fit["cov"], fit["Ki"], _Xs, _ms, fit.get("ls"))


FIT_MEMO = {}


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
    _ls = ard_scales(_X, _duels, _lam)
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


RTP_MEMO = {}


def rt_exponent(responses, grid=(0.0, 0.25, 0.5, 0.75), refit_every=25):
    """The RT exponent that predicts his next answer best, refit occasionally.

    Returns (best exponent, {exponent: held-out log-loss}). Zero is in the grid on
    purpose: if ignoring the clock predicts as well, the channel is noise dressed as
    evidence and the model should say so rather than carry a flattering heuristic.
    """
    _nd = sum(1 for _r in responses if _r.get("mode") == "duel" and _r.get("choice") in (0, 1))
    _bucket = _nd // refit_every
    if _bucket in RTP_MEMO:
        return RTP_MEMO[_bucket]
    _scores = {}
    for _p in grid:
        _v = cv_logloss(responses, _p)
        if _v is not None:
            _scores[_p] = _v
    _out = (0.5, {}) if not _scores else (min(_scores, key=_scores.get), _scores)
    if len(RTP_MEMO) > 3:
        RTP_MEMO.pop(next(iter(RTP_MEMO)))
    RTP_MEMO[_bucket] = _out
    return _out


def fitted(responses, rt_p=None):
    # Keyed by how many duels have been answered: the fit is a pure function of the
    # log, three cells ask for the same one, and it is the cubic-cost step. Only the
    # newest entry is kept -- an older fit is never asked for again.
    _key = sum(1 for _r in responses if _r.get("mode") == "duel" and _r.get("choice") in (0, 1))
    if rt_p is None:
        rt_p = rt_exponent(responses)[0]
    _key = (_key, rt_p)
    if _key in FIT_MEMO:
        return FIT_MEMO[_key]
    _d = duels_from(responses, rt_p)
    if _d is None:
        return None
    _X, _duels, _lam, _m, _sides = _d
    _ls = ard_scales(_X, _duels, _lam)
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
    if len(FIT_MEMO) > 4:
        FIT_MEMO.pop(next(iter(FIT_MEMO)))
    FIT_MEMO[_key] = _out
    return _out


def mu_at(fit, thetas, polarity):
    """Posterior-mean utility at arbitrary thetas — the analysis cell's window in."""
    return posterior_over(fit, thetas, polarity)[0]
