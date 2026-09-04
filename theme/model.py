"""The preference and legibility models.

A Gaussian process over theme space with a Bradley-Terry likelihood on duels, fit by
Laplace approximation (Chu & Ghahramani's preferential GP), with QUEST+'s
generate-the-most-informative-trial loop on top; and a second GP regression over log
reaction time that turns the timed arms into a legibility constraint on the verdict.

Extracted verbatim from calibrate-aesthetics.py's model cell on 2026-09-04 -- the
behaviour, the constants and the reasoning comments are unchanged; only marimo's
cell-local underscores are gone.
"""

import math

import numpy as np
from scipy.stats import qmc

from .space import POOL, prior_mean, realize_many

# The preference model: a Gaussian process over theme space with a Bradley-Terry
# likelihood on duels, fit by Laplace approximation — Chu & Ghahramani's preferential
# GP, QUEST+'s generate-the-most-informative-trial loop on top. Reaction time enters
# the likelihood drift-diffusion-style: decision time falls as the utility gap grows,
# so a fast click steepens that duel's slope and a slow one flattens it toward a tie.
# Length-scales are ARD: one per axis, estimated from the data rather than fixed, so
# axes his choices ignore get long scales and stop costing sample efficiency. Nine
# dimensions at ~100 duels is the binding constraint on how fast this converges, and
# ARD is the cheapest honest way to shrink the effective dimension.
LS0 = np.array([0.35] * 9 + [0.9])
SF2 = 4.0


def kmat(A, B, ls=None):
    _l = LS0 if ls is None else ls
    _d2 = (((A[:, None, :] - B[None, :, :]) / _l) ** 2).sum(-1)
    _r = np.sqrt(_d2 + 1e-12)
    return SF2 * (1 + np.sqrt(5) * _r + 5 * _r**2 / 3) * np.exp(-np.sqrt(5) * _r)


def ard_scales(X, duels, lam):
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
        return LS0.copy()
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
    _ls = (1.0 - _w_ard) * LS0 + _w_ard * _ls
    _ls[9] = 0.9
    return _ls


def coords(theta, polarity):
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
    _Xs = np.array([coords(_t, polarity) for _t in thetas])
    _ms = np.array([prior_mean(_t, polarity) for _t in thetas])
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
    _Xs = np.array([coords(_t, polarity) for _t in thetas])
    _ms = np.array([prior_mean(_t, polarity) for _t in thetas])
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
def sobol_block(n_log2, offset_blocks):
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


class _CandidateSet:
    """Candidates in proposal order, deduplicated, each already known to be legible.

    Proposals arrive in batches because realizing a theme costs one colour-library call per
    batch rather than per theme (measured: a call converting one colour costs 312 us, and a
    call converting sixty-four costs 325 us). Realizing four hundred candidates one at a
    time spent 3.8 s of a 4 s trial inside that library's argument validation.

    Order is preserved and duplicates are dropped on first sight, because the index where
    the standing stratum ends is what the explore/exploit split is declared against.
    """

    def __init__(self, polarity):
        self.polarity = polarity
        self.entries = []
        self._seen = set()

    def offer(self, thetas, themes=None):
        """Add each theta that is new and legible. `themes` skips realization for the pool,
        whose members are realized once at startup and never change."""
        thetas = [np.clip(np.asarray(t, dtype=float), 0.0, 1.0) for t in thetas]
        fresh = [t for t in thetas if tuple(np.round(t, 4)) not in self._seen]
        if themes is None:
            themes = realize_many(np.array(fresh), self.polarity) if fresh else []
            offered = zip(fresh, themes, strict=True)
        else:
            offered = zip(thetas, themes, strict=True)
        for theta, theme in offered:
            key = tuple(np.round(theta, 4))
            if theme is None or key in self._seen:
                continue
            self._seen.add(key)
            self.entries.append((theta, theme))

    @property
    def thetas(self):
        return [theta for theta, _theme in self.entries]


def candidates(fit, polarity, nprng, n_trial=0, n_elite=10, n_mut=20, n_cross=48, imm_log2=6):
    """(candidates, index where the standing global stratum ends) for this trial."""
    pool = _CandidateSet(polarity)
    pool.offer([t for t, _ in POOL[polarity]], [theme for _, theme in POOL[polarity]])
    pool.offer(list(sobol_block(imm_log2, n_trial)))
    n_standing = len(pool.entries)
    if fit is None:
        return pool.entries, n_standing

    want = 1.0 if polarity == "night" else 0.0
    archive = [x[:9] for x in fit["X"] if abs(x[9] - want) < 0.5]
    seed_set = archive + pool.thetas
    mu = posterior_over(fit, seed_set, polarity)[0]
    ls = fit.get("ls")
    top = np.argsort(-mu)[: 6 * n_elite]
    # Elites for spread as well as for mean: the best few, then the most different among
    # the rest of the leaders, so refinement is not confined to one basin. Deliberately NOT
    # Thompson-sampled elites: tried, and measured clearly worse (reach 3/12 runs,
    # t = -2.6). Refining around a high-variance region spends the mutation budget on noise
    # and displaces elites that are actually good; explore belongs in the standing stratum,
    # refine belongs where the mean is already high.
    keep = [int(i) for i in top[: n_elite // 2]]
    weights = 1.0 / (LS0[:9] if ls is None else ls[:9])
    scaled = np.array([np.asarray(seed_set[int(i)]) * weights for i in top])
    top_list = list(top)
    while len(keep) < n_elite and len(keep) < len(top):
        chosen = [top_list.index(i) for i in keep if i in top_list] or [0]
        spread = np.min(np.linalg.norm(scaled[:, None, :] - scaled[None, chosen, :], axis=-1), axis=1)
        spread[chosen] = -1.0
        keep.append(int(top[int(np.argmax(spread))]))
    elites = [np.asarray(seed_set[i]) for i in keep]

    # Mutation sigma scales with the ARD length-scale per axis: fine steps where utility
    # actually turns, coarse where the model has learned that nothing rides.
    sigma = 0.25 * (LS0[:9] if ls is None else ls[:9])
    bred = []
    for elite in elites:
        bred.append(elite)
        bred.extend(np.clip(elite[None, :] + nprng.normal(0, sigma, (n_mut, 9)), 0, 1))
    if len(elites) >= 2:
        # Uniform per-axis recombination. Worth having because the axes are semi-separable
        # -- ground, accent set, comment recession, find-highlight -- so a good ground and a
        # good accent set recombine into a plausible page, which is the building-block case
        # where crossover earns its keep rather than adding noise.
        for _ in range(n_cross):
            first, second = nprng.choice(len(elites), 2, replace=False)
            mask = nprng.random(9) < 0.5
            bred.append(np.where(mask, elites[first], elites[second]))
    pool.offer(bred)
    return pool.entries, n_standing


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


BEST_MEMO = {}


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
    if _ck in BEST_MEMO:
        return BEST_MEMO[_ck]
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
    _w_ax = 1.0 / (LS0[:9] if fit.get("ls") is None else fit["ls"][:9])
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
    if len(BEST_MEMO) > 8:
        BEST_MEMO.pop(next(iter(BEST_MEMO)))
    BEST_MEMO[_ck] = _res
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
    _w = 1.0 / (LS0[:9] if ls is None else ls[:9])
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
SURF_MEMO = {}


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
    if _key in SURF_MEMO:
        return SURF_MEMO[_key]
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
    SURF_MEMO[_key] = _out
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
