"""The readouts that say whether the answer is settled -- and whether to believe them.

P(best) as a distribution over argmaxes and its credible set; which axes the clicks have
settled; whether another sitting is worth clicking; and a permutation test for whether
the preferred theme depends on some logged property of how it was shown.

Every one of these prints a number that looks like a measurement, so each carries the
calibration that says when to believe it.
"""

import numpy as np

from .kernel import LS0
from .preference import fitted, posterior_joint

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
    # nperm and seed belong in the key. Without them a coarse call -- a quick 20-permutation
    # sanity check, say -- poisons the cache for the careful 200-permutation reading that
    # follows, and the caller gets a p-value computed against a null it never asked for.
    # A cache key must name every input that changes the answer.
    _key = (
        "f",
        key,
        polarity,
        nperm,
        seed,
        hash(tuple((_r["choice"], str(_r[key]), _r["theta_a"][0]) for _r in _ds)),
    )
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
