"""Does the preference model recover a planted truth?

A statistical instrument with no recovery tests is the kind of thing that mis-measures in
silence: every number it prints looks like a measurement. These tests give the Bradley-Terry
GP synthetic observers whose truth is known and ask whether it hands the truth back.

The model is imported, not reconstructed. It used to live in a notebook cell that a
standalone script AST-loaded rather than imported; the instrument is ordinary Python now,
so these run with the rest of the suite (`pixi run test`) against the real module -- a test
that rebuilds its subject from source is testing the reconstruction as much as the code.

The colour layer is stubbed out by the `search_model` fixture, because everything here is
about the SEARCH -- does ARD find the right axes, is an injected side bias subtracted. A
feasibility filter rejecting candidates for contrast reasons would silently change which
points the search ever sees, and a failure would no longer say which half was wrong.

Numbers quoted in the test docstrings below were calibrated 2026-09-03.

Two changes were tried, measured, and REJECTED -- recorded because a plausible-sounding
change that degrades an instrument is the expensive kind of mistake:

  * Replacing the pool with bred candidates entirely: lost global reach, scored worse.
    The pool is a codebook whose repeated visits concentrate information; a churning
    candidate set spreads every duel over ground never seen again.
  * Thompson-sampled elites (refining where variance is high): reach fell to 3 of 12
    runs, t = -2.6. Explore belongs in the standing stratum; refinement belongs where
    the posterior mean is already high.
"""

import numpy as np
import pytest
from conftest import POOL_THETA
from scipy.stats import qmc

# Imported directly rather than through the `search_model` fixture: the fingerprint is a
# pure function of rows and touches none of the colour layer that fixture stubs out.
from theme.preference import log_fingerprint


def synth_duels(model, n, active=(0, 3, 6), delta=0.0, seed=1):
    """Duels from a linear observer, with an optional left-card advantage."""
    r = np.random.default_rng(seed)
    w = np.zeros(9)
    for a in active:
        w[a] = 1.0
    X, duels, lam, sides = [], [], [], []
    for _ in range(n):
        a, b = r.random(9), r.random(9)
        ia, ib = len(X), len(X) + 1
        X += [model.coords(a, "day"), model.coords(b, "day")]
        side = 1.0 if r.random() < 0.5 else -1.0
        z = 3.0 * float(w @ (a - b)) + delta * side
        a_wins = r.random() < 1 / (1 + np.exp(-z))
        duels.append((ia, ib) if a_wins else (ib, ia))
        sides.append(side if a_wins else -side)
        lam.append(1.0)
    return np.array(X), duels, np.array(lam), np.zeros(len(X)), np.array(sides), w


def duel_log(model, n, active=(0, 3, 6), seed=1):
    """The same duels as log rows -- the shape `fitted` reads.

    Imported by test_diagnostics, which needs a fitted model over a known truth for the
    same reason; a second copy of the generator would drift from this one.
    """
    X, duels, _lam, _m, sides, _w = synth_duels(model, n, active=active, seed=seed)
    return [
        {
            "mode": "duel",
            "choice": 0,
            "theta_a": list(map(float, X[a][:9])),
            "theta_b": list(map(float, X[b][:9])),
            "polarity": "day",
            "rt_ms": 3000.0,
            "swap": sd < 0,
        }
        for (a, b), sd in zip(duels, sides, strict=True)
    ]


def simulate(model, strategy, u_true, warm_near=None, n_adaptive=50, seed=1, warm=10):
    """A full preference-learning run against a synthetic observer."""
    r = np.random.default_rng(seed)
    start = (
        np.argsort(np.linalg.norm(POOL_THETA - warm_near, axis=1))[:40] if warm_near is not None else np.arange(512)
    )
    resp, shown = [], []

    def record(ta, tb):
        z = 3.0 * float(u_true(ta)[0] - u_true(tb)[0])
        shown.extend([np.asarray(ta), np.asarray(tb)])
        resp.append(
            {
                "mode": "duel",
                "choice": 0 if r.random() < 1 / (1 + np.exp(-z)) else 1,
                "theta_a": list(map(float, ta)),
                "theta_b": list(map(float, tb)),
                "polarity": "day",
                "rt_ms": 3000.0,
                "swap": bool(r.random() < 0.5),
            }
        )

    for _ in range(warm):
        i, j = r.choice(start, 2, replace=False)
        record(POOL_THETA[i], POOL_THETA[j])
    for n in range(n_adaptive):
        fit = model.fitted(resp)
        npr = np.random.default_rng(n * 7919 + 13)
        if strategy == "pool":
            cand = list(POOL_THETA)
            champ = cand[int(np.argmax(model.posterior_over(fit, cand, "day")[0]))]
            cand = cand + list(np.clip(champ + npr.normal(0, 0.08, (48, 9)), 0, 1))
        else:
            bred, n_std = model.candidates(fit, "day", npr, n_trial=n)
            cand = [c[0] for c in bred]
        mu, var, _ks, _A = model.posterior_over(fit, cand, "day")
        samp = mu + np.sqrt(var) * npr.standard_normal(len(mu))
        if strategy == "pool":
            i1 = int(np.argmax(samp))
        else:
            lo, hi = (n_std, len(cand)) if (npr.random() < 0.5 and n_std < len(cand)) else (0, n_std)
            i1 = lo + int(np.argmax(samp[lo:hi]))
        gap, s2 = mu - mu[i1], np.maximum(var + var[i1], 1e-9)
        pb = 1 / (1 + np.exp(-gap / np.sqrt(1 + np.pi * s2 / 8)))
        h = model.h2(pb)
        h[i1] = -1.0
        record(cand[i1], cand[int(np.argmax(h))])
    fit = model.fitted(resp)
    ref = np.vstack([POOL_THETA, qmc.Sobol(d=9, scramble=True, seed=5).random(4096)])
    best = ref[int(np.argmax(model.posterior_over(fit, list(ref), "day")[0]))]
    return {
        "shown": float(u_true(np.array(shown)).max()),
        "argmax": float(u_true(best)[0]),
        "ls": fit["ls"],
    }


def two_modes(seed, sigma, sparse=False):
    """A lower mode to start in and a better one elsewhere."""
    r = np.random.default_rng(seed)
    a = np.clip(r.random(9) * 0.35 + 0.05, 0, 1)
    if sparse:
        c = r.random((20000, 9))
        nn = np.min(np.linalg.norm(c[:, None, :] - POOL_THETA[None, :, :], axis=-1), axis=1)
        b = c[int(np.argmax(nn))]
    else:
        b = np.clip(1.0 - a, 0, 1)

    def u(T):
        T = np.atleast_2d(np.asarray(T, float))
        da = np.linalg.norm(T - a, axis=1)
        db = np.linalg.norm(T - b, axis=1)
        return 1.0 * np.exp(-(da**2) / (2 * sigma**2)) + 1.8 * np.exp(-(db**2) / (2 * sigma**2))

    return a, b, u


def low_dim(seed, active=(1, 4, 7)):
    """The realistic case: smooth utility riding on a few axes."""
    peak = np.random.default_rng(seed).random(len(active))

    def u(T):
        T = np.atleast_2d(np.asarray(T, float))[:, list(active)]
        return 1.8 * np.exp(-(np.linalg.norm(T - peak, axis=1) ** 2) / (2 * 0.45**2))

    return u


def mean_argmax(model, strategy, truths, seeds):
    """One run per truth; the mean utility of the theme each run ends up calling best.

    `truths` are (warm_start, utility) pairs -- the warm start is the mode a run begins
    inside, which is what makes the two-mode landscapes a test of REACH rather than of
    luck. None starts from the whole pool.
    """
    runs = [simulate(model, strategy, u, warm_near=w, seed=s) for (w, u), s in zip(truths, seeds, strict=True)]
    return float(np.mean([run["argmax"] for run in runs])), runs


def test_ard_finds_the_axes_that_drive_choices(search_model):
    """Relevance is recoverable given enough duels -- 400 here.

    It is not recoverable early, which is why `ard_scales` blends toward isotropy with
    weight n/160: at 60 duels the estimated ranking of nine axes was measured to be noise.
    """
    X, duels, lam, _m, _sides, _w = synth_duels(search_model, 400, active=(0, 3, 6), seed=2)
    ls = search_model.ard_scales(X, duels, lam)
    short = set(np.argsort(ls[:9])[:3].tolist())
    assert short == {0, 3, 6}, f"shortest length-scales {sorted(short)} of 9"


@pytest.mark.parametrize("delta", [-0.8, -0.3, 0.0, 0.6])
def test_the_position_bias_term_recovers_an_injected_side_advantage(search_model, delta):
    """Within 0.45 logit of the truth, over four truths (worst error measured: 0.23).

    The term is deliberately shrunk toward zero by an L2 prior: under-correcting a real
    bias is safer than inventing one, so the tolerance is one-sided in spirit even though
    the assertion is not.
    """
    X, duels, lam, m, sides, _w = synth_duels(search_model, 500, delta=delta, seed=int(abs(delta) * 100) + 5)
    fitted_delta = search_model.fit_laplace(X, duels, lam, m, sides, search_model.ard_scales(X, duels, lam))[3]
    err = abs(fitted_delta - delta)
    assert err < 0.45, f"recovered {fitted_delta:.2f} against a planted {delta:+.1f}: error {err:.2f} logit"


def test_modelling_the_position_bias_is_free(search_model):
    """Fitting the side term must not degrade utility recovery -- it slightly improves it.

    Measured with a 0.9-logit bias planted: correlation with the true utility 0.898
    ignoring the bias against 0.901 modelling it. The tolerance allows a 0.01 loss, so
    this is a guard against the extra parameter eating signal, not a claim of a win.
    """
    X, duels, lam, m, sides, w = synth_duels(search_model, 500, delta=-0.9, seed=11)
    lsx = search_model.ard_scales(X, duels, lam)
    f_no = search_model.fit_laplace(X, duels, lam, m, None, lsx)[0]
    f_yes = search_model.fit_laplace(X, duels, lam, m, sides, lsx)[0]
    truth = np.array([3.0 * float(w @ x[:9]) for x in X])
    c_no = float(np.corrcoef(truth, f_no)[0, 1])
    c_yes = float(np.corrcoef(truth, f_yes)[0, 1])
    assert c_yes >= c_no - 0.01, f"corr {c_no:.3f} ignoring the bias vs {c_yes:.3f} modelling it"


def test_bred_candidates_do_not_lose_reach_against_the_frozen_pool(search_model):
    """REACH -- warm-started inside the lower of two broad modes 2.0 apart.

    Bred candidates win the majority of paired runs (7 of 12) but not the mean (paired
    diff -0.14 +/- 0.11, t = -1.25): a few runs lose badly. Honest reading: no significant
    difference, so this assertion guards against REGRESSION rather than claiming a win,
    and the three seeds here are indicative only (measured bred 0.78 against pool 0.71) --
    the 12-run comparison is what the numbers above come from.

    This landscape is also the least theme-like of the three: real theme utility is smooth
    and its prior mean informative, neither of which holds here.
    """
    truths = [(a, u) for a, _b, u in (two_modes(11 + i, 0.55, False) for i in range(3))]
    got = {strategy: mean_argmax(search_model, strategy, truths, (1, 2, 3))[0] for strategy in ("pool", "bred")}
    assert got["bred"] > got["pool"] - 0.45, (
        f"argmax utility bred {got['bred']:.2f} vs pool {got['pool']:.2f} (3 seeds, indicative only)"
    )


def test_no_strategy_resolves_a_mode_narrower_than_the_kernel(search_model):
    """RESOLUTION -- a 0.30-wide mode placed where the pool is sparsest.

    In nine dimensions no strategy finds it: both score ~0 (measured bred 0.024, pool
    0.016). This is not a bug to fix but the constraint to respect -- it is why ARD, which
    cuts the effective dimension, matters more than any sampler change.
    """
    truths = [(a, u) for a, _b, u in (two_modes(11 + i, 0.30, True) for i in range(3))]
    got = {strategy: mean_argmax(search_model, strategy, truths, (1, 2, 3))[0] for strategy in ("pool", "bred")}
    assert max(got.values()) < 0.2, f"bred {got['bred']:.3f}, pool {got['pool']:.3f} -- 9-d volume, not a bug"


def test_the_realistic_regime_favours_bred_candidates(search_model):
    """A smooth utility riding on three of nine axes -- the case the real instrument is in.

    This is where bred+ARD wins clearly (measured argmax 1.61 against the pool's 1.45), and
    where ARD shrinkage earns its keep: the active axes were recovered in 2 of 4 runs at 60
    duels, against 0 of 4 unshrunk.
    """
    truths = [(None, low_dim(21 + i)) for i in range(4)]
    res = {}
    for strategy in ("pool", "bred"):
        mean, runs = mean_argmax(search_model, strategy, truths, (1, 2, 3, 4))
        found = sum(set(np.argsort(run["ls"][:9])[:3].tolist()) == {1, 4, 7} for run in runs)
        res[strategy] = (mean, found)
    assert res["bred"][0] > res["pool"][0], (
        f"argmax bred {res['bred'][0]:.2f} vs pool {res['pool'][0]:.2f}; "
        f"ARD found the active axes in {res['bred'][1]}/4 runs"
    )


def clone_and_spread_candidates():
    """20 near-identical themes and 20 spread across the cube.

    The clones sit within 0.004 of one another on every axis -- far below anything an eye
    could see -- so they are one theme as far as any verdict is concerned. Imported by
    test_diagnostics, whose progress readout needs the same fixed candidate set.

    Clones come FIRST in the candidate list, because the grouping tests read the group of
    candidate i for i below len(clones).
    """
    base = np.random.default_rng(11).random(9)
    clones = [np.clip(base + np.random.default_rng(50 + i).normal(0, 0.004, 9), 0, 1) for i in range(20)]
    spread = [np.random.default_rng(80 + i).random(9) for i in range(20)]
    return clones, spread


@pytest.fixture(scope="module")
def grouped_best_set(search_model):
    """P(best) over the clones and the spread candidates together."""
    fit = search_model.fitted(duel_log(search_model, 300, seed=8))
    clones, spread = clone_and_spread_candidates()
    return search_model.best_set(fit, "day", clones + spread, seed=3), clones


def test_near_identical_themes_count_once(grouped_best_set):
    """Grouping before counting, or every sibling steals argmax mass from the others.

    On the real log that failure read as a plateau while the leader held 1.6% -- a number
    that says nothing about whether one theme leads.

    Read off `group_of`, which is the assignment the grouping actually made. An earlier
    version of this assertion took the nearest of `groups` -- a list of candidate INDICES
    -- to each clone VECTOR, so it compared an integer against nine coordinates. Every
    clone got the same meaningless answer, the count came out at 1, and the test passed
    without touching the grouping at all.
    """
    best, clones = grouped_best_set
    clone_groups = {int(best["group_of"][i]) for i in range(len(clones))}
    assert len(clone_groups) <= 2, f"{len(clones)} clones fell into {len(clone_groups)} group(s)"


def test_p_best_counts_contenders_rather_than_coordinates(grouped_best_set):
    """40 candidates, 21 of them perceptually distinct, must not come back as 40 contenders.

    The verdict is read off cumulative group mass, so an inflated group count inflates the
    plateau: "single", "plateau" and "undecided" mean different things only if the groups
    really are groups.
    """
    best, _clones = grouped_best_set
    assert len(best["groups"]) <= 25, f"{len(best['groups'])} groups from 40 candidates"


def test_the_fit_memo_names_the_log_it_cached(search_model):
    """Two different logs of the same length must not be served one fit.

    The memo key was (duel count, RT exponent): how MUCH data was fitted, never WHICH. Two
    logs of equal length collide, and the second caller gets the first log's utilities back
    with nothing to say they are not its own. Reachable from `progress_report`, which fits
    a truncated history alongside the full one, and from any analysis that compares two
    logs -- and the returned object looks exactly like a measurement either way.
    """
    search_model.FIT_MEMO.clear()
    search_model.RTP_MEMO.clear()
    first = search_model.fitted(duel_log(search_model, 60, active=(0, 3, 6), seed=1))
    second = search_model.fitted(duel_log(search_model, 60, active=(2, 5, 8), seed=2))
    assert not np.array_equal(first["f"], second["f"]), "two different 60-duel logs were served one fit"


def test_the_fit_fingerprint_is_the_same_in_every_process():
    """A fingerprint that is written down must not be salted per process.

    This value does not stay inside the process: `verdict.provenance` puts it in every
    published palette and every response row carries it, so re-reading a year of
    measurements means asking which fit a row was taken under. The builtin `hash()` salts
    strings per run, so the same log fingerprinted differently every time -- an identifier
    that changes when nothing changed, which is worse than none, because it looks like one.

    Pinned as a literal rather than compared against itself: a self-comparison inside one
    process passes with the salted hash too, which is exactly the bug that would be missed.
    """
    rows = [
        {
            "choice": 0,
            "polarity": "day",
            "theta_a": [0.1, 0.2],
            "theta_b": [0.3, 0.4],
            "rt_ms": 1234.5,
            "paused": False,
            "swap": True,
        }
    ]
    assert log_fingerprint(rows) == "83fcd5ee5829a592"


def test_the_fingerprint_moves_when_any_field_it_names_moves():
    """Complete, not merely stable: a key must name every input that changes the answer."""
    base = {
        "choice": 0,
        "polarity": "day",
        "theta_a": [0.1, 0.2],
        "theta_b": [0.3, 0.4],
        "rt_ms": 1234.5,
        "paused": False,
        "swap": True,
    }
    reference = log_fingerprint([base])
    for field, value in [
        ("choice", 1),
        ("polarity", "night"),
        ("theta_a", [0.9, 0.2]),
        ("theta_b", [0.3, 0.9]),
        ("rt_ms", 1234.6),
        ("paused", True),
        ("swap", False),
    ]:
        assert log_fingerprint([{**base, field: value}]) != reference, f"changing {field} left the fingerprint unmoved"
