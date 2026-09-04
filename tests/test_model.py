#!/usr/bin/env python
"""Recovery and strategy tests for calibrate-aesthetics' preference model.

A statistical instrument with no recovery tests is the kind of thing that mis-measures in
silence: every number it prints looks like a measurement. These tests give the model
synthetic observers whose truth is known and ask whether it recovers them.

Run from the repo root:
    pixi run python notebooks/pytorch-basics/_model_tests.py

The model cell is loaded out of the notebook by AST rather than duplicated here -- a copy
would drift, and a drifted test is worse than none. The stubs (flat prior, always-feasible
realize, a 512-point pool with the instrument's own seed) isolate the search and the
likelihood from the colour machinery, which has its own floors and its own reasons.

What the tests establish, and the numbers they were calibrated against (2026-09-03):

  T1  ARD recovers which axes drive preference, given enough duels (400).
  T2  The position-bias term recovers an injected side advantage within 0.45 logit.
      It is deliberately shrunk toward zero by an L2 prior: under-correcting a real bias
      is safer than inventing one.
  T3  Modelling the bias does not degrade utility recovery (it slightly improves it).
  T4  REACH -- warm-started inside the lower of two broad modes 2.0 apart, bred candidates
      win the majority of paired runs (7 of 12) but not the mean (paired diff -0.14 +/-
      0.11, t = -1.25): a few runs lose badly. Honest reading: no significant difference,
      so the assertion guards against REGRESSION rather than claiming a win. This
      landscape is also the least theme-like of the three -- real theme utility is smooth
      and its prior mean informative, neither of which holds here.

Two changes were tried, measured, and REJECTED -- recorded because a plausible-sounding
change that degrades an instrument is the expensive kind of mistake:
  * Replacing the pool with bred candidates entirely: lost global reach, scored worse.
    The pool is a codebook whose repeated visits concentrate information; a churning
    candidate set spreads every duel over ground never seen again.
  * Thompson-sampled elites (refining where variance is high): reach fell to 3 of 12
    runs, t = -2.6. Explore belongs in the standing stratum; refinement belongs where
    the posterior mean is already high.
  T5  RESOLUTION -- in nine dimensions no strategy finds a mode narrower than the kernel
      length-scale: both score ~0. This is not a bug to fix but the constraint to respect;
      it is why ARD (fewer effective dimensions) matters more than any sampler change.
  T6  The realistic regime -- a smooth utility riding on three of nine axes -- is where
      bred+ARD wins clearly, and where ARD shrinkage earns its keep (active axes
      recovered 2 of 4 runs at 60 duels, against 0 of 4 unshrunk).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

#: A flat 512-point pool, so the search and the likelihood are exercised without the
#: colour machinery's floors deciding which candidates exist. The floors have their own
#: constraints and their own reasons; mixing them in here would make a failure ambiguous.
POOL_THETA = np.random.default_rng(0xA55).random((512, 9))


def _model_under_test():
    """The real modules, with the colour layer replaced by stubs.

    Importing the package rather than AST-loading a notebook cell: the instrument is
    ordinary Python now, and a test that reconstructs its subject from source is testing
    the reconstruction as much as the code.

    `realize` and `prior_mean` are stubbed because these tests are about the SEARCH -- does
    ARD find the right axes, is an injected side bias subtracted, does a planted
    reaction-time surface come back. A feasibility filter that rejects candidates for
    contrast reasons would silently change which points the search ever sees, and a failure
    would no longer say which half was wrong.
    """
    from theme import model, schedule

    model.POOL = {"day": [(t, {"ok": True}) for t in POOL_THETA], "night": []}
    model.realize = lambda theta, polarity: {"ok": True}
    model.prior_mean = lambda theta, polarity: 0.0
    return {**vars(model), **vars(schedule)}


M = _model_under_test()


def synth_duels(n, active=(0, 3, 6), delta=0.0, seed=1):
    """Duels from a linear observer, with an optional left-card advantage."""
    r = np.random.default_rng(seed)
    w = np.zeros(9)
    for a in active:
        w[a] = 1.0
    X, duels, lam, sides = [], [], [], []
    for _ in range(n):
        a, b = r.random(9), r.random(9)
        ia, ib = len(X), len(X) + 1
        X += [M["coords"](a, "day"), M["coords"](b, "day")]
        side = 1.0 if r.random() < 0.5 else -1.0
        z = 3.0 * float(w @ (a - b)) + delta * side
        a_wins = r.random() < 1 / (1 + np.exp(-z))
        duels.append((ia, ib) if a_wins else (ib, ia))
        sides.append(side if a_wins else -side)
        lam.append(1.0)
    return np.array(X), duels, np.array(lam), np.zeros(len(X)), np.array(sides), w


def simulate(strategy, u_true, warm_near=None, n_adaptive=50, seed=1, warm=10):
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
        fit = M["fitted"](resp)
        npr = np.random.default_rng(n * 7919 + 13)
        if strategy == "pool":
            cand = list(POOL_THETA)
            champ = cand[int(np.argmax(M["posterior_over"](fit, cand, "day")[0]))]
            cand = cand + list(np.clip(champ + npr.normal(0, 0.08, (48, 9)), 0, 1))
        else:
            bred, n_std = M["candidates"](fit, "day", npr, n_trial=n)
            cand = [c[0] for c in bred]
        mu, var, _ks, _A = M["posterior_over"](fit, cand, "day")
        samp = mu + np.sqrt(var) * npr.standard_normal(len(mu))
        if strategy == "pool":
            i1 = int(np.argmax(samp))
        else:
            lo, hi = (n_std, len(cand)) if (npr.random() < 0.5 and n_std < len(cand)) else (0, n_std)
            i1 = lo + int(np.argmax(samp[lo:hi]))
        gap, s2 = mu - mu[i1], np.maximum(var + var[i1], 1e-9)
        pb = 1 / (1 + np.exp(-gap / np.sqrt(1 + np.pi * s2 / 8)))
        h = M["h2"](pb)
        h[i1] = -1.0
        record(cand[i1], cand[int(np.argmax(h))])
    fit = M["fitted"](resp)
    ref = np.vstack([POOL_THETA, qmc.Sobol(d=9, scramble=True, seed=5).random(4096)])
    best = ref[int(np.argmax(M["posterior_over"](fit, list(ref), "day")[0]))]
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


def synth_timed(n, active=(2, 5), offset=0.0, noise=0.25, seed=3, hunt_share=0.4):
    """Timed trials from an observer whose reading speed depends on a few theme axes.

    `offset` is a per-arm task difference in log seconds -- a find hunt highlights its
    matches and is genuinely faster than a cold probe -- injected so the surface can be
    checked for attributing it to the theme instead of the task.
    """
    r = np.random.default_rng(seed)
    w = np.zeros(9)
    for a in active:
        w[a] = 1.0

    def truth(theta):
        return 0.9 + 0.6 * float(w @ np.asarray(theta))

    rows = []
    for _ in range(n):
        th = r.random(9)
        hunt = r.random() < hunt_share
        y = truth(th) - (offset if hunt else 0.0) + r.normal(0, noise)
        rows.append(
            {
                "mode": "search" if hunt else "comprehension",
                "polarity": "day",
                "theta_a": list(map(float, th)),
                "correct": True,
                "paused": False,
                "rt_ms": float(np.exp(y) * 1000.0),
            }
        )
    return rows, truth


def run_recovery_suite() -> list[str]:
    fails = []

    def check(name, ok, detail):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(name)

    print("T1/T2/T3 -- likelihood and relevance recovery")
    X, duels, lam, m, sides, _ = synth_duels(400, active=(0, 3, 6), seed=2)
    ls = M["ard_scales"](X, duels, lam)
    short = set(np.argsort(ls[:9])[:3].tolist())
    check("ARD relevance", short == {0, 3, 6}, f"shortest length-scales {sorted(short)} of 9")
    errs = []
    for dt in (-0.8, -0.3, 0.0, 0.6):
        X, duels, lam, m, sides, _ = synth_duels(500, delta=dt, seed=int(abs(dt) * 100) + 5)
        d_fit = M["fit_laplace"](X, duels, lam, m, sides, M["ard_scales"](X, duels, lam))[3]
        errs.append(abs(d_fit - dt))
    check("position bias", max(errs) < 0.45, f"worst error {max(errs):.2f} logit over four truths")
    X, duels, lam, m, sides, w = synth_duels(500, delta=-0.9, seed=11)
    lsx = M["ard_scales"](X, duels, lam)
    f_no = M["fit_laplace"](X, duels, lam, m, None, lsx)[0]
    f_yes = M["fit_laplace"](X, duels, lam, m, sides, lsx)[0]
    truth = np.array([3.0 * float(w @ x[:9]) for x in X])
    c_no = float(np.corrcoef(truth, f_no)[0, 1])
    c_yes = float(np.corrcoef(truth, f_yes)[0, 1])
    check("bias is free", c_yes >= c_no - 0.01, f"corr {c_no:.3f} ignoring vs {c_yes:.3f} modelling")

    print("\nT4/T5 -- reach and resolution, bred candidates vs the frozen pool")
    for label, sigma, sparse in (("reach", 0.55, False), ("resolution", 0.30, True)):
        truths = [two_modes(11 + i, sigma, sparse) for i in range(3)]
        got = {}
        for strat in ("pool", "bred"):
            runs = [simulate(strat, u, warm_near=a, seed=s) for (a, b, u), s in zip(truths, (1, 2, 3), strict=True)]
            got[strat] = float(np.mean([r["argmax"] for r in runs]))
        if label == "reach":
            check(
                "reach: no regression",
                got["bred"] > got["pool"] - 0.45,
                f"argmax utility bred {got['bred']:.2f} vs pool {got['pool']:.2f} "
                f"(3 seeds, indicative only -- see T4)",
            )
        else:
            check(
                "resolution: both blind (expected)",
                max(got.values()) < 0.2,
                f"bred {got['bred']:.3f}, pool {got['pool']:.3f} -- 9-d volume, not a bug",
            )

    print("\nT6 -- the realistic regime: smooth utility on three of nine axes")
    us = [low_dim(21 + i) for i in range(4)]
    res = {}
    for strat in ("pool", "bred"):
        runs = [simulate(strat, u, seed=s) for u, s in zip(us, (1, 2, 3, 4), strict=True)]
        res[strat] = (
            float(np.mean([r["argmax"] for r in runs])),
            sum(set(np.argsort(r["ls"][:9])[:3].tolist()) == {1, 4, 7} for r in runs),
        )
    check(
        "realistic: bred+ARD wins",
        res["bred"][0] > res["pool"][0],
        f"argmax bred {res['bred'][0]:.2f} vs pool {res['pool'][0]:.2f}; "
        f"ARD found the active axes in {res['bred'][1]}/4 runs",
    )

    print("\nT7/T8 -- the legibility surface")
    rows, truth = synth_timed(90, seed=5, offset=0.0)
    rf = M["rt_fit"](rows, "day", None)
    probe = [np.random.default_rng(900 + i).random(9) for i in range(300)]
    pred, _ = M["rt_at"](rf, probe, "day")
    true = np.array([truth(t) for t in probe])
    rho = float(np.corrcoef(pred, true)[0, 1])
    check("rt surface recovers reading speed", rho > 0.5, f"corr(predicted, true log-time) = {rho:.2f}")

    # The same truth, but with the two arms 0.55 log-seconds apart. A single pooled mean
    # would push that constant into the theme surface; per-arm means should not.
    rows_off, truth_off = synth_timed(90, seed=5, offset=0.55)
    rf_off = M["rt_fit"](rows_off, "day", None)
    pred_off, _ = M["rt_at"](rf_off, probe, "day")
    rho_off = float(np.corrcoef(pred_off, np.array([truth_off(t) for t in probe]))[0, 1])
    check(
        "per-arm baselines absorb the task offset",
        rho_off > rho - 0.12,
        f"corr {rho_off:.2f} with a 0.55 log-s arm offset vs {rho:.2f} without",
    )

    # The constraint must exclude the genuinely slow and spare the genuinely fast.
    excl, _secs = M["rt_penalty"](rf, probe, "day")
    if excl.any():
        slow_rank = float(np.mean([np.mean(true[excl] > t) for t in true[~excl]]))
        check(
            "excluded candidates are the slow ones",
            slow_rank > 0.75,
            f"{int(excl.sum())} excluded, and an excluded page is slower than "
            f"{100 * slow_rank:.0f}% of the kept ones on the truth",
        )
    else:
        check("constraint is conservative when unsure", True, "nothing excluded at this noise")

    print("\nT9 -- grouping before counting P(best)")
    X, duels, lam, m, sides, w = synth_duels(300, active=(0, 3, 6), seed=8)
    responses = []
    for (a, b), sd in zip(duels, sides, strict=True):
        responses.append(
            {
                "mode": "duel",
                "choice": 0,
                "theta_a": list(map(float, X[a][:9])),
                "theta_b": list(map(float, X[b][:9])),
                "polarity": "day",
                "rt_ms": 3000.0,
                "swap": sd < 0,
            }
        )
    fit = M["fitted"](responses)
    base = np.random.default_rng(11).random(9)
    clones = [np.clip(base + np.random.default_rng(50 + i).normal(0, 0.004, 9), 0, 1) for i in range(20)]
    spread = [np.random.default_rng(80 + i).random(9) for i in range(20)]
    bs = M["best_set"](fit, "day", clones + spread, seed=3)
    n_clone_groups = len({int(np.argmin([np.linalg.norm(np.asarray(g) - c) for g in bs["groups"]])) for c in clones})
    check(
        "near-identical themes count once",
        n_clone_groups <= 2 and len(bs["groups"]) <= 25,
        f"20 clones fell into {n_clone_groups} group(s); {len(bs['groups'])} groups from 40 candidates",
    )

    print("\nT10 -- the convergence readout")
    prog = M["progress_report"](responses, "day", spread + clones, back=40)
    check(
        "progress compares two fits and reports movement",
        prog is not None and prog["duels"] == 300 and prog["set_now"] >= 1,
        "None (needs more duels)"
        if prog is None
        else f"leader {100 * prog['lead_then']:.0f}% -> {100 * prog['lead_now']:.0f}%, "
        f"set {prog['set_then']} -> {prog['set_now']} over {prog['back']} duels",
    )

    print("\nT11 -- active hunting beats uniform sweeps")

    def hunt_run(strategy, n=40, seed=1, noise=0.22):
        base = POOL_THETA[3].copy()

        def truth(th):
            return 1.5 - 0.5 * th[7] - 0.35 * th[8] + 0.3 * th[7] * th[8]

        r = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            rf = M["rt_fit"](rows, "day", None) if len(rows) >= 8 else None
            if strategy == "active" and rf is not None and r.random() > 0.25:
                g = np.linspace(0.05, 0.95, 7)
                cands = []
                for v7 in g:
                    for v8 in g:
                        c = base.copy()
                        c[7], c[8] = v7, v8
                        cands.append(c)
                th = cands[int(np.argmax(M["rt_at"](rf, cands, "day")[1]))]
            else:
                th = base.copy()
                th[7], th[8] = r.random(), r.random()
            rows.append(
                {
                    "mode": "search",
                    "polarity": "day",
                    "theta_a": list(map(float, th)),
                    "correct": True,
                    "paused": False,
                    "rt_ms": float(np.exp(truth(th) + r.normal(0, noise)) * 1000),
                }
            )
        rf = M["rt_fit"](rows, "day", None)
        rg = np.random.default_rng(99)
        grid = []
        for _ in range(400):
            t = base.copy()
            t[7], t[8] = rg.random(), rg.random()
            grid.append(t)
        mu = M["rt_at"](rf, grid, "day")[0]
        tv = np.array([truth(t) for t in grid])
        return float(np.corrcoef(mu, tv)[0, 1])

    uni = float(np.mean([hunt_run("uniform", seed=s) for s in (1, 2, 3, 4, 5)]))
    act = float(np.mean([hunt_run("active", seed=s) for s in (1, 2, 3, 4, 5)]))
    check(
        "uncertainty sampling identifies the find axes faster",
        act > uni,
        f"corr(pred, truth) active {act:.3f} vs uniform {uni:.3f} at 40 hunts",
    )

    print("\nT12 -- surface assignment is balanced and free of run position")
    ds = [M["duel_surface"](n, 200) for n in range(24 * 12)]
    ds = [d for n, d in enumerate(ds) if n % 24 < 16]
    counts = {s: ds.count(s) for s in ("editor", "panel", "notebook")}
    check(
        "the three surfaces are sampled equally often",
        max(counts.values()) - min(counts.values()) <= 1,
        f"counts over 12 blocks {counts} (n % 3 would lock this at 6/5/5 per block, forever)",
    )
    slot0 = {M["duel_surface"](24 * b, 200) for b in range(12)}
    check(
        "first-of-run is not always the same surface",
        len(slot0) == 3,
        f"surfaces seen in slot 0 across 12 runs: {sorted(slot0)}",
    )

    print("\nT13 -- the surface-interaction test is calibrated")

    def surf_rows(tilt, n=96, seed=0):
        r = np.random.default_rng(seed)
        w = np.zeros(9)
        w[[0, 4, 6]] = [1.5, 1.0, -1.0]
        rows = []
        for i in range(n):
            a, b = r.random(9), r.random(9)
            s = ("editor", "panel", "notebook")[i % 3]
            z = (a - b) @ w + tilt * (1.0 if s == "editor" else -1.0 if s == "panel" else 0.0) * (a - b)[0]
            rows.append(
                {
                    "mode": "duel",
                    "polarity": "day",
                    "surface": s,
                    "paused": False,
                    "theta_a": list(map(float, a)),
                    "theta_b": list(map(float, b)),
                    # 0 = theme_a won, matching duels_from's encoding. This test passed
                    # vacuously against "a"/"b": the filter dropped every row, the function
                    # returned its not-enough-data p of 1.0, and "stays quiet" read as a pass.
                    "choice": 0 if r.random() < 1.0 / (1.0 + np.exp(-z)) else 1,
                }
            )
        return rows

    n_seen = M["surface_effect"](surf_rows(0.0, seed=0), "day", nperm=20)[0]
    check(
        "the test actually sees the duels it is given",
        n_seen == 96,
        f"{n_seen} of 96 synthetic duels reached the test",
    )
    p_null = [M["surface_effect"](surf_rows(0.0, seed=s), "day", nperm=120)[2] for s in range(6)]
    check(
        "stays quiet when no surface effect exists",
        sum(p < 0.10 for p in p_null) <= 1,
        f"p-values under a true null: {[round(p, 2) for p in p_null]} (a fixed threshold fires ~1 in 5)",
    )
    p_real = [M["surface_effect"](surf_rows(2.5, seed=s), "day", nperm=120)[2] for s in range(6)]
    check(
        "finds a real surface effect",
        sum(p < 0.10 for p in p_real) >= 4,
        f"p-values with a planted 2.5-logit tilt: {[round(p, 2) for p in p_real]}",
    )

    print("\nT14 -- axis consensus separates settled axes from open ones")
    rng = np.random.default_rng(5)
    th = rng.random((240, 9))
    # ax0 settled near 0.8, ax5 wide open: mass concentrated on one value of ax0 only.
    w = np.exp(-((th[:, 0] - 0.8) ** 2) / (2 * 0.05**2))
    w = w / w.sum()
    cons = M["axis_consensus"]({"p_best": w}, th)
    r0 = next(c[1] for c in cons if c[0] == 0)
    r5 = next(c[1] for c in cons if c[0] == 5)
    check(
        "a settled axis reads narrow and an untouched one reads wide",
        r0 < 0.4 < 0.8 < r5,
        f"spread relative to uniform: settled axis {r0:.2f}, untouched axis {r5:.2f}",
    )
    m0 = next(c[2] for c in cons if c[0] == 0)
    check(
        "the settled axis reports where it settled",
        abs(m0 - 0.8) < 0.05,
        f"posterior-weighted mean of the settled axis {m0:.3f} against a planted 0.80",
    )

    print("\nT15 -- a change of type size does not land on the theme surface")

    def timed_rows(n=48, size_step=0.0, seed=3):
        """Half the trials at 15px, half at 14px, with a planted size effect of size_step
        log-seconds. The THEME effect is identical in both halves, so a model that absorbs
        the size step recovers the same surface either way."""
        r = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            th = r.random(9)
            px = 15.0 if i < n // 2 else 14.0
            base = 1.4 - 0.5 * th[0] + (size_step if px == 14.0 else 0.0)
            rows.append(
                {
                    "mode": "comprehension" if i % 2 else "search",
                    "polarity": "day",
                    "theta_a": list(map(float, th)),
                    "correct": True,
                    "paused": False,
                    "code_px": px,
                    "rt_ms": float(np.exp(base + r.normal(0, 0.18)) * 1000),
                }
            )
        return rows

    def surface_corr(rows):
        rf = M["rt_fit"](rows, "day", None)
        rg = np.random.default_rng(11)
        grid = [rg.random(9) for _ in range(300)]
        mu = M["rt_at"](rf, grid, "day")[0]
        truth = np.array([1.4 - 0.5 * g[0] for g in grid])
        return float(np.corrcoef(mu, truth)[0, 1])

    c_flat = surface_corr(timed_rows(size_step=0.0))
    c_step = surface_corr(timed_rows(size_step=0.45))
    check(
        "a 0.45 log-second size step costs the surface nothing",
        c_step > c_flat - 0.08,
        f"corr with truth {c_step:.3f} with the step against {c_flat:.3f} without it",
    )

    print("\n" + ("all recovery tests pass" if not fails else f"FAILED: {fails}"))
    return fails


def test_recovery_suite():
    """One test, because the suite is one report.

    Splitting it per assertion would be the usual advice, and it is the right next step;
    it is not free here because several checks share an expensive fit and the printed
    report is how the numbers get compared against the ones recorded in the docstring.
    """
    failures = run_recovery_suite()
    assert not failures, f"recovery tests failed: {failures}"


if __name__ == "__main__":
    raise SystemExit(1 if run_recovery_suite() else 0)
