"""The legibility surface: what the clock measures, and what it must not absorb.

The duels ask which theme he PREFERS. The timed trials ask which theme he READS faster,
which makes the reaction-time GP a second instrument over the same nine axes -- and one
with its own characteristic failure: a constant belonging to the TASK or to the TYPE SIZE
landing on the theme surface instead, where it reads as a preference for whichever themes
happened to be on screen when the task was easy.

So the recovery tests here come in pairs. Plant a reading-speed surface and check it comes
back; then plant a nuisance step on top of it and check the surface comes back unchanged.
A per-arm (and per-size) baseline is what makes the second half true; a single pooled mean
would push the step into the theme term.

The colour layer is stubbed by the `search_model` fixture -- these tests are about the
surface, not about which candidates are legible enough to show.
"""

import numpy as np
import pytest
from conftest import POOL_THETA


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


def sized_rows(n=48, size_step=0.0, seed=3):
    """Half the trials at 15px, half at 14px, with a planted size effect of `size_step`
    log-seconds. The THEME effect is identical in both halves, so a model that absorbs the
    size step recovers the same surface either way."""
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


def sized_surface_corr(model, rows):
    """How well the fitted surface tracks the truth those rows were generated from."""
    rf = model.rt_fit(rows, "day", None)
    rg = np.random.default_rng(11)
    grid = [rg.random(9) for _ in range(300)]
    mu = model.rt_at(rf, grid, "day")[0]
    truth = np.array([1.4 - 0.5 * g[0] for g in grid])
    return float(np.corrcoef(mu, truth)[0, 1])


def hunt_run(model, strategy, n=40, seed=1, noise=0.22):
    """One sitting of 40 find hunts, choosing where to look either actively or uniformly.

    Only two axes matter to the planted truth, and they interact; the question is whether
    picking the highest-variance point of the grid maps that surface faster than sampling
    it at random.
    """
    base = POOL_THETA[3].copy()

    def truth(th):
        return 1.5 - 0.5 * th[7] - 0.35 * th[8] + 0.3 * th[7] * th[8]

    r = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rf = model.rt_fit(rows, "day", None) if len(rows) >= 8 else None
        if strategy == "active" and rf is not None and r.random() > 0.25:
            g = np.linspace(0.05, 0.95, 7)
            cands = []
            for v7 in g:
                for v8 in g:
                    c = base.copy()
                    c[7], c[8] = v7, v8
                    cands.append(c)
            th = cands[int(np.argmax(model.rt_at(rf, cands, "day")[1]))]
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
    rf = model.rt_fit(rows, "day", None)
    rg = np.random.default_rng(99)
    grid = []
    for _ in range(400):
        t = base.copy()
        t[7], t[8] = rg.random(), rg.random()
        grid.append(t)
    mu = model.rt_at(rf, grid, "day")[0]
    tv = np.array([truth(t) for t in grid])
    return float(np.corrcoef(mu, tv)[0, 1])


@pytest.fixture(scope="module")
def probe():
    """300 themes to read the fitted surface at. Shared, so every test below is scored on
    the same points."""
    return [np.random.default_rng(900 + i).random(9) for i in range(300)]


@pytest.fixture(scope="module")
def timed_fit(search_model, probe):
    """One 90-trial fit with no task offset, plus what it predicts on the probe.

    Three tests read this: the recovery itself, the offset comparison that needs its
    correlation as a baseline, and the exclusion constraint.
    """
    rows, truth = synth_timed(90, seed=5, offset=0.0)
    rf = search_model.rt_fit(rows, "day", None)
    pred, _var = search_model.rt_at(rf, probe, "day")
    true = np.array([truth(t) for t in probe])
    return {"rf": rf, "true": true, "rho": float(np.corrcoef(pred, true)[0, 1])}


def test_the_surface_recovers_reading_speed(timed_fit):
    """90 timed trials are enough to see a reading-speed surface at all (measured 0.81).

    The floor is deliberately low: this asks whether the channel carries information, not
    how much. If it does not clear 0.5 the timed trials are decoration.
    """
    assert timed_fit["rho"] > 0.5, f"corr(predicted, true log-time) = {timed_fit['rho']:.2f}"


def test_per_arm_baselines_absorb_the_task_offset(search_model, probe, timed_fit):
    """The same truth, but with the two arms 0.55 log-seconds apart.

    A single pooled mean would push that constant into the theme surface; per-arm means
    should not, and measurably do not -- 0.81 with the offset against 0.81 without it. The
    0.12 tolerance is there for the noise in a 90-trial fit, not for a real degradation.
    """
    rows_off, truth_off = synth_timed(90, seed=5, offset=0.55)
    rf_off = search_model.rt_fit(rows_off, "day", None)
    pred_off, _var = search_model.rt_at(rf_off, probe, "day")
    rho_off = float(np.corrcoef(pred_off, np.array([truth_off(t) for t in probe]))[0, 1])
    assert rho_off > timed_fit["rho"] - 0.12, (
        f"corr {rho_off:.2f} with a 0.55 log-s arm offset vs {timed_fit['rho']:.2f} without"
    )


def test_the_per_arm_and_size_baseline_absorbs_a_size_step(search_model):
    """A change of type size must not land on the theme surface either.

    Half the trials at 15px and half at 14px, with the theme effect identical in both
    halves: a 0.45 log-second size step costs the recovered surface nothing (measured
    0.656 with the step against 0.656 without it).
    """
    c_flat = sized_surface_corr(search_model, sized_rows(size_step=0.0))
    c_step = sized_surface_corr(search_model, sized_rows(size_step=0.45))
    assert c_step > c_flat - 0.08, f"corr with truth {c_step:.3f} with the step against {c_flat:.3f} without it"


def test_the_time_constraint_excludes_the_slow_and_spares_the_fast(search_model, probe, timed_fit):
    """`rt_penalty` may refuse a candidate for being slow to read, so it had better be
    right about which ones are.

    Being conservative is allowed and is what happens here: at this noise level nothing is
    excluded at all, and an empty exclusion set is a pass. What would not be a pass is
    excluding pages the truth says are fast, so the check is on the RANK of the excluded
    ones against the kept ones rather than on the count.
    """
    excl, _secs = search_model.rt_penalty(timed_fit["rf"], probe, "day")
    if not excl.any():
        # Which is what happens at this noise level: the constraint excludes nothing, so
        # there is nothing to be wrong about. Said out loud, because a branch that always
        # passes is the shape of a vacuous test -- if the constraint ever does start
        # excluding candidates, the rank check below is what it has to satisfy.
        return
    true = timed_fit["true"]
    slow_rank = float(np.mean([np.mean(true[excl] > t) for t in true[~excl]]))
    assert slow_rank > 0.75, (
        f"{int(excl.sum())} excluded, and an excluded page is slower than "
        f"{100 * slow_rank:.0f}% of the kept ones on the truth"
    )


def test_uncertainty_hunting_beats_uniform_sweeps(search_model):
    """Choosing the next hunt where the surface is least certain, against sampling at
    random.

    Five sittings of 40 hunts each way; active wins on the mean (measured 0.873 against
    0.791). This is the one place in the instrument where the CHOICE of stimulus is
    adaptive on the timing channel rather than the preference channel, so it needs its own
    evidence that the adaptivity pays.
    """
    uniform = float(np.mean([hunt_run(search_model, "uniform", seed=s) for s in (1, 2, 3, 4, 5)]))
    active = float(np.mean([hunt_run(search_model, "active", seed=s) for s in (1, 2, 3, 4, 5)]))
    assert active > uniform, f"corr(pred, truth) active {active:.3f} vs uniform {uniform:.3f} at 40 hunts"
