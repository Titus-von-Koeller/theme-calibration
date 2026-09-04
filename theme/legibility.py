"""The legibility surface: how fast a theme can actually be read.

The comprehension probes and find hunts were once only described in the analysis -- a
median, a slope -- and never touched the verdict, so a third of every sitting's clicks
bought nothing. They measure a different quantity from preference: not which page one
would rather live in, but how fast a name can actually be found in it. So they get their
own function over the same theme space.

A Gaussian process on log time-to-click, which is closed-form (no Laplace, no Newton)
because the observation is a number rather than a comparison: log-RT is roughly normal,
its noise is multiplicative, and the same ARD length-scales carry over since the axes that
move preference are the ones likely to move legibility. Correct and never-paused trials
only -- a paused clock measures the break and a wrong click measures something else
entirely.

Preference chooses; legibility constrains. That is the program's constitution applied one
level deeper: the contrast floors keep a page readable in principle, and this keeps it
readable in fact.
"""

import math

import numpy as np

from .kernel import SIGNAL_VARIANCE, coords, kmat, quadratic_form

# A trial faster than this is a slip and one slower is a distraction, neither of them a
# reading time.
MIN_RT_MS = 250.0
MAX_RT_MS = 30000.0

# Below this many trials a per-(arm, size) cell's own mean is a worse estimate than the
# arm's, and the cell keeps the arm mean.
MIN_CELL_TRIALS = 6

# Fewer timed trials than this and there is no surface to fit.
MIN_TIMED_TRIALS = 8

# The type size a trial is assumed to have run at when the log does not say.
DEFAULT_CODE_PX = 15.0

TIMED_MODES = ("comprehension", "search")


def _timed_trials(responses, polarity):
    """(GP inputs, log times, is-a-hunt flags, type sizes) from the usable timed trials."""
    gp_inputs, log_times, is_hunt, pixel_sizes = [], [], [], []
    for row in responses:
        if row.get("mode") not in TIMED_MODES:
            continue
        if row.get("polarity") != polarity or not row.get("correct") or row.get("paused"):
            continue
        rt = float(row.get("rt_ms") or 0.0)
        if rt < MIN_RT_MS or rt > MAX_RT_MS:
            continue
        gp_inputs.append(coords(row["theta_a"], polarity))
        log_times.append(np.log(rt))
        is_hunt.append(1.0 if row["mode"] == "search" else 0.0)
        pixel_sizes.append(float(row.get("code_px") or DEFAULT_CODE_PX))
    return np.array(gp_inputs), np.array(log_times), np.array(is_hunt), np.asarray(pixel_sizes)


def _nuisance_baselines(log_times, is_hunt, pixel_sizes):
    """Per-arm and per-size means to remove, plus what a prediction is relative to.

    A PER-ARM baseline, not one global mean. A find hunt highlights every match and asks
    which is current; a comprehension probe gives a bare page and a name. The second is
    systematically slower, and folding both into one mean would push that constant
    difference into the theme surface as if some regions of theme space were slow -- when
    what was slow was the task. Each arm's own mean is removed, and the surface then models
    only what the THEME does to the clock. Needs both arms present to be worth doing; with
    one arm this collapses to the global mean.

    And a per-SIZE offset, for exactly the reason there is a per-arm one. Glyph scale moves
    reading time on its own, and the timed arms have not always run at one size: they were
    15 or 16 before the stimulus was pinned to the sizes actually read at (14 in editors,
    16 in notebook cells). Without this the step from one size regime to the next lands on
    the theme surface as if some region of theme space had got slower on the day the size
    changed. Only fitted where a size has enough trials to mean anything; the rest fall
    back to the arm's own mean.

    Returns (per-trial baseline, prediction baseline, probe mean, hunt mean).
    """
    has_both_arms = 0 < float(is_hunt.mean()) < 1
    probe_mean = float(log_times[is_hunt == 0].mean()) if (is_hunt == 0).any() else float(log_times.mean())
    hunt_mean = float(log_times[is_hunt == 1].mean()) if (is_hunt == 1).any() else float(log_times.mean())
    baseline = (
        np.where(is_hunt > 0.5, hunt_mean, probe_mean) if has_both_arms else np.full(len(log_times), log_times.mean())
    ).astype(float)
    for arm in (0.0, 1.0):
        for size in np.unique(pixel_sizes):
            cell = (pixel_sizes == size) & (is_hunt == arm)
            if cell.sum() >= MIN_CELL_TRIALS:
                baseline[cell] = float(log_times[cell].mean())
    prediction_baseline = probe_mean if has_both_arms else float(log_times.mean())
    return baseline, prediction_baseline, probe_mean, hunt_mean


def rt_fit(responses, polarity, length_scales=None, noise_share=0.45):
    """The reading-time surface over theme space, or None if the log is too thin."""
    gp_inputs, log_times, is_hunt, pixel_sizes = _timed_trials(responses, polarity)
    if len(gp_inputs) < MIN_TIMED_TRIALS:
        return None
    baseline, prediction_baseline, probe_mean, hunt_mean = _nuisance_baselines(log_times, is_hunt, pixel_sizes)
    # Signal and noise variance estimated from the data rather than borrowed from the
    # preference kernel: the preference GP's prior sd of 2 means a factor of seven in
    # log time, which produced a predicted span of 1.4 to 14 seconds -- nonsense on a
    # task completed in two to four. Total variance is what log-RT actually shows, and
    # reaction time is famously noisy, so a large share of it is called noise (0.45):
    # the surface then claims a real difference only where the data insists.
    residual = log_times - baseline
    total_variance = max(float(residual.var()), 1e-4)
    signal_variance = max((1.0 - noise_share) * total_variance, 1e-4)
    noise_variance = max(noise_share * total_variance, 1e-4)
    cov = (signal_variance / SIGNAL_VARIANCE) * kmat(gp_inputs, gp_inputs, length_scales) + noise_variance * np.eye(
        len(gp_inputs)
    )
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Unreachable in principle: noise_variance is at least 1e-4, so the diagonal is
        # strictly positive and the matrix is positive definite. Kept because the cost of
        # being wrong about that is a raised exception in the middle of generating a trial,
        # and the cost of being right is that the legibility constraint sits out one trial.
        return None
    return {
        "X": gp_inputs,
        "y": log_times,
        "resid": residual,
        "base": baseline,
        "mu0": prediction_baseline,
        "m_probe": probe_mean,
        "m_hunt": hunt_mean,
        "Ki": precision,
        "ls": length_scales,
        "n": len(gp_inputs),
        "sf2": signal_variance,
        "noise": noise_variance,
    }


def rt_at(surface, thetas, polarity):
    """Posterior mean and variance of log time-to-click at arbitrary themes."""
    query_inputs = np.array([coords(theta, polarity) for theta in thetas])
    scale = surface["sf2"] / SIGNAL_VARIANCE
    cross_cov = scale * kmat(query_inputs, surface["X"], surface.get("ls"))
    mean = surface["mu0"] + cross_cov @ (surface["Ki"] @ surface["resid"])
    variance = np.maximum(surface["sf2"] - quadratic_form(cross_cov, surface["Ki"]), 1e-9)
    return mean, variance


def rt_penalty(surface, thetas, polarity, tolerance=0.10, confidence=0.9):
    """Which candidates are CREDIBLY slower to read than the fastest, and by how much.

    Returns (excluded mask, predicted time in milliseconds). A candidate is excluded only
    when the posterior says it is worse than the best by more than `tolerance` in log time
    with at least `confidence` probability -- so a thin or noisy RT log excludes nothing,
    which is the correct behaviour rather than a convenient one. The floor is relative: the
    question is never "is this page fast enough" in the abstract but "is it needlessly
    slower than a page liked just as much".
    """
    mean, variance = rt_at(surface, thetas, polarity)
    best = float(np.min(mean))
    # P(mu_i - best > tolerance) under a normal, without the covariance between i and the
    # argmin, and with the argmin's variance approximated by the smallest in the set:
    # conservative, which is the right direction for a constraint.
    sd = np.sqrt(variance + float(np.min(variance)))
    z = (mean - best - tolerance) / np.maximum(sd, 1e-9)
    p_worse = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    return p_worse > confidence, np.exp(mean)
