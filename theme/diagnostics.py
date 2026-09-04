"""The readouts that say whether the answer is settled -- and whether to believe them.

P(best) as a distribution over argmaxes and its credible set; which axes the clicks have
settled; whether another sitting is worth clicking; and a permutation test for whether the
preferred theme depends on some logged property of how it was shown.

Every one of these prints a number that looks like a measurement, so each carries the
calibration that says when to believe it.
"""

import math

import numpy as np

from .kernel import scale_thetas, spread_positions
from .preference import duel_rows, fitted, posterior_joint

BEST_MEMO = {}

# The standard deviation of a uniform axis, which every axis spread is reported relative
# to, so a settled axis and an untouched one are comparable numbers. Spelled as the
# expression rather than as 0.2887, which was the rounded figure the readout divided by.
UNIFORM_AXIS_SD = 1.0 / math.sqrt(12.0)

# A leader holding more than this much of the argmax mass is a winner; more than the
# smaller figure is a real plateau; below it the log cannot yet tell.
SINGLE_WINNER_MASS = 0.5
PLATEAU_MASS = 0.12


def _argmax_probabilities(mean, cov, samples, seed):
    """P(each candidate is the argmax), from samples of the JOINT posterior.

    Sample the joint, because candidates near each other share almost all their
    uncertainty and marginals would scatter the probability of being best across a cluster
    of effectively identical pages.
    """
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # posterior_joint symmetrizes and adds 1e-8 to the diagonal, so this is rare -- but
        # a candidate set holding exact duplicates is singular by construction, and the
        # breeder can propose one. An eigendecomposition with the negative eigenvalues
        # clipped gives a usable factor where a Cholesky cannot, and P(best) over duplicates
        # is meaningful either way because grouping merges them straight afterwards.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 1e-12))
    normals = np.random.default_rng(seed).standard_normal((len(mean), samples))
    draws = mean[:, None] + factor @ normals
    return np.bincount(np.argmax(draws, axis=0), minlength=len(mean)) / float(samples)


def _group_by_resolution(scaled_thetas, p_best, radius):
    """Group candidates into perceptually distinct themes: greedy, best-first.

    A candidate set of eight hundred contains many pages that differ by less than anyone
    could see, and each sibling steals argmax mass from the others: measured on the real
    log, the leader held 1.6% while the report claimed a plateau -- a number that says
    nothing about whether one theme leads. Mass belongs to a perceptually distinct group,
    not to a coordinate.

    Returns (representative indices, group index per candidate, order of candidates).
    """
    order = np.argsort(-p_best)
    representatives, group_of = [], np.full(len(scaled_thetas), -1)
    for i in order:
        if representatives:
            distances = np.linalg.norm(scaled_thetas[representatives] - scaled_thetas[i], axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= radius:
                group_of[i] = nearest
                continue
        group_of[i] = len(representatives)
        representatives.append(int(i))
    return representatives, group_of, order


def _fit_identity(fit):
    """What names a fit for caching.

    `preference.fitted` stamps a content fingerprint on every fit it builds. A fit
    assembled by hand has none, and falls back to its object identity -- which is only
    sound while the object is alive, because an address gets reused.
    """
    return fit["fingerprint"] if "fingerprint" in fit else ("id", id(fit))


def _candidate_fingerprint(thetas):
    """A hash of every coordinate of every candidate.

    The key used to name only `len(thetas)` and `sum(thetas[0])`, so two candidate sets of
    equal length sharing a first entry collided and the second caller read the first one's
    P(best) as its own. 57 kB of bytes for eight hundred candidates, against a Cholesky
    over the same eight hundred.
    """
    return hash(np.ascontiguousarray(np.asarray(thetas, dtype=float)).tobytes())


def _group_masses(group_of, p_best, n_groups):
    """Each group's share of the argmax probability."""
    return np.bincount(group_of, weights=p_best, minlength=n_groups)


def _credible_groups(group_p, group_order, mass):
    """The smallest set of leading groups holding `mass` of the argmax probability."""
    keep, accumulated = [], 0.0
    for group in group_order:
        keep.append(int(group))
        accumulated += group_p[group]
        if accumulated >= mass:
            break
    return keep


def best_set(fit, polarity, thetas, samples=2048, mass=0.5, seed=0, radius=0.9):
    """Which theme is best, or which SET is -- as a distribution over argmaxes.

    Three things have to be right for this to answer the question honestly: sample the
    JOINT posterior, GROUP before counting, and read the verdict off CUMULATIVE mass rather
    than an absolute cutoff. The credible set is the smallest group of groups holding
    `mass` of the argmax probability: one group over half of it is a winner; a handful
    sharing it is a real plateau; and when even the top group is thin, the honest answer is
    that the log cannot yet tell -- which is a state this reports rather than dressing up
    as a plateau.
    """
    # Memoized on the fit, the polarity and the candidate set: the analysis asks for the
    # same verdict three times per polarity (the shelf, and the two historical fits behind
    # the progress readout), and each call is a Cholesky over eight hundred candidates.
    cache_key = (_fit_identity(fit), polarity, _candidate_fingerprint(thetas), samples, mass, seed, radius)
    if cache_key in BEST_MEMO:
        return BEST_MEMO[cache_key]
    mean, cov = posterior_joint(fit, thetas, polarity)
    p_best = _argmax_probabilities(mean, cov, samples, seed)

    scaled = scale_thetas(thetas, fit.get("ls"))
    representatives, group_of, order = _group_by_resolution(scaled, p_best, radius)
    group_p = _group_masses(group_of, p_best, len(representatives))
    group_order = np.argsort(-group_p)
    keep = _credible_groups(group_p, group_order, mass)
    lead = float(group_p[group_order[0]])
    verdict = "single" if lead > SINGLE_WINNER_MASS else ("plateau" if lead > PLATEAU_MASS else "undecided")
    result = {
        "p_best": p_best,
        "order": order,
        "groups": representatives,
        "group_p": group_p,
        "group_order": group_order,
        "group_of": group_of,
        "credible": [representatives[group] for group in keep],
        "credible_p": [float(group_p[group]) for group in keep],
        "lead": lead,
        "mu": mean,
        "verdict": verdict,
    }
    if len(BEST_MEMO) > 8:
        BEST_MEMO.pop(next(iter(BEST_MEMO)))
    BEST_MEMO[cache_key] = result
    return result


def axis_consensus(best, thetas):
    """Which axes the clicks have SETTLED, and which are still open.

    The plateau readout says how many themes are still in contention; it does not say
    what they disagree about. Measured on the four leading day themes: their grounds sit
    within 4 units of one cream, while their keyword hues run violet, dark green, dark
    red and blue. Reading "four distinct themes" against four pages that look alike at a
    glance is confusing; reading "the ground is decided, the accent hue is not" says
    what the remaining duels are for.

    Per axis, the posterior-weighted spread of theta under P(best), against the 0.289 of a
    uniform axis. Small means the mass has collected on one value; near 1 means the clicks
    have not distinguished anything along it yet.
    """
    p_best = np.asarray(best["p_best"], dtype=float)
    points = np.asarray(thetas, dtype=float)
    if p_best.sum() <= 0 or len(points) == 0:
        return []
    p_best = p_best / p_best.sum()
    mean = p_best @ points
    sd = np.sqrt(np.maximum(p_best @ (points - mean) ** 2, 0.0))
    return [(axis, float(sd[axis] / UNIFORM_AXIS_SD), float(mean[axis])) for axis in range(points.shape[1])]


def _log_as_of(responses, n_duels_kept):
    """The log with all but the first `n_duels_kept` answered duels dropped.

    Non-duel rows are kept wholesale: nothing downstream of here reads them, and dropping
    them by position would need a second definition of "as of when".
    """
    kept, history = 0, []
    for row in responses:
        if row.get("mode") == "duel" and row.get("choice") in (0, 1):
            if kept >= n_duels_kept:
                continue
            kept += 1
        history.append(row)
    return history


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
    duels = duel_rows(responses)
    if len(duels) < back + 12:
        return None
    now = fitted(responses)
    then = fitted(_log_as_of(responses, len(duels) - back))
    if then is None:
        return None
    best_now = best_set(now, polarity, thetas, seed=17)
    best_then = best_set(then, polarity, thetas, seed=17)
    lead_gain = best_now["lead"] - best_then["lead"]
    duels_to_decide = None
    if lead_gain > 1e-3 and best_now["lead"] < SINGLE_WINNER_MASS:
        duels_to_decide = int(np.ceil((SINGLE_WINNER_MASS - best_now["lead"]) / (lead_gain / back)))
    return {
        "duels": len(duels),
        "lead_now": best_now["lead"],
        "lead_then": best_then["lead"],
        "set_now": len(best_now["credible"]),
        "set_then": len(best_then["credible"]),
        "back": back,
        "duels_to_decide": duels_to_decide,
    }


def spread_out(thetas, indices, n_wanted, length_scales=None):
    """`n_wanted` maximally different members of a set -- greedy max-min in scaled theta space.

    A plateau is only useful if its members actually look different; picking the top-k
    by probability would return k variations of one page.
    """
    if not indices:
        return []
    scaled = scale_thetas([thetas[i] for i in indices], length_scales)
    return [indices[position] for position in spread_positions(scaled, [0], n_wanted)]


# LOAD-BEARING placement in the notebook this came from: above the function that closes
# over it. marimo mangles a cell-local underscore name only where it has already seen the
# assignment, so a memo declared BELOW its user resolves fine under `marimo edit` and
# raises NameError under `marimo run` the moment another cell calls in. Same trap as
# _CONTROL in the stimulus cell. Harmless here, where module scope is not cell scope, but
# the notebook copy still depends on it.
SURF_MEMO = {}


def _factor_duels(responses, polarity, key):
    """The duels this factor test can use, truncated to a whole bucket of eight.

    Recomputed every EIGHTH duel, not every click. 200 permutations x 5-fold Newton fits
    costs about 2.8 s per factor, and with two factors over two polarities that was 8.3 s
    of the analysis on every single answer -- which is not just slow, it is long enough for
    two widget re-renders to overlap and leave a full-screen orphan stage over the page
    (measured 2026-09-04, and the reason a trial appeared to blank mid-sitting). Truncating
    to a whole bucket keeps the memo key honest: within a bucket the INPUT is identical, so
    the cached answer is the exact answer for the data named.
    """
    rows = [
        row
        for row in responses
        if row.get("mode") == "duel"
        and row.get(key) is not None
        and row.get("polarity") == polarity
        and not row.get("paused")
        and row.get("choice") in (0, 1)
    ]
    return rows[: (len(rows) // 8) * 8]


def _winner_minus_loser(rows):
    """Per duel, the winning theme's axes minus the losing theme's.

    choice 0 = theme_a won (duels_from's convention; `swap` governs only which SIDE a card
    appeared on, not which theme it was).
    """
    return np.array(
        [(np.array(row["theta_a"]) - np.array(row["theta_b"])) * (1.0 if row["choice"] == 0 else -1.0) for row in rows]
    )


def _tilt_features(axis_differences, levels_of_row, n_levels, n_tilt_axes):
    """The design matrix: one shared utility, plus a sum-to-zero per-level tilt.

    `n_tilt_axes` of 0 is one shared utility; above 0 it adds the tilt on that many leading
    axes -- the cheapest form the interaction can take, and so the one with the best chance
    of showing in the data there is.
    """
    if not n_tilt_axes:
        return axis_differences
    columns = [axis_differences]
    for axis in range(n_tilt_axes):
        for level in range(n_levels - 1):
            column = np.where(
                levels_of_row == level,
                axis_differences[:, axis],
                np.where(levels_of_row == n_levels - 1, -axis_differences[:, axis], 0.0),
            )
            columns.append(column[:, None])
    return np.hstack(columns)


def _held_out_loglik(axis_differences, levels_of_row, n_levels, n_tilt_axes, seed):
    """Mean held-out Bradley-Terry log-likelihood under five-fold cross-validation.

    Higher is better: a per-level tilt has to EARN its extra parameters on held-out
    choices, because fit alone always improves.
    """
    order = np.random.default_rng(seed).permutation(len(axis_differences))
    total, n_scored = 0.0, 0
    for fold in range(5):
        held_out = order[fold::5]
        train = np.setdiff1d(order, held_out)
        if len(train) < 10:
            continue
        train_levels = levels_of_row[train] if n_tilt_axes else None
        features = _tilt_features(axis_differences[train], train_levels, n_levels, n_tilt_axes)
        coefficients = np.zeros(features.shape[1])
        for _ in range(60):
            p_first_wins = 1.0 / (1.0 + np.exp(-(features @ coefficients)))
            gradient = features.T @ (1.0 - p_first_wins) - coefficients
            hessian = (features * (p_first_wins * (1 - p_first_wins))[:, None]).T @ features + np.eye(
                len(coefficients)
            )
            coefficients = coefficients + np.linalg.solve(hessian, gradient)
        held_out_levels = levels_of_row[held_out] if n_tilt_axes else None
        features = _tilt_features(axis_differences[held_out], held_out_levels, n_levels, n_tilt_axes)
        log_odds = features @ coefficients
        # log(sigmoid(z)), by the stable identity. Written as log1p(exp(-z)) it
        # overflows to inf for z below about -709 and takes the whole permutation null
        # to -inf with it; logaddexp is exact over that range and identical where the
        # naive form was already accurate.
        total += float(np.sum(-np.logaddexp(0.0, -log_odds)))
        n_scored += len(held_out)
    return total / max(n_scored, 1)


def _tilt_free_loglik(axis_differences, n_seeds):
    """The held-out log-likelihood of one shared utility, averaged over fold splits.

    It takes no levels argument because it cannot use one: with no tilt the design matrix
    is the axis differences alone. That is why the permutation null does not need to
    recompute it -- shuffling the labels cannot move a number that never saw them.
    """
    return np.mean([_held_out_loglik(axis_differences, None, 0, 0, s) for s in range(n_seeds)])


def _tilt_gain(axis_differences, levels_of_row, n_levels, n_seeds, tilt_free=None):
    """How much held-out log-likelihood the per-level tilt buys, averaged over fold splits."""
    with_tilt = np.mean([_held_out_loglik(axis_differences, levels_of_row, n_levels, 1, s) for s in range(n_seeds)])
    if tilt_free is None:
        tilt_free = _tilt_free_loglik(axis_differences, n_seeds)
    return float(with_tilt - tilt_free)


def _factor_cache_key(rows, key, polarity, nperm, seed):
    """Everything about this call that changes its answer.

    nperm and seed belong in it. Without them a coarse call -- a quick 20-permutation
    sanity check, say -- poisons the cache for the careful 200-permutation reading that
    follows, and the caller gets a p-value computed against a null it never asked for.
    So do both thetas in full: the key named `theta_a[0]` and nothing of `theta_b`, while
    the statistic is built from theta_a - theta_b on all nine axes, so a hit could return
    a permutation test run against data the caller never supplied.
    """
    return (
        "f",
        key,
        polarity,
        nperm,
        seed,
        hash(tuple((row["choice"], str(row[key])) for row in rows)),
        _candidate_fingerprint([row["theta_a"] for row in rows] + [row["theta_b"] for row in rows]),
    )


def _factor_verdict(key, p_value):
    """Three-state, because "quiet" is not "absent" when the test has little power."""
    if p_value < 0.02:
        return f"{key} changes the optimum -- one theme is the wrong answer shape"
    if p_value < 0.10:
        return f"suggestive; keep {key} balanced and re-read"
    return f"no {key} effect this data can see"


def _permutation_null(axis_differences, levels_of_row, n_levels, nperm, seed):
    """The gain statistic under `nperm` shuffles of the level labels.

    The tilt-free half of the gain is the same number for every permutation -- with no
    tilt the design matrix never sees the labels -- so it is computed once. Half the
    permutation test was recomputing it.
    """
    tilt_free = _tilt_free_loglik(axis_differences, 2)
    rng = np.random.default_rng(seed)
    return np.array(
        [_tilt_gain(axis_differences, rng.permutation(levels_of_row), n_levels, 2, tilt_free) for _ in range(nperm)]
    )


def factor_effect(responses, polarity, key, nperm=200, seed=7, min_n=24):
    """Does the preferred theme depend on some logged property of how it was SHOWN?

    The same question for any stimulus factor -- which surface, what pixel size, which
    kind of code -- because the machinery is identical and a second copy of a
    permutation test is a second place for it to be subtly wrong. `key` names the field
    in the log; its distinct values become the levels.

    See surface_effect below for what the test does and why the null is permutation.
    """
    rows = _factor_duels(responses, polarity, key)
    levels = sorted({row[key] for row in rows}, key=str)
    if len(rows) < min_n or len(levels) < 2:
        return len(rows), 0.0, 1.0, f"not enough {polarity} duels with a {key} to compare"
    cache_key = _factor_cache_key(rows, key, polarity, nperm, seed)
    if cache_key in SURF_MEMO:
        return SURF_MEMO[cache_key]
    levels_of_row = np.array([levels.index(row[key]) for row in rows])
    axis_differences = _winner_minus_loser(rows)
    observed = _tilt_gain(axis_differences, levels_of_row, len(levels), 6)
    null = _permutation_null(axis_differences, levels_of_row, len(levels), nperm, seed)
    p_value = float((null >= observed).mean())
    result = (len(rows), observed, p_value, _factor_verdict(key, p_value))
    if len(SURF_MEMO) > 8:
        SURF_MEMO.pop(next(iter(SURF_MEMO)))
    SURF_MEMO[cache_key] = result
    return result


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
