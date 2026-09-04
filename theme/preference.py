"""The preference model: which theme is liked better, from duels.

A Gaussian process over theme space with a Bradley-Terry likelihood on duels, fit by
Laplace approximation -- Chu & Ghahramani's preferential GP, with QUEST+'s
generate-the-most-informative-trial loop on top. Reaction time enters the likelihood
drift-diffusion-style: decision time falls as the utility gap grows, so a fast click
steepens that duel's slope and a slow one flattens it toward a tie.

Also here: the held-out cross-validation that decides how much the clock is allowed to
weight a duel, and the two information-theoretic helpers the trial chooser acquires
against.
"""

import numpy as np

from .kernel import N_AXES, POLARITY_AXIS, SIGNAL_VARIANCE, ard_scales, coords, kmat


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


def duel_rows(responses):
    """The answered duels in a response log, in order.

    One predicate in one place: an earlier version of the diagnostics encoded the winner
    as "a"/"b", every row failed the `choice in (0, 1)` test, and the readout reported its
    not-enough-data answer as if it had measured something.
    """
    return [row for row in responses if row.get("mode") == "duel" and row.get("choice") in (0, 1)]


def log_fingerprint(rows):
    """A hash of everything in these duel rows that changes a fit.

    A cache key must name every input that changes the answer. Keying a fit on how MANY
    duels were answered names how much data there was and never which, so two logs of
    equal length were served one fit -- and a fit looks exactly like a measurement whether
    or not it belongs to the log that asked for it. In production the log only ever grows,
    so the count happened to identify it; the progress readout, which fits a truncated
    history beside the full one, is one step from the collision, and any analysis
    comparing two logs is already in it.

    These are exactly the fields `duels_from` reads, so the fingerprint is complete for
    everything downstream of it.
    """
    return hash(
        tuple(
            (
                row["choice"],
                row["polarity"],
                tuple(row["theta_a"]),
                tuple(row["theta_b"]),
                row.get("rt_ms"),
                bool(row.get("paused")),
                bool(row.get("swap")),
            )
            for row in rows
        )
    )


def duels_from(responses, rt_p=0.5):
    """(GP inputs, duel index pairs, per-duel slopes, prior means, winner sides) from a log.

    One GP input per distinct (theme, polarity) shown, so a theme duelled twice is one
    point carrying two comparisons.
    """
    points, index_of = [], {}
    duels, reaction_times, was_paused, winner_sides = [], [], [], []
    for row in duel_rows(responses):
        ids = []
        for theta in (row["theta_a"], row["theta_b"]):
            key = (tuple(round(float(v), 6) for v in theta), row["polarity"])
            if key not in index_of:
                index_of[key] = len(points)
                points.append(coords(theta, row["polarity"]))
            ids.append(index_of[key])
        duels.append((ids[row["choice"]], ids[1 - row["choice"]]))
        reaction_times.append(float(row.get("rt_ms", 2500.0)))
        was_paused.append(bool(row.get("paused")))
        # Which SIDE the winner was displayed on. Measured 2026-09-03 over 79 duels: the
        # right-hand card is picked 61% of the time (z = -1.91 against no bias).
        # Unmodelled, that lands on the utility as noise; as a fitted term it is
        # subtracted out. Reconstructible from the log, so no past duel is wasted.
        winner_slot = (1 - row["choice"]) if row.get("swap") else row["choice"]
        winner_sides.append(1.0 if winner_slot == 0 else -1.0)
    if not points:
        return None
    gp_inputs = np.array(points)
    was_paused = np.array(was_paused)
    unpaused_times = np.array(reaction_times)[~was_paused]
    median_time = float(np.median(unpaused_times)) if len(unpaused_times) >= 8 else 2500.0
    # The exponent is FITTED, not assumed (see rt_exponent below). p = 0.5 was a
    # hand-rolled square root; p = 0 means the clock is ignored entirely, so the same
    # search that calibrates this channel also tests whether it earns its keep.
    duel_slopes = np.clip((median_time / np.maximum(np.array(reaction_times), 200.0)) ** rt_p, 0.6, 1.8)
    # A paused trial's time says nothing about the utility gap: its choice still counts,
    # at the neutral slope, neither sharpened nor flattened by the clock.
    duel_slopes[was_paused] = 1.0
    prior_mean = realized_space().prior_mean
    prior_means = np.array([prior_mean(x[:N_AXES], "night" if x[POLARITY_AXIS] > 0.5 else "day") for x in gp_inputs])
    return gp_inputs, duels, duel_slopes, prior_means, np.array(winner_sides)


def duel_difference_matrix(duels, n_points):
    """Rows of +1 at the winner and -1 at the loser, so `D @ utilities` is the utility gaps.

    One BLAS product per Newton step instead of a Python loop over duels. Each duel
    contributes q_k (e_win - e_lose)(e_win - e_lose)^T to the Hessian, which is exactly
    D^T diag(q) D for this matrix -- so the whole update is two matrix products. Measured
    on the live log: the loop cost 108 ms per fit, an np.add.at scatter cost 166 ms
    (add.at is unbuffered and slow), and this costs 128 ms -- SLOWER than the loop at 121
    duels, because building D dominates at that size. Kept anyway: the loop pays one
    interpreter trip per duel per Newton step, so it degrades linearly in log length where
    this is one BLAS call, and 20 ms is noise against a 350 ms trial. Revisit only if a
    fit ever dominates again. Identical arithmetic either way -- the recovery tests
    reproduce every number.
    """
    differences = np.zeros((len(duels), n_points))
    for k, (winner, loser) in enumerate(duels):
        differences[k, winner] += 1.0
        differences[k, loser] -= 1.0
    return differences


def _bt_derivatives(differences, scaled_differences, duel_slopes, utilities, sides, side_bias):
    """Bradley-Terry gradient in the utilities, and D^T diag(q) D at the current point."""
    log_odds = scaled_differences @ utilities + side_bias * sides
    p_winner_wins = 1.0 / (1.0 + np.exp(-log_odds))
    gradient = scaled_differences.T @ (1.0 - p_winner_wins)
    duel_curvature = duel_slopes * duel_slopes * p_winner_wins * (1.0 - p_winner_wins)
    return gradient, (differences * duel_curvature[:, None]).T @ differences


def _newton_side_bias(utility_gaps, sides, side_bias):
    """One-dimensional Newton for the position-bias term, at fixed utilities.

    Shrunk toward zero by an L2 prior (the 4.0 terms): under-correcting a real bias is
    safer than inventing one.
    """
    for _ in range(40):
        p_winner_wins = 1.0 / (1.0 + np.exp(-(utility_gaps + side_bias * sides)))
        gradient = float(sides @ (1.0 - p_winner_wins)) - 4.0 * side_bias
        curvature = -float((sides * sides) @ (p_winner_wins * (1 - p_winner_wins))) - 4.0
        step = -gradient / curvature
        side_bias = float(np.clip(side_bias + step, -2.0, 2.0))
        if abs(step) < 1e-10:
            break
    return side_bias


def fit_laplace(gp_inputs, duels, duel_slopes, prior_means, winner_sides=None, length_scales=None):
    """Laplace posterior over utilities, alternating with the position-bias term.

    The bias is one number shared by every duel: the log-odds advantage of the card on
    the left. It and the utilities are identifiable because side is randomized
    independently of theme, and they are fitted by alternation -- utilities given the bias
    by Newton, then the bias given the utilities by its own one-dimensional Newton --
    which converges in two or three rounds at this scale.

    Returns (utilities, utility covariance, prior precision, side bias).
    """
    n_points = len(gp_inputs)
    prior_cov = kmat(gp_inputs, gp_inputs, length_scales) + 1e-6 * np.eye(n_points)
    prior_precision = np.linalg.inv(prior_cov)
    utilities = prior_means.copy()
    sides = np.zeros(len(duels)) if winner_sides is None else np.asarray(winner_sides, dtype=float)
    duel_slopes = np.asarray(duel_slopes, dtype=float)
    side_bias = 0.0
    # Built once, not once per alternation round: it is a function of the duels and the
    # point count, and neither changes inside this fit.
    differences = duel_difference_matrix(duels, n_points)
    scaled_differences = differences * duel_slopes[:, None]
    for _ in range(3):
        for _ in range(60):
            gradient, likelihood_precision = _bt_derivatives(
                differences, scaled_differences, duel_slopes, utilities, sides, side_bias
            )
            step = np.linalg.solve(
                prior_precision + likelihood_precision,
                gradient - prior_precision @ (utilities - prior_means),
            )
            utilities = utilities + step
            if np.abs(step).max() < 1e-8:
                break
        if winner_sides is None or len(duels) < 12:
            break
        side_bias = _newton_side_bias(scaled_differences @ utilities, sides, side_bias)
    utility_cov = np.linalg.inv(prior_precision + likelihood_precision)
    return utilities, utility_cov, prior_precision, side_bias


def predict(
    gp_inputs,
    utilities,
    prior_means,
    utility_cov,
    prior_precision,
    query_inputs,
    query_prior_means,
    length_scales=None,
):
    """Posterior mean and marginal variance at the query inputs.

    Returns (mean, variance, cross-covariance, variance reduction). The last two are what
    a caller needs to get the covariance BETWEEN two queries, which is what the
    expected-information-gain acquisition in theme.schedule asks for.
    """
    cross_cov = kmat(query_inputs, gp_inputs, length_scales)
    mean = query_prior_means + cross_cov @ (prior_precision @ (utilities - prior_means))
    variance_reduction = prior_precision - prior_precision @ utility_cov @ prior_precision
    variance = np.maximum(SIGNAL_VARIANCE - np.einsum("ij,jk,ik->i", cross_cov, variance_reduction, cross_cov), 1e-9)
    return mean, variance, cross_cov, variance_reduction


def query_inputs_for(thetas, polarity):
    """(GP inputs, prior means) for a list of thetas at one polarity."""
    prior_mean = realized_space().prior_mean
    return (
        np.array([coords(theta, polarity) for theta in thetas]),
        np.array([prior_mean(theta, polarity) for theta in thetas]),
    )


def posterior_over(fit, thetas, polarity):
    """`predict` against a fit, at arbitrary thetas."""
    inputs, means = query_inputs_for(thetas, polarity)
    return predict(fit["X"], fit["f"], fit["m"], fit["cov"], fit["Ki"], inputs, means, fit.get("ls"))


def mean_utility_at(fit, thetas, polarity):
    """Posterior-mean utility at arbitrary thetas -- the analysis notebook's window in."""
    return posterior_over(fit, thetas, polarity)[0]


def posterior_joint(fit, thetas, polarity):
    """Mean and FULL covariance over candidates -- what P(best) needs.

    Marginal variances cannot answer "which of these is the best theme": candidates
    near each other in theme space share almost all their uncertainty, and ignoring
    that correlation would scatter the probability of being best across a cluster of
    effectively identical pages.
    """
    inputs, means = query_inputs_for(thetas, polarity)
    length_scales = fit.get("ls")
    cross_cov = kmat(inputs, fit["X"], length_scales)
    mean = means + cross_cov @ (fit["Ki"] @ (fit["f"] - fit["m"]))
    variance_reduction = fit["Ki"] - fit["Ki"] @ fit["cov"] @ fit["Ki"]
    cov = kmat(inputs, inputs, length_scales) - cross_cov @ variance_reduction @ cross_cov.T
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(len(thetas))
    return mean, cov


def h2(p):
    """Binary entropy in nats. Named for the frozen import in theme.schedule."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# Nine-node Gauss-Hermite quadrature, for the expectation of the binary entropy over a
# normal utility gap -- the conditional term of a duel's expected information gain. Named
# for the frozen imports in theme.schedule. The weights are normalized to sum to one, so
# the quadrature is an expectation under a standard normal rather than an integral.
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9)
GH_W = GH_W / GH_W.sum()


FIT_MEMO = {}
RTP_MEMO = {}


def cv_logloss(responses, rt_p, folds=5, seed=0):
    """Held-out log-loss of predicted duel outcomes at a given RT exponent.

    Cross-validation rather than marginal likelihood: the Laplace approximation makes
    the latter awkward to compare across likelihoods, while held-out predictive accuracy
    asks the question that matters -- does weighting a duel by how fast it was answered
    predict the NEXT answer better than ignoring the clock?
    """
    parsed = duels_from(responses, rt_p)
    if parsed is None:
        return None
    gp_inputs, duels, duel_slopes, prior_means, winner_sides = parsed
    if len(duels) < 5 * folds:
        return None
    order = np.random.default_rng(seed).permutation(len(duels))
    length_scales = ard_scales(gp_inputs, duels, duel_slopes)
    total_loss, n_scored = 0.0, 0
    for fold in range(folds):
        held_out = set(order[fold::folds].tolist())
        train = [i for i in range(len(duels)) if i not in held_out]
        if len(train) < 8:
            continue
        utilities, _cov, _precision, side_bias = fit_laplace(
            gp_inputs,
            [duels[i] for i in train],
            duel_slopes[train],
            prior_means,
            winner_sides[train],
            length_scales,
        )
        for i in held_out:
            winner, loser = duels[i]
            log_odds = duel_slopes[i] * (utilities[winner] - utilities[loser]) + side_bias * winner_sides[i]
            p_winner_wins = 1.0 / (1.0 + np.exp(-log_odds))
            total_loss -= np.log(max(p_winner_wins, 1e-9))
            n_scored += 1
    return None if n_scored == 0 else total_loss / n_scored


def rt_exponent(responses, grid=(0.0, 0.25, 0.5, 0.75), refit_every=25):
    """The RT exponent that predicts the next answer best, refit occasionally.

    Returns (best exponent, {exponent: held-out log-loss}). Zero is in the grid on
    purpose: if ignoring the clock predicts as well, the channel is noise dressed as
    evidence and the model should say so rather than carry a flattering heuristic.
    """
    rows = duel_rows(responses)
    bucket = len(rows) // refit_every
    # Refitting only once per `refit_every` duels is deliberate -- each refit costs a
    # hundred-odd GP fits -- but the bucket alone names how many duels have arrived and
    # not which, so two logs at the same count shared an exponent. The key names the
    # prefix the bucket refers to: one refit per bucket, and it belongs to the log that
    # asked for it.
    key = (bucket, log_fingerprint(rows[: bucket * refit_every]))
    if key in RTP_MEMO:
        return RTP_MEMO[key]
    scores = {}
    for exponent in grid:
        score = cv_logloss(responses, exponent)
        if score is not None:
            scores[exponent] = score
    result = (0.5, {}) if not scores else (min(scores, key=scores.get), scores)
    if len(RTP_MEMO) > 3:
        RTP_MEMO.pop(next(iter(RTP_MEMO)))
    RTP_MEMO[key] = result
    return result


def fitted(responses, rt_p=None):
    """The Laplace fit over a response log, memoized.

    The fit is a pure function of the log, three cells ask for the same one, and it is the
    cubic-cost step. A few entries rather than one, because the progress readout fits the
    log as it stood some duels ago and compares, which needs two fits alive at once -- so
    the key has to tell those two logs apart, which is what `log_fingerprint` is for.
    """
    rows = duel_rows(responses)
    if rt_p is None:
        rt_p = rt_exponent(responses)[0]
    key = (len(rows), rt_p, log_fingerprint(rows))
    if key in FIT_MEMO:
        return FIT_MEMO[key]
    parsed = duels_from(responses, rt_p)
    if parsed is None:
        return None
    gp_inputs, duels, duel_slopes, prior_means, winner_sides = parsed
    length_scales = ard_scales(gp_inputs, duels, duel_slopes)
    utilities, utility_cov, prior_precision, side_bias = fit_laplace(
        gp_inputs, duels, duel_slopes, prior_means, winner_sides, length_scales
    )
    fit = {
        "X": gp_inputs,
        "duels": duels,
        "lam": duel_slopes,
        "m": prior_means,
        "f": utilities,
        "cov": utility_cov,
        "Ki": prior_precision,
        "ls": length_scales,
        "delta": side_bias,
        "sides": winner_sides,
        "rt_p": rt_p,
        # What names this fit downstream. `diagnostics.best_set` memoizes per fit and had
        # only `id()` to key on, which is an address: a freed fit's slot gets reused and
        # answers for an unrelated one.
        "fingerprint": key,
    }
    if len(FIT_MEMO) > 4:
        FIT_MEMO.pop(next(iter(FIT_MEMO)))
    FIT_MEMO[key] = fit
    return fit
