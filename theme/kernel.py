"""The Gaussian-process kernel over theme space, and its length-scales.

One Matern-5/2 covariance serves both models in this package: the Bradley-Terry
preference GP in `preference` and the reaction-time regression in `legibility`. The axes
that move preference are the ones likely to move reading speed, so they share both the
kernel and the length-scales fitted from the duels.

The length-scales double as a METRIC: two themes are "the same theme" when they are close
in length-scale-scaled coordinates, which is what `spread_positions` measures difference
in, so a group is "themes the model cannot separate" rather than an arbitrary grid cell.
"""

import numpy as np

# Nine theme axes (named in space.AXES), plus polarity as a tenth, binary coordinate: a
# light page and a dark page are a block factor rather than an axis, and carrying polarity
# in the kernel lets learning transfer between them without conflating them.
N_AXES = 9
POLARITY_AXIS = N_AXES

# Length-scales are ARD: one per axis, estimated from the data rather than fixed, so axes
# the choices ignore get long scales and stop costing sample efficiency. Nine dimensions
# at ~100 duels is the binding constraint on how fast this converges, and ARD is the
# cheapest honest way to shrink the effective dimension. These are the isotropic defaults
# a thin log falls back to; ard_scales below estimates the rest.
DEFAULT_LENGTH_SCALES = np.array([0.35] * N_AXES + [0.9])

# Prior variance of the latent utility, i.e. a prior sd of 2 logits.
SIGNAL_VARIANCE = 4.0


def kmat(left, right, length_scales=None):
    """Matern-5/2 covariance between two sets of GP inputs, shape (len(left), len(right)).

    Named for the frozen import in theme.schedule; it is the kernel matrix.
    """
    scales = DEFAULT_LENGTH_SCALES if length_scales is None else length_scales
    squared_distance = (((left[:, None, :] - right[None, :, :]) / scales) ** 2).sum(-1)
    distance = np.sqrt(squared_distance + 1e-12)
    return SIGNAL_VARIANCE * (1 + np.sqrt(5) * distance + 5 * distance**2 / 3) * np.exp(-np.sqrt(5) * distance)


def ard_scales(gp_inputs, duels, duel_slopes):
    """Per-axis length-scales from a ridge-regularized linear Bradley-Terry fit.

    The principled route is maximizing the Laplace log-marginal-likelihood over ten
    log-length-scales, which costs a hundred-odd GP refits per trial and would make
    the instrument wait on itself. A linear BT model on the winner-minus-loser axis
    differences is the same question asked cheaply -- which axes move the choices --
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
        return DEFAULT_LENGTH_SCALES.copy()
    ard_weight = min(1.0, len(duels) / 160.0)
    scaled_differences = np.array(
        [
            (gp_inputs[winner] - gp_inputs[loser]) * slope
            for (winner, loser), slope in zip(duels, duel_slopes, strict=True)
        ]
    )
    coefficients = np.zeros(scaled_differences.shape[1])
    for _ in range(60):
        p_winner_wins = 1.0 / (1.0 + np.exp(-(scaled_differences @ coefficients)))
        gradient = scaled_differences.T @ (1.0 - p_winner_wins) - 2.0 * coefficients
        curvature = p_winner_wins * (1 - p_winner_wins)
        hessian = -(scaled_differences.T * curvature) @ scaled_differences - 2.0 * np.eye(scaled_differences.shape[1])
        step = np.linalg.solve(hessian, -gradient)
        coefficients = coefficients + step
        if np.abs(step).max() < 1e-10:
            break
    relevance = np.abs(coefficients) / max(float(np.abs(coefficients).max()), 1e-9)
    length_scales = np.clip(0.30 / np.sqrt(np.clip(relevance, 0.10, 1.0)), 0.25, 1.4)
    length_scales = (1.0 - ard_weight) * DEFAULT_LENGTH_SCALES + ard_weight * length_scales
    # Polarity is a block factor, not something relevance should shrink or stretch.
    length_scales[POLARITY_AXIS] = DEFAULT_LENGTH_SCALES[POLARITY_AXIS]
    return length_scales


def coords(theta, polarity):
    """A theme as a GP input: its nine axes, then the polarity coordinate."""
    return np.concatenate([np.asarray(theta, dtype=float), [1.0 if polarity == "night" else 0.0]])


def theta_length_scales(length_scales=None):
    """The nine theme-axis length-scales, without the polarity coordinate."""
    scales = DEFAULT_LENGTH_SCALES if length_scales is None else length_scales
    return scales[:N_AXES]


def scale_thetas(thetas, length_scales=None):
    """Thetas in length-scale-scaled coordinates, where distance means perceptual
    difference."""
    return np.asarray(thetas, dtype=float)[:, :N_AXES] * (1.0 / theta_length_scales(length_scales))


def spread_positions(scaled_points, initial, n_wanted):
    """Greedy max-min: extend `initial` to `n_wanted` positions, each as far as possible
    from those already taken.

    One implementation, because there were two -- the elite selection in `breeding` and
    the plateau readout in `diagnostics` -- and two copies of a greedy selection is two
    places for it to be subtly wrong. Positions index `scaled_points`; pass points that
    are already scaled, so the metric is the caller's choice of length-scales.

    An empty `initial` measures against position 0 without taking it, which is what the
    elite selection did and depends on.
    """
    picked = list(initial)
    limit = min(n_wanted, len(scaled_points))
    if len(picked) >= limit:
        return picked
    # Distance to the nearest already-taken point, carried forward rather than recomputed
    # against the whole taken set each step: taking one more point can only lower a
    # nearest-neighbour distance, so a running minimum is the same number for O(k n) work
    # instead of O(k^2 n).
    against = picked or [0]
    nearest = np.min(np.linalg.norm(scaled_points[:, None, :] - scaled_points[None, against, :], axis=-1), axis=1)
    while True:
        # Taken positions are masked rather than skipped, so a duplicate coordinate cannot
        # be taken twice.
        spread = nearest.copy()
        spread[against] = -1.0
        picked.append(int(np.argmax(spread)))
        if len(picked) >= limit:
            return picked
        if len(picked) == 1:
            # `initial` was empty, so the set measured against was position 0 without
            # taking it; from the first real pick onward it is `picked` itself.
            nearest = np.linalg.norm(scaled_points - scaled_points[picked[0]], axis=1)
        else:
            nearest = np.minimum(nearest, np.linalg.norm(scaled_points - scaled_points[picked[-1]], axis=1))
        against = picked
