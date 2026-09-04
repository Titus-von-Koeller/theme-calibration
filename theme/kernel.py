"""The Gaussian-process kernel over theme space, and its length-scales.

One Matern-5/2 covariance serves both models in this package: the Bradley-Terry
preference GP in `preference` and the reaction-time regression in `legibility`. The axes
that move preference are the ones likely to move reading speed, so they share both the
kernel and the length-scales fitted from the duels.

Extracted from calibrate-aesthetics.py's model cell on 2026-09-04 -- the behaviour, the
constants and the reasoning comments are unchanged; only marimo's cell-local underscores
are gone.
"""

import numpy as np

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
