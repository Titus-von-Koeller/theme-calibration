"""The preference and legibility models, as one importable namespace.

The implementation lives in five modules, one concern each:

    kernel       the Matern-5/2 covariance over theme space and its ARD length-scales
    preference   the Bradley-Terry preferential GP fitted to the duels
    breeding     the candidate set one trial chooses between
    legibility   the GP regression on log time-to-click
    diagnostics  P(best), axis consensus, progress, and the permutation tests

This module stays because it is the import surface the rest of the package was written
against -- `theme.schedule` imports ten names from it -- and because it is the seam the
test suite substitutes: `tests/conftest.py` replaces `POOL`, `prior_mean` and
`realize_many` HERE, and `preference.realized_space()` reads them back off this module at
call time so the stub is seen. Keep those three bindings.

It re-exports rather than aliases, so `theme.model.fitted is theme.preference.fitted`, and
mutating `theme.model.FIT_MEMO` mutates the one memo there is.

Two names read here as abbreviations rather than as what they hold -- `kmat` for the
kernel matrix and `h2` for binary entropy, plus `GH_X`/`GH_W` for the Gauss-Hermite nodes
and weights. They are spelled that way because `theme.schedule` imports them and the
public interface is frozen; renaming them is a separate, serialized commit.
"""

from .breeding import candidates, sobol_block
from .diagnostics import (
    BEST_MEMO,
    SURF_MEMO,
    axis_consensus,
    best_set,
    factor_effect,
    progress_report,
    spread_out,
    surface_effect,
)
from .kernel import (
    DEFAULT_LENGTH_SCALES,
    N_AXES,
    POLARITY_AXIS,
    SIGNAL_VARIANCE,
    ard_scales,
    coords,
    kmat,
    scale_thetas,
    spread_positions,
)
from .legibility import rt_at, rt_fit, rt_penalty
from .preference import (
    FIT_MEMO,
    GH_W,
    GH_X,
    RTP_MEMO,
    cv_logloss,
    duel_rows,
    duels_from,
    fit_laplace,
    fitted,
    h2,
    mean_utility_at,
    posterior_joint,
    posterior_over,
    predict,
    rt_exponent,
)

# The realized-theme layer, re-exported so it can be substituted here. See the module
# docstring and `preference.realized_space`.
from .space import POOL, prior_mean, realize_many

__all__ = [
    "BEST_MEMO",
    "DEFAULT_LENGTH_SCALES",
    "FIT_MEMO",
    "GH_W",
    "GH_X",
    "N_AXES",
    "POLARITY_AXIS",
    "POOL",
    "RTP_MEMO",
    "SIGNAL_VARIANCE",
    "SURF_MEMO",
    "ard_scales",
    "axis_consensus",
    "best_set",
    "candidates",
    "coords",
    "cv_logloss",
    "duel_rows",
    "duels_from",
    "factor_effect",
    "fit_laplace",
    "fitted",
    "h2",
    "kmat",
    "mean_utility_at",
    "posterior_joint",
    "posterior_over",
    "predict",
    "prior_mean",
    "progress_report",
    "realize_many",
    "rt_at",
    "rt_exponent",
    "rt_fit",
    "rt_penalty",
    "scale_thetas",
    "sobol_block",
    "spread_out",
    "spread_positions",
    "surface_effect",
]
