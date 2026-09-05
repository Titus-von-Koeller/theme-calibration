"""The theta nearest to a palette that was never a point in theme space.

The hand-chosen Horizon layer Titus lived in for a week is a palette, not a theta: it was
made by walking the theme's own hues to a contrast bar, not by the search. To duel it
against a measured theme inside the model, the model needs coordinates for it, so this
finds the theta whose REALIZED palette is closest to the hand palette in CAM16-UCS, and
says how close. The lived-duel row carries both -- the actual hexes he saw, and the
fitted theta the model reads -- with the fit distance beside them, so nobody later mistakes
an approximation for the thing it approximates.

Search rather than inversion: realize() walks each role's lightness to a contrast bar and
refuses infeasible points, so there is no closed form to invert. The candidate set is the
standing pool plus a Sobol block, then a few rounds of Gaussian refinement around the best,
every round realized in one batched call.
"""

import numpy as np

from .breeding import sobol_block
from .color import hex_to_rgb, rgb_to_ucs
from .space import FIND_HUE_AXIS, FIND_SALIENCE_AXIS, POOL, realize_many

#: The roles a palette is matched on. The find fill is left out: the hand layer's highlight
#: had no salience concept, and the fit would otherwise chase an alpha nobody measured.
MATCHED_ROLES = ("ground", "keyword", "function", "string", "ink", "comment")

#: Refinement: rounds of Gaussian children around the best few incumbents, tightening each
#: round. Several incumbents rather than one because the match landscape has flat ridges --
#: comment recession barely enters the distance -- so a single incumbent can sit in a
#: shallow basin next to the true one. Measured on a planted theta: one incumbent over four
#: rounds landed 0.9 to 1.3 dE off; eight incumbents over eight rounds land inside a hex
#: step.
#:
#: The find axes are not searched; every candidate carries these. They enter the distance
#: nowhere, so moving them is noise -- and since the highlight baseline moved into realize()
#: it is noise that gets children refused: with them free, a planted theta was recovered to
#: 0.91 dE instead of inside a hex step, and with each incumbent keeping its own seed values
#: to 0.77, because a seed's highlight is feasible on the seed's page and not necessarily on
#: the page the search is walking toward. So one setting for all, chosen for feasibility:
#: measured over every pool page with the find axes overridden, (hue 0.5, salience 0.6) is
#: realizable on 100% of day pages and (0.95, 0.6) on 100% of night pages, where salience
#: 1.0 drops to 55-92% because a loud fill starts failing the ink-on-fill floor.
MATCHING_FIND_AXES = {"day": (0.5, 0.6), "night": (0.95, 0.6)}
REFINE_ELITES = 8
REFINE_ROUNDS = 8
REFINE_CHILDREN_PER_ELITE = 64
REFINE_SIGMA = 0.15
REFINE_SHRINK = 0.6


def palette_distances(themes, palette):
    """RMS CAM16-UCS distance over MATCHED_ROLES from each realized theme to `palette`.

    Refused themes (None) come back as infinity so they never win.
    """
    target = rgb_to_ucs(hex_to_rgb([palette[role] for role in MATCHED_ROLES]))
    distances = np.full(len(themes), np.inf)
    built = [i for i, theme in enumerate(themes) if theme is not None]
    if built:
        hexes = [themes[i][role] for i in built for role in MATCHED_ROLES]
        ucs = rgb_to_ucs(hex_to_rgb(hexes)).reshape(len(built), len(MATCHED_ROLES), 3)
        distances[built] = np.sqrt(((ucs - target[None]) ** 2).sum(-1).mean(-1))
    return distances


def nearest_theta(palette, polarity, seed=0):
    """(theta, RMS dE) of the realizable theme closest to `palette` at this polarity."""
    rng = np.random.default_rng(seed)
    thetas = np.array([theta for theta, _theme in POOL[polarity]] + list(sobol_block(11, 0)), dtype=float)
    thetas[:, [FIND_HUE_AXIS, FIND_SALIENCE_AXIS]] = MATCHING_FIND_AXES[polarity]
    elites, distances = _best_few(thetas, palette, polarity, REFINE_ELITES)
    searched = np.ones(thetas.shape[1])
    searched[[FIND_HUE_AXIS, FIND_SALIENCE_AXIS]] = 0.0
    sigma = REFINE_SIGMA
    for _ in range(REFINE_ROUNDS):
        noise = rng.normal(0.0, sigma, (len(elites), REFINE_CHILDREN_PER_ELITE, thetas.shape[1])) * searched
        children = np.clip(elites[:, None, :] + noise, 0.0, 1.0).reshape(-1, thetas.shape[1])
        elites, distances = _best_few(np.vstack([elites, children]), palette, polarity, REFINE_ELITES)
        sigma *= REFINE_SHRINK
    return [round(float(v), 6) for v in elites[0]], float(distances[0])


def _best_few(thetas, palette, polarity, count):
    """The `count` thetas nearest to the palette, nearest first, with their distances."""
    distances = palette_distances(realize_many(thetas, polarity), palette)
    order = np.argsort(distances)[:count]
    return thetas[order], distances[order]
