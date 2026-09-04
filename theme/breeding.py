"""Where to look next: the candidate set one trial chooses between.

Reach and refinement together, every trial -- a standing low-discrepancy stratum for
global coverage, then elites, mutation and crossover for resolution around what the
posterior already likes.
"""

import numpy as np
from scipy.stats import qmc

from .kernel import LS0
from .preference import posterior_over, realized_space


# ---- candidate generation: global reach PLUS bred refinement --------------------
#
# The pool was 512 points drawn once with a fixed seed, and the only refinement was 48
# jittered children of the single argmax champion. Measured against a synthetic
# two-mode utility (see the escape test in the commit that added this), that design
# has good REACH -- 512 uniform points cover nine dimensions well enough for Thompson
# sampling to discover a distant better mode -- and poor RESOLUTION: nothing can sit
# between pool points except near one champion, at a fixed step size.
#
# The first attempt at fixing it replaced the pool with bred children and lost the
# reach, scoring *worse* in simulation. So candidates are now reach and refinement
# together, every trial:
#
#   standing    the full pool plus a SMALL fresh scrambled-Sobol block (64), advanced
#               by trial number. The pool is a codebook: revisiting the same points
#               concentrates information there and sharpens the posterior, where a
#               fully churning candidate set spreads every duel over ground never
#               seen again -- measured, a 512-per-trial immigrant flood scored worse
#               than no immigrants at all. Sixty-four is the measured sweet spot: a
#               trickle of genuinely new ground each trial, never enough to drown the
#               codebook, and enough that no region stays permanently unvisited.
#   elites      the best already-evaluated themes, chosen for spread as well as for
#               posterior mean, so refinement is not confined to one basin.
#   mutation    Gaussian children of each elite, per-axis sigma proportional to the
#               ARD length-scale: fine steps where utility actually turns, coarse
#               where the model has learned that nothing rides.
#   crossover   uniform per-axis recombination between elite pairs. Worth having
#               because the axes are semi-separable (ground, accent set, comment
#               recession, find-highlight): a good ground and a good accent set
#               recombine into a plausible page, the building-block case where
#               crossover earns its keep rather than adding noise.
#
# Infeasible children are dropped by the floors rather than penalized, so the whole
# candidate set is legible-by-construction.
def sobol_block(n_log2, offset_blocks):
    """A power-of-two block from one fixed scrambled Sobol sequence.

    Deterministic in the block index, so trial n always draws the same immigrants and
    successive trials continue the sequence instead of resampling the same clumps.
    random() rather than random_base2(): the latter also demands that the TOTAL drawn
    be a power of two, which a fast-forwarded engine cannot satisfy. n itself is a
    power of two, which is what the balance property needs.
    """
    _n = 2**n_log2
    _eng = qmc.Sobol(d=9, scramble=True, seed=0xC0FFEE)
    _skip = (offset_blocks * _n) % 65536
    if _skip:
        _eng.fast_forward(_skip)
    return _eng.random(_n)


class _CandidateSet:
    """Candidates in proposal order, deduplicated, each already known to be legible.

    Proposals arrive in batches because realizing a theme costs one colour-library call per
    batch rather than per theme (measured: a call converting one colour costs 312 us, and a
    call converting sixty-four costs 325 us). Realizing four hundred candidates one at a
    time spent 3.8 s of a 4 s trial inside that library's argument validation.

    Order is preserved and duplicates are dropped on first sight, because the index where
    the standing stratum ends is what the explore/exploit split is declared against.
    """

    def __init__(self, polarity):
        self.polarity = polarity
        self.entries = []
        self._seen = set()

    def offer(self, thetas, themes=None):
        """Add each theta that is new and legible. `themes` skips realization for the pool,
        whose members are realized once at startup and never change."""
        thetas = [np.clip(np.asarray(t, dtype=float), 0.0, 1.0) for t in thetas]
        fresh = [t for t in thetas if tuple(np.round(t, 4)) not in self._seen]
        if themes is None:
            realize_many = realized_space().realize_many
            themes = realize_many(np.array(fresh), self.polarity) if fresh else []
            offered = zip(fresh, themes, strict=True)
        else:
            offered = zip(thetas, themes, strict=True)
        for theta, theme in offered:
            key = tuple(np.round(theta, 4))
            if theme is None or key in self._seen:
                continue
            self._seen.add(key)
            self.entries.append((theta, theme))

    @property
    def thetas(self):
        return [theta for theta, _theme in self.entries]


def candidates(fit, polarity, nprng, n_trial=0, n_elite=10, n_mut=20, n_cross=48, imm_log2=6):
    """(candidates, index where the standing global stratum ends) for this trial."""
    pool = _CandidateSet(polarity)
    standing = realized_space().POOL[polarity]
    pool.offer([t for t, _ in standing], [theme for _, theme in standing])
    pool.offer(list(sobol_block(imm_log2, n_trial)))
    n_standing = len(pool.entries)
    if fit is None:
        return pool.entries, n_standing

    want = 1.0 if polarity == "night" else 0.0
    archive = [x[:9] for x in fit["X"] if abs(x[9] - want) < 0.5]
    seed_set = archive + pool.thetas
    mu = posterior_over(fit, seed_set, polarity)[0]
    ls = fit.get("ls")
    top = np.argsort(-mu)[: 6 * n_elite]
    # Elites for spread as well as for mean: the best few, then the most different among
    # the rest of the leaders, so refinement is not confined to one basin. Deliberately NOT
    # Thompson-sampled elites: tried, and measured clearly worse (reach 3/12 runs,
    # t = -2.6). Refining around a high-variance region spends the mutation budget on noise
    # and displaces elites that are actually good; explore belongs in the standing stratum,
    # refine belongs where the mean is already high.
    keep = [int(i) for i in top[: n_elite // 2]]
    weights = 1.0 / (LS0[:9] if ls is None else ls[:9])
    scaled = np.array([np.asarray(seed_set[int(i)]) * weights for i in top])
    top_list = list(top)
    while len(keep) < n_elite and len(keep) < len(top):
        chosen = [top_list.index(i) for i in keep if i in top_list] or [0]
        spread = np.min(np.linalg.norm(scaled[:, None, :] - scaled[None, chosen, :], axis=-1), axis=1)
        spread[chosen] = -1.0
        keep.append(int(top[int(np.argmax(spread))]))
    elites = [np.asarray(seed_set[i]) for i in keep]

    # Mutation sigma scales with the ARD length-scale per axis: fine steps where utility
    # actually turns, coarse where the model has learned that nothing rides.
    sigma = 0.25 * (LS0[:9] if ls is None else ls[:9])
    bred = []
    for elite in elites:
        bred.append(elite)
        bred.extend(np.clip(elite[None, :] + nprng.normal(0, sigma, (n_mut, 9)), 0, 1))
    if len(elites) >= 2:
        # Uniform per-axis recombination. Worth having because the axes are semi-separable
        # -- ground, accent set, comment recession, find-highlight -- so a good ground and a
        # good accent set recombine into a plausible page, which is the building-block case
        # where crossover earns its keep rather than adding noise.
        for _ in range(n_cross):
            first, second = nprng.choice(len(elites), 2, replace=False)
            mask = nprng.random(9) < 0.5
            bred.append(np.where(mask, elites[first], elites[second]))
    pool.offer(bred)
    return pool.entries, n_standing
