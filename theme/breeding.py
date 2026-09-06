"""Where to look next: the candidate set one trial chooses between.

Global reach PLUS bred refinement.

The pool was 512 points drawn once with a fixed seed, and the only refinement was 48
jittered children of the single argmax champion. Measured against a synthetic two-mode
utility (see the escape test in the commit that added this), that design has good REACH --
512 uniform points cover nine dimensions well enough for Thompson sampling to discover a
distant better mode -- and poor RESOLUTION: nothing can sit between pool points except
near one champion, at a fixed step size.

The first attempt at fixing it replaced the pool with bred children and lost the reach,
scoring *worse* in simulation. So candidates are now reach and refinement together, every
trial:

  standing    the full pool plus a SMALL fresh scrambled-Sobol block (64), advanced by
              trial number. The pool is a codebook: revisiting the same points
              concentrates information there and sharpens the posterior, where a fully
              churning candidate set spreads every duel over ground never seen again --
              measured, a 512-per-trial immigrant flood scored worse than no immigrants at
              all. Sixty-four is the measured sweet spot: a trickle of genuinely new
              ground each trial, never enough to drown the codebook, and enough that no
              region stays permanently unvisited.
  elites      the best already-evaluated themes, chosen for spread as well as for
              posterior mean, so refinement is not confined to one basin.
  mutation    Gaussian children of each elite, per-axis sigma proportional to the ARD
              length-scale: fine steps where utility actually turns, coarse where the
              model has learned that nothing rides.
  crossover   uniform per-axis recombination between elite pairs. Worth having because
              the axes are semi-separable (ground, accent set, comment recession,
              find-highlight): a good ground and a good accent set recombine into a
              plausible page, the building-block case where crossover earns its keep
              rather than adding noise.

Infeasible children are dropped by the floors rather than penalized, so the whole
candidate set is legible-by-construction.
"""

import numpy as np
from scipy.stats import qmc

from .kernel import N_AXES, POLARITY_AXIS, scale_thetas, spread_positions, theta_length_scales
from .preference import posterior_over, realized_space

# One fixed scrambled Sobol sequence, wrapped every 65536 draws so a long run stays inside
# the engine's balanced regime. At 64 immigrants a trial the wrap comes round after 1024
# trials, after which the immigrant blocks repeat.
SOBOL_SEED = 0xC0FFEE
SOBOL_WRAP = 65536


def sobol_block(n_log2, offset_blocks):
    """A power-of-two block from one fixed scrambled Sobol sequence.

    Deterministic in the block index, so trial n always draws the same immigrants and
    successive trials continue the sequence instead of resampling the same clumps -- until
    the wrap at SOBOL_WRAP draws, after which the blocks repeat.
    random() rather than random_base2(): the latter also demands that the TOTAL drawn
    be a power of two, which a fast-forwarded engine cannot satisfy. n itself is a
    power of two, which is what the balance property needs.
    """
    block_size = 2**n_log2
    engine = qmc.Sobol(d=N_AXES, scramble=True, seed=SOBOL_SEED)
    skip = (offset_blocks * block_size) % SOBOL_WRAP
    if skip:
        engine.fast_forward(skip)
    return engine.random(block_size)


class _CandidateSet:
    """Candidates in proposal order, deduplicated, each already known to be legible.

    Proposals arrive in batches because realizing a theme costs one colour-library call per
    batch rather than per theme (measured: a call converting one colour costs 312 us, and a
    call converting sixty-four costs 325 us). Realizing four hundred candidates one at a
    time spent 3.8 s of a 4 s trial inside that library's argument validation.

    Order is preserved and duplicates are dropped on first sight, because the index where
    the standing stratum ends is what the explore/exploit split is declared against.
    """

    #: The three places a candidate can come from, in the order they are offered.
    STRATA = ("pool", "fresh", "bred")

    def __init__(self, polarity):
        self.polarity = polarity
        self.entries = []
        self._seen = set()
        self.strata = []

    def offer(self, thetas, themes=None, stratum="bred"):
        """Add each theta that is new and legible. `themes` skips realization for the pool,
        whose members are realized once at startup and never change. `stratum` names where
        the batch came from, so a verdict can say whether its leader is a standing grid
        point, a fresh immigrant, or a bred child."""
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
            self.strata.append(stratum)

    @property
    def thetas(self):
        return [theta for theta, _theme in self.entries]


def _elites(fit, polarity, seed_set, n_elite, length_scales):
    """The best themes so far, chosen for spread as well as for posterior mean.

    The best few, then the most different among the rest of the leaders, so refinement is
    not confined to one basin. Deliberately NOT Thompson-sampled elites: tried, and
    measured clearly worse (reach 3/12 runs, t = -2.6). Refining around a high-variance
    region spends the mutation budget on noise and displaces elites that are actually good;
    explore belongs in the standing stratum, refine belongs where the mean is already high.
    """
    mean_utility = posterior_over(fit, seed_set, polarity)[0]
    leaders = np.argsort(-mean_utility)[: 6 * n_elite]
    scaled_leaders = scale_thetas([seed_set[int(i)] for i in leaders], length_scales)
    best_half = list(range(min(n_elite // 2, len(leaders))))
    picked = spread_positions(scaled_leaders, best_half, n_elite)
    return [np.asarray(seed_set[int(leaders[p])]) for p in picked]


def _breed(elites, rng, n_mutants, n_crossovers, length_scales):
    """Each elite, its Gaussian children, and uniform per-axis recombinations of pairs.

    Mutation sigma scales with the ARD length-scale per axis: fine steps where utility
    actually turns, coarse where the model has learned that nothing rides. Crossover is
    worth having because the axes are semi-separable -- ground, accent set, comment
    recession, find-highlight -- so a good ground and a good accent set recombine into a
    plausible page, which is the building-block case where crossover earns its keep rather
    than adding noise.
    """
    mutation_sigma = 0.25 * theta_length_scales(length_scales)
    bred = []
    for elite in elites:
        bred.append(elite)
        bred.extend(np.clip(elite[None, :] + rng.normal(0, mutation_sigma, (n_mutants, N_AXES)), 0, 1))
    if len(elites) >= 2:
        for _ in range(n_crossovers):
            first, second = rng.choice(len(elites), 2, replace=False)
            from_first = rng.random(N_AXES) < 0.5
            bred.append(np.where(from_first, elites[first], elites[second]))
    return bred


def candidates(fit, polarity, rng, n_trial=0, n_elite=10, n_mutants=20, n_crossovers=48, immigrants_log2=6):
    """(candidates, index where the standing global stratum ends) for this trial."""
    entries, n_standing, _strata = candidates_with_strata(
        fit, polarity, rng, n_trial, n_elite, n_mutants, n_crossovers, immigrants_log2
    )
    return entries, n_standing


def candidates_with_strata(
    fit, polarity, rng, n_trial=0, n_elite=10, n_mutants=20, n_crossovers=48, immigrants_log2=6
):
    """`candidates`, plus one stratum name per candidate: "pool" for the standing grid,
    "fresh" for this trial's Sobol immigrants, "bred" for children of the elites.

    The verdict reads the strata to say where its leader and shelf came from. Once the model
    has learned anything the leader and the whole shelf are bred -- children refine the
    elites, so they outrank the uniform grid at the top by construction -- and a leader
    from the pool or the immigrants means the refinements are not yet beating coverage.
    That used to be a hunch, and this makes it a count.
    """
    pool = _CandidateSet(polarity)
    standing = realized_space().POOL[polarity]
    pool.offer([theta for theta, _theme in standing], [theme for _theta, theme in standing], stratum="pool")
    pool.offer(list(sobol_block(immigrants_log2, n_trial)), stratum="fresh")
    n_standing = len(pool.entries)
    if fit is None:
        return pool.entries, n_standing, pool.strata

    this_polarity = 1.0 if polarity == "night" else 0.0
    archive = [x[:N_AXES] for x in fit["X"] if abs(x[POLARITY_AXIS] - this_polarity) < 0.5]
    length_scales = fit.get("ls")
    elites = _elites(fit, polarity, archive + pool.thetas, n_elite, length_scales)
    pool.offer(_breed(elites, rng, n_mutants, n_crossovers, length_scales), stratum="bred")
    return pool.entries, n_standing, pool.strata
