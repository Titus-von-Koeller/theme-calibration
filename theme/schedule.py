"""What the next trial should be.

The polarity blocks, the arm schedule inside a run, the surface assignment, and the trial
generator that decides which two themes to show and on what page.

Trial generation is a pure function of the trial number and the responses before it. That
is not a nicety: the recorder rebuilds a trial from the log at record time rather than
trusting what the page sent back, so purity here is what makes every archived row describe
the stimulus that was actually shown.
"""

import random

import numpy as np

from .model import (
    GH_W,
    GH_X,
    candidates,
    coords,
    fitted,
    h2,
    kmat,
    posterior_over,
    rt_at,
    rt_fit,
)
from .space import POOL, conspicuous_enough, realize_many
from .stimulus import DUEL_WIDTH, READING_PX, SURFACES

#: Trials per polarity block, and how the block divides into runs of one kind.
BLOCK = 24
DUELS_PER_BLOCK = 16
PROBES_PER_BLOCK = 4

#: Duels needed before the model has enough to probe; until then every trial is a duel.
BOOTSTRAP_DUELS = 6

#: Lines per page. A page, not a snippet: fourteen lines centred on a very large display is
#: an island spanning a quarter of the field, and a probe needs distractors to reject --
#: accuracy saturated at 100% over twenty probes, and a 28-line page offers ~97 identifiers
#: to reject instead of ~28.
PAGE_LINES = 28

#: Theme-space axes the find hunts sweep: 7 is find hue, 8 is find salience.
FIND_HUE_AXIS = 7
FIND_SALIENCE_AXIS = 8

#: Share of duels drawn as a uniform feasible pair instead of from the model, as insurance
#: against model misspecification.
UNIFORM_DUEL_SHARE = 0.07

#: Share of duels run as champion-vs-worst anchors: engagement breathers that double as
#: sanity checks on the fit.
ANCHOR_DUEL_SHARE = 0.054

#: Share of probes and hunts left un-targeted. An acquisition that only ever probes its own
#: uncertainty can leave a region unvisited that it is wrongly confident about.
UNTARGETED_SHARE = 0.25


def schedule_mode(n, n_duels):
    """Twenty-four-trial polarity blocks, each a run of sixteen duels, then four
    comprehension probes, then four find hunts — same-kind trials batched so one
    instruction serves a whole run and no click is spent re-reading. All-duel until the
    model has something to probe."""
    polarity = ("day", "night")[(n // BLOCK) % 2]
    if n_duels < BOOTSTRAP_DUELS:
        return polarity, "duel"
    slot = n % BLOCK
    if slot < DUELS_PER_BLOCK:
        return polarity, "duel"
    if slot < DUELS_PER_BLOCK + PROBES_PER_BLOCK:
        return polarity, "comprehension"
    return polarity, "search"


def duel_surface(n, n_duels):
    """Which of the three surfaces duel n is shown on.

    NOT `n % 3`. The schedule's block is 24 trials of which the first 16 are duels, and
    3 divides 24, so a modular rotation never de-phases: editor takes 6 of every 16
    duels and the other two 5 each, forever, and slot 0 is editor every single run. The
    log showed exactly that lock -- 6/5/5 by day, 12/10/10 by night. It is both a
    standing 20% over-sample of one surface and a hard confound between surface and
    position within the run, where first-of-run means the freshest eyes and the largest
    adaptation step from whatever was on screen before.

    Instead: consecutive groups of three duels each get a shuffled permutation of the
    three surfaces. Balance is exact every three duels rather than asymptotic, and the
    shuffle decorrelates surface from run position. Deterministic in the duel index, so
    replaying a log reconstructs the same assignment."""
    in_block = (n // BLOCK) * DUELS_PER_BLOCK + min(n % BLOCK, DUELS_PER_BLOCK)
    duel_index = n if n_duels < BOOTSTRAP_DUELS else in_block
    rotation = list(SURFACES)
    random.Random(0xC0FFEE + duel_index // 3).shuffle(rotation)
    return rotation[duel_index % 3]


def run_info(n, n_duels):
    """(polarity, mode, position within the run, run length) for trial n."""
    polarity, arm = schedule_mode(n, n_duels)
    if n_duels < BOOTSTRAP_DUELS:
        return polarity, arm, min(n_duels, BOOTSTRAP_DUELS - 1), BOOTSTRAP_DUELS
    slot = n % BLOCK
    if slot < DUELS_PER_BLOCK:
        return polarity, arm, slot, DUELS_PER_BLOCK
    if slot < DUELS_PER_BLOCK + PROBES_PER_BLOCK:
        return polarity, arm, slot - DUELS_PER_BLOCK, PROBES_PER_BLOCK
    return polarity, arm, slot - DUELS_PER_BLOCK - PROBES_PER_BLOCK, PROBES_PER_BLOCK


def page_seed(n):
    """The seed picking trial n's code page. Every arm draws its page the same way, so a
    page is comparable across arms at the same trial number."""
    return n * 7919 + 17


# Deterministic given the log, so a memo keyed by trial number is a pure cache: several
# callers ask for the same trial and pay for one fit.
TRIAL_MEMO = {}


def _uniform_pair(pool, rng):
    """Two distinct feasible themes drawn uniformly -- the model's control condition."""
    return [pool[i] for i in rng.sample(range(len(pool)), 2)]


def _thompson_champion(mean, variance, n_standing, rng, numpy_rng):
    """Index of one arm: the argmax of a Thompson draw within a declared stratum.

    Stratified Thompson: the explore/exploit split is DECLARED, not left to however many
    candidates each stratum happened to contribute. Measured: adding local children
    silently pulled the sampled argmax toward the incumbent's basin and cost reach (paired
    diff -0.14 on the two-mode test). Drawing the champion arm from the global stratum half
    the time restores it without giving up refinement.
    """
    sampled = mean + np.sqrt(variance) * numpy_rng.standard_normal(len(mean))
    explore = rng.random() < 0.5 and n_standing < len(mean)
    start, stop = (n_standing, len(mean)) if explore else (0, n_standing)
    return start + int(np.argmax(sampled[start:stop]))


def _most_informative_challenger(fit, thetas, polarity, mean, variance, cross, precision, champion):
    """Index of the arm whose duel against `champion` has the largest expected information
    gain about the outcome: the entropy of the predicted win probability, less its expected
    entropy once the duel is observed, integrated over the posterior by Gauss-Hermite."""
    covariance = kmat(
        np.array([coords(theta, polarity) for theta in thetas]),
        np.array([coords(thetas[champion], polarity)]),
        fit.get("ls"),
    )[:, 0] - np.einsum("ij,jk,k->i", cross, precision, cross[champion])
    utility_gap = mean - mean[champion]
    gap_variance = np.maximum(variance + variance[champion] - 2 * covariance, 1e-9)
    win_probability = 1.0 / (1.0 + np.exp(-utility_gap / np.sqrt(1 + np.pi * gap_variance / 8)))
    conditional = (
        h2(1.0 / (1.0 + np.exp(-(utility_gap[:, None] + np.sqrt(gap_variance)[:, None] * GH_X[None, :])))) @ GH_W
    )
    information_gain = h2(win_probability) - conditional
    information_gain[champion] = -1.0
    return int(np.argmax(information_gain))


def _duel_arms(fit, polarity, n, pool, rng, numpy_rng):
    """(kind, (theta, theme), (theta, theme)) -- the two candidates a duel compares.

    Candidates are bred fresh (see candidates() -- elites, mutation, crossover, Sobol
    immigrants), one arm is a Thompson sample's argmax over them (explore where the optimum
    might be), the other the challenger with maximal expected information gain about the
    duel's outcome.
    """
    if fit is None or rng.random() < UNIFORM_DUEL_SHARE:
        first, second = _uniform_pair(pool, rng)
        return "probe", first, second

    bred, n_standing = candidates(fit, polarity, numpy_rng, n_trial=n)
    thetas = [theta for theta, _theme in bred]
    themes = [theme for _theta, theme in bred]
    mean, variance, cross, precision = posterior_over(fit, thetas, polarity)

    if rng.random() < ANCHOR_DUEL_SHARE:
        champion, challenger = int(np.argmax(mean)), int(np.argmin(mean))
        kind = "anchor"
    else:
        champion = _thompson_champion(mean, variance, n_standing, rng, numpy_rng)
        challenger = _most_informative_challenger(fit, thetas, polarity, mean, variance, cross, precision, champion)
        kind = "eig"
    return kind, (thetas[champion], themes[champion]), (thetas[challenger], themes[challenger])


def _rounded(theta):
    return [round(float(value), 6) for value in theta]


def _duel_trial(n, polarity, fit, pool, rng, numpy_rng, n_duels):
    kind, (theta_a, theme_a), (theta_b, theme_b) = _duel_arms(fit, polarity, n, pool, rng, numpy_rng)
    surface = duel_surface(n, n_duels)
    return {
        "mode": "duel",
        # Both arms share surface and page: a duel varies the theme, nothing else. A duel
        # is judged full screen, so the sample must BE a page -- a fourteen-line block
        # adrift in half a screen says nothing about how a screen of this theme reads. Long
        # enough to fill the half, and smaller type, which is also what a full screen at
        # this pixel density looks like in the editor itself. Both stay logged as stimulus
        # parameters.
        "snippet_width": DUEL_WIDTH,
        "snippet_lines": PAGE_LINES,
        "surface": surface,
        "kind": kind,
        "polarity": polarity,
        "theta_a": _rounded(theta_a),
        "theta_b": _rounded(theta_b),
        "theme_a": theme_a,
        "theme_b": theme_b,
        "snippet": page_seed(n),
        # The size this surface is read at (see READING_PX): the stimulus is then the thing
        # the answer is for, rather than a shrunken proxy for it.
        "code_px": READING_PX[surface],
        "swap": rng.random() < 0.5,
    }


def _probe_theme(fit, history, polarity, pool, rng, numpy_rng, n):
    """Which theme a comprehension probe is run on.

    Among the pages that might plausibly be lived in (the top of a Thompson draw), probe
    the one whose reading time the model is least sure of. Probing a page that would never
    be chosen measures legibility nobody will use; probing the champion again measures what
    is already known.
    """
    if fit is None or rng.random() <= UNTARGETED_SHARE:
        return pool[rng.randrange(len(pool))]

    bred = candidates(fit, polarity, numpy_rng, n_trial=n)[0]
    mean, variance = posterior_over(fit, [theta for theta, _ in bred], polarity)[:2]
    sampled = mean + np.sqrt(variance) * numpy_rng.standard_normal(len(mean))
    plausible = np.argsort(-sampled)[: max(8, len(sampled) // 20)]
    reading_time = rt_fit(history, polarity, fit.get("ls"))
    if reading_time is None:
        return bred[int(np.argmax(sampled))]
    predicted_variance = rt_at(reading_time, [bred[int(i)][0] for i in plausible], polarity)[1]
    return bred[int(plausible[int(np.argmax(predicted_variance))])]


def _comprehension_trial(n, history, polarity, fit, pool, rng, numpy_rng):
    theta, theme = _probe_theme(fit, history, polarity, pool, rng, numpy_rng, n)
    return {
        "mode": "comprehension",
        "surface": "editor",
        # Probes require a CALL-site target: a name at its `def` sits at a line start, at a
        # predictable indent, one or two to a page, and is found far faster than the same
        # name inside an expression. Mixing the two kinds puts a step in the task's
        # difficulty, and reaction time then measures which kind was drawn rather than how
        # the theme reads -- 12 of 60 probe pages were handing out the easy kind.
        "target_kind": "call",
        "snippet_lines": PAGE_LINES,
        "kind": "task",
        "polarity": polarity,
        "theta_a": _rounded(theta),
        "theme_a": theme,
        "snippet": page_seed(n),
        # A size code is actually read at. 15 was not one: the editors in use sit at 14 and
        # notebook code cells at 16, so a legibility constraint measured at 15 was
        # constraining a size that never appears. These arms run on the editor surface, so
        # they take the editor's size. The per-size baseline in rt_fit absorbs the step from
        # the earlier 15/16 trials.
        "code_px": READING_PX["editor"],
    }


def _champion_theta(fit, polarity, pool, rng, numpy_rng, n):
    """The page a find hunt holds fixed while it sweeps the find axes."""
    if fit is None:
        return np.array(pool[rng.randrange(len(pool))][0])
    bred = candidates(fit, polarity, numpy_rng, n_trial=n)[0]
    mean = posterior_over(fit, [theta for theta, _ in bred], polarity)[0]
    return np.array(bred[int(np.argmax(mean))][0])


def _find_axis_grid(base):
    """The champion's page with the two find axes swept over a 7x7 grid."""
    grid = []
    for hue in np.linspace(0.05, 0.95, 7):
        for salience in np.linspace(0.05, 0.95, 7):
            point = base.copy()
            point[FIND_HUE_AXIS], point[FIND_SALIENCE_AXIS] = hue, salience
            grid.append(point)
    return grid


def _conspicuous_grid(grid, polarity):
    """The grid points whose highlight is loud enough for a timed hunt to mean anything.

    Every candidate the sweep may choose has to clear the conspicuity floor first. Without
    that the sampler picks the faintest highlight in the grid, because an unexplored corner
    is where a GP's variance is highest -- and a highlight that cannot be seen measures how
    long someone was willing to hunt, not what the theme costs. Filtering the grid BEFORE
    the acquisition, rather than rejecting afterwards, keeps the choice the best available
    one rather than the first acceptable one.
    """
    themes = realize_many(np.array(grid), polarity)
    pairs = list(zip(grid, themes, strict=True))
    usable = [(theta, theme) for theta, theme in pairs if conspicuous_enough(theme, polarity)]
    if usable:
        return usable
    # Nothing on this champion's page can carry a loud enough highlight. Take the loudest
    # that exists rather than showing an unmeasurable trial.
    built = [(theta, theme) for theta, theme in pairs if theme is not None]
    return sorted(built, key=lambda pair: -pair[1]["salience"])[:1]


def _search_trial(n, history, polarity, fit, pool, rng, numpy_rng):
    base = _champion_theta(fit, polarity, pool, rng, numpy_rng, n)
    usable = _conspicuous_grid(_find_axis_grid(base), polarity)
    # Sweep the find axes where the LEGIBILITY SURFACE is least certain, not uniformly.
    # Measured after 29 uniform hunts: the surface's posterior sd along these axes (~0.38
    # log-units, a factor of 1.5 in time) dwarfed the effect it was trying to see (a 10-15%
    # swing), so uniform coverage was not identifying them -- while the salience axis
    # meanwhile ranks second of nine for PREFERENCE, so the question is worth answering.
    # Uncertainty sampling is the standard active choice for a GP regression and costs one
    # posterior evaluation over a grid.
    reading_time = rt_fit(history, polarity, fit.get("ls") if fit else None)
    if reading_time is not None and rng.random() > UNTARGETED_SHARE:
        predicted_variance = rt_at(reading_time, [theta for theta, _ in usable], polarity)[1]
        theta, theme = usable[int(np.argmax(predicted_variance))]
    else:
        theta, theme = usable[rng.randrange(len(usable))]
    if theme is None:
        index = rng.randrange(len(pool))
        theta, theme = np.array(pool[index][0]), pool[index][1]
    return {
        "mode": "search",
        "surface": "editor",
        "snippet_lines": PAGE_LINES,
        "kind": "task",
        "polarity": polarity,
        "theta_a": _rounded(theta),
        "theme_a": theme,
        "snippet": page_seed(n),
        "code_px": READING_PX["editor"],
    }


def trial_for(n, responses):
    """The nth trial, generated to maximize expected information about the utility.

    Duels compare a Thompson-sampled champion against the most informative challenger;
    comprehension probes ride the Thompson argmax; find hunts hold the champion's page and
    sweep the find axes. Each arm is built by its own function below.

    Depends only on `n` and the responses BEFORE trial n, so the recorder can rebuild it
    from the log instead of trusting what the page sent back.
    """
    if n in TRIAL_MEMO:
        return TRIAL_MEMO[n]
    history = responses[:n]
    n_duels = sum(1 for row in history if row.get("mode") == "duel")
    polarity, arm = schedule_mode(n, n_duels)
    rng = random.Random(n * 2654435761 % (2**31))
    numpy_rng = np.random.default_rng(n * 7919 + 13)
    pool = POOL[polarity]
    fit = fitted(history) if n_duels >= 4 else None

    if arm == "duel":
        trial = _duel_trial(n, polarity, fit, pool, rng, numpy_rng, n_duels)
    elif arm == "comprehension":
        trial = _comprehension_trial(n, history, polarity, fit, pool, rng, numpy_rng)
    else:
        trial = _search_trial(n, history, polarity, fit, pool, rng, numpy_rng)
    TRIAL_MEMO[n] = trial
    return trial
