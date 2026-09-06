"""What the log says the best theme is, per polarity, as one object.

The analysis notebook used to compute this inside a three-hundred-line cell: breed the
candidates, drop the ones the legibility surface calls slower, find the champion, sample
P(best), pick the shelf, run the factor tests, write the champion file. None of it could
be tested, none of it could run without a browser, and one bug hid there for a week -- the
"leader reads in about N s" sentence reported the reading time of the first candidate in
the pool, because a variable called `lead` held the first KEPT index rather than the
champion's.

So the computation lives here, as data, and the notebook only words it. Everything that
reads the verdict -- the notebook, `theme.publish`, the rotation that decides which shelf
member to live in next -- gets the same numbers from the same place.

Constraint first, preference second, the order the whole instrument uses: candidates the
legibility surface says are credibly slower than the fastest are removed BEFORE the
preference posterior is allowed to pick a winner, exactly as the contrast floors remove
illegible candidates before anything is shown.
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from . import paths
from .diagnostics import axis_consensus, best_set, factor_effect, progress_report, spread_out
from .kernel import N_AXES, scale_thetas
from .legibility import rt_at, rt_fit, rt_penalty
from .model import candidates_with_strata
from .observer import MODEL_VERSION as OBSERVER_MODEL
from .preference import duel_rows, fitted, mean_utility_at, realized_space
from .signals import signals_for
from .space import DE_MIN, READING_SIZE_PX, VISION_N, separation_floor
from .surfaces import derived_surfaces

#: The candidate set the verdict is read over is bred from THIS seed and trial number, so
#: two readings of one log see the same candidates and differ only in what the log says.
CANDIDATE_SEED = 4242

#: Fewer duels than this and there is no fit worth reading.
MIN_DUELS = 5

#: Below this many surviving candidates the legibility filter is not applied: dropping
#: most of the field on a thin reaction-time log would let noise choose the champion.
MIN_SURVIVORS = 32

#: How many shelf members the readout shows, chosen for spread rather than by rank.
SHOWN_MEMBERS = 4

#: The stimulus factors every verdict tests for an interaction with the preferred theme.
#: Two per polarity, four per reading -- the multiplicity the report has to state.
FACTORS = ("surface", "code_px")


@dataclass(frozen=True)
class Legibility:
    """What the timed arms did to this verdict."""

    n_timed: int
    n_candidates: int
    n_excluded: int
    champion_seconds: float
    #: Champion minus fastest, in log time, with its standard deviation. A DIFFERENCE with
    #: an interval, never two point estimates side by side: the posterior sd on either
    #: page is around 0.3 in log time, and the minimum over several hundred noisy
    #: predictions is an extreme order statistic, biased low.
    gap_log_time: float
    gap_sd: float

    @property
    def champion_credibly_slower(self) -> bool:
        return self.gap_log_time - 1.96 * self.gap_sd > 0

    @property
    def gap_interval(self) -> tuple[float, float]:
        return self.gap_log_time - 1.96 * self.gap_sd, self.gap_log_time + 1.96 * self.gap_sd


@dataclass(frozen=True)
class Verdict:
    """The best theme for one polarity, and how sure the log is."""

    polarity: str
    n_duels: int
    thetas: list = field(repr=False)
    themes: list = field(repr=False)
    champion: int
    mean_utility: np.ndarray = field(repr=False)
    best: dict = field(repr=False)
    #: Shelf members to show, leader first, the rest chosen for maximal difference.
    shown: list
    #: Each shown member's GROUP probability of being best -- the number the verdict quotes.
    #: A single point's own p_best is the mass of one point in a continuum and always far
    #: smaller, which once made the leader's card read 2% under a headline of 24%.
    shown_probability: dict
    legibility: Legibility | None
    progress: dict | None
    factors: dict
    consensus: list
    #: Where each candidate came from: "pool" (the standing grid), "fresh" (this reading's
    #: Sobol immigrants) or "bred" (a child of the elites). One per entry of `thetas`.
    strata: list = field(repr=False)
    #: How far apart the standing grid's nearest neighbours sit in the model's own metric,
    #: in correlation lengths -- the one comparison that decides whether the grid wants
    #: widening (method reef: judge a pool's density against the length scales, not against
    #: how many a floor removed). `resolves` is a median neighbour within one length.
    grid: dict
    #: The fit the verdict was read from, kept so the sweep and the provenance can reach it.
    fit: dict = field(repr=False)

    @property
    def verdict(self) -> str:
        return self.best["verdict"]

    @property
    def lead(self) -> float:
        return float(self.best["lead"])

    @property
    def credible(self) -> list:
        return list(self.best["credible"])

    @property
    def champion_theta(self):
        return self.thetas[self.champion]

    @property
    def champion_theme(self) -> dict:
        return self.themes[self.champion]

    @property
    def champion_stratum(self) -> str:
        return self.strata[self.champion]

    @property
    def shelf_strata(self) -> dict:
        """How many of the shown shelf members came from each stratum, leader included.

        All bred is the designed steady state (children refine the elites); a leader from
        the pool or the immigrants says the model's refinements are not yet beating uniform
        coverage. Whether the grid is wide enough is not read here but from `grid`.
        """
        counts = dict.fromkeys(("pool", "fresh", "bred"), 0)
        for i in self.shown:
            counts[self.strata[i]] += 1
        return counts

    @property
    def beats_random(self) -> float:
        """How often the champion would beat a random feasible theme in a duel not yet run."""
        return float(np.mean(1.0 / (1.0 + np.exp(-(self.mean_utility[self.champion] - self.mean_utility)))))

    def axis_sweep(self, axes, low=0.15, high=0.85):
        """Posterior-mean change from the champion when one axis is pushed to its walls.

        Negative means the champion's own setting is better. Returns one row per axis.
        """
        rows = []
        for axis in range(N_AXES):
            at_low = np.array(self.champion_theta, dtype=float)
            at_high = at_low.copy()
            at_low[axis], at_high[axis] = low, high
            change = mean_utility_at(self.fit, [at_low, at_high], self.polarity) - self.mean_utility[self.champion]
            rows.append(
                {
                    "axis": axes[axis],
                    f"low ({low})": round(float(change[0]), 2),
                    f"high ({high})": round(float(change[1]), 2),
                }
            )
        return rows


#: A standing grid point whose nearest neighbour sits within this many correlation lengths
#: has the surface between them interpolated rather than guessed: at one length the Matern
#: 5/2 kernel still correlates the two at about 0.5. Past it the model sees structure finer
#: than a uniform grid of any affordable size carries -- in nine dimensions doubling the
#: pool tightens neighbours by only 2^(1/9), 8 to 9% measured -- so the answer is never
#: "widen"; it is that refinement rests on the bred children, which is what they are for.
GRID_RESOLVES_WITHIN = 1.0


def grid_resolution(fit, polarity):
    """{neighbour_lengths, resolves}: does the standing grid still resolve the surface the
    model can see?

    The median distance from each pool theta to its nearest pool neighbour, measured in the
    kernel's own coordinates (each axis divided by its fitted ARD length scale), so one
    unit is one correlation length whatever the axes' scales. A per-axis gap against the
    finest length scale was the first attempt and it misjudged: the pool's 0.16 per axis
    against 0.30 read as too coarse while the same points sit at 0.95 (day) and 1.01 (night)
    correlation lengths once every axis is scaled by its own length -- at the edge, which
    is where a uniform grid in nine dimensions lives whatever its size (see
    GRID_RESOLVES_WITHIN).
    """
    thetas = [theta for theta, _theme in realized_space().POOL[polarity]]
    if len(thetas) < 2:
        return {"neighbour_lengths": float("nan"), "resolves": False}
    scaled = scale_thetas(thetas, fit.get("ls"))
    gaps = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    neighbour = float(np.median(gaps.min(axis=1)))
    return {"neighbour_lengths": neighbour, "resolves": bool(neighbour <= GRID_RESOLVES_WITHIN)}


def _bred_candidates(fit, polarity):
    bred, _n_standing, strata = candidates_with_strata(fit, polarity, np.random.default_rng(CANDIDATE_SEED), n_trial=0)
    return [theta for theta, _theme in bred], [theme for _theta, theme in bred], strata


def _apply_legibility(responses, polarity, fit, thetas, themes, strata):
    """(kept thetas, kept themes, kept strata, the exclusion mask, the surface) after the
    timed arms bind.

    A thin or noisy reaction-time log excludes nothing, by construction of rt_penalty; and a
    log that would exclude nearly everything is not trusted to choose either.
    """
    surface = rt_fit(responses, polarity, fit.get("ls"))
    if surface is None:
        return thetas, themes, strata, None, None
    excluded, _seconds = rt_penalty(surface, thetas, polarity)
    kept = [i for i in range(len(thetas)) if not excluded[i]]
    if len(kept) < MIN_SURVIVORS:
        return thetas, themes, strata, None, surface
    return [thetas[i] for i in kept], [themes[i] for i in kept], [strata[i] for i in kept], excluded, surface


def _legibility_note(surface, excluded, thetas, polarity, champion):
    """The champion's reading time against the fastest page, as a difference with an
    interval."""
    if surface is None or excluded is None:
        return None
    mean_log_time, variance = rt_at(surface, thetas, polarity)
    fastest = int(np.argmin(mean_log_time))
    gap = float(mean_log_time[champion] - mean_log_time[fastest])
    return Legibility(
        n_timed=int(surface["n"]),
        n_candidates=len(excluded),
        n_excluded=int(excluded.sum()),
        champion_seconds=float(np.exp(mean_log_time[champion])),
        gap_log_time=gap,
        gap_sd=float(np.sqrt(variance[champion] + variance[fastest])),
    )


def verdict_for(responses, polarity, fit=None):
    """The verdict for one polarity, or None while the log is too thin to read.

    `responses` may be the union of several logs -- the instrument's own and the lived
    duels -- because everything here reads duel rows and nothing here schedules trials.
    """
    duels = duel_rows(responses)
    if len(duels) < MIN_DUELS:
        return None
    fit = fit or fitted(responses)
    if fit is None:
        return None
    thetas, themes, strata = _bred_candidates(fit, polarity)
    thetas, themes, strata, excluded, surface = _apply_legibility(responses, polarity, fit, thetas, themes, strata)
    mean_utility = mean_utility_at(fit, thetas, polarity)
    champion = int(np.argmax(mean_utility))
    best = best_set(fit, polarity, thetas, seed=17)
    probability_of = dict(zip(best["credible"], best["credible_p"], strict=True))
    shown = spread_out(thetas, best["credible"], SHOWN_MEMBERS, fit.get("ls"))
    shown = sorted(shown, key=lambda i: -probability_of.get(i, 0.0))
    return Verdict(
        polarity=polarity,
        n_duels=sum(1 for row in duels if row["polarity"] == polarity),
        thetas=thetas,
        themes=themes,
        champion=champion,
        mean_utility=mean_utility,
        best=best,
        shown=shown,
        shown_probability={i: float(probability_of.get(i, 0.0)) for i in shown},
        legibility=_legibility_note(surface, excluded, thetas, polarity, champion),
        progress=progress_report(responses, polarity, thetas),
        factors={key: factor_effect(responses, polarity, key) for key in FACTORS},
        consensus=axis_consensus(best, thetas),
        strata=strata,
        grid=grid_resolution(fit, polarity),
        fit=fit,
    )


def _code_revision():
    """The commit the verdict was computed by, so a published palette can be traced to the
    code that produced it. None outside a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return None


def provenance(verdict):
    """Everything a consumer needs to know WHICH measurement a palette came from.

    Tightening a threshold silently re-bases every past duel, and a palette applied last
    week was chosen under last week's floors. Pixel size already rides along with every
    response; this is the same discipline for the fit and the observer behind it.
    """
    floor, regime = separation_floor(verdict.polarity, READING_SIZE_PX)
    fit = verdict.fit
    return {
        "published": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": _code_revision(),
        "n_duels": verdict.n_duels,
        "fit": str(fit["fingerprint"][2]) if "fingerprint" in fit else None,
        "rt_exponent": fit.get("rt_p"),
        "observer": {
            "model": OBSERVER_MODEL,
            "n_trials": VISION_N,
            "de_min": round(float(DE_MIN[verdict.polarity]), 3),
            "separation_floor": round(float(floor), 3),
            "regime": regime,
        },
    }


def palette_of(verdict):
    """One polarity's published payload: the palette, the verdict, and where it came from.

    `theta` is here so a lived duel between two applied themes can be recorded as a duel
    the model already understands; `page` and `border` are the two surfaces the applied
    elevation system derives from the ground, and `signals` are the convention-bound
    colours (error red, git green, the ANSI set) walked to this ground's floors. All are
    computed here, once, rather than in an applier that has no colour engine: the applier
    maps names onto keys and never does colour arithmetic.
    """
    theme = verdict.champion_theme
    return {
        **{role: theme[role] for role in ("ground", "keyword", "function", "string", "ink", "comment", "punct")},
        "find_fill": theme["find_fill"],
        **derived_surfaces(theme["ground"], verdict.polarity),
        "signals": signals_for(
            theme["ground"], (theme["keyword"], theme["function"], theme["string"]), verdict.polarity
        ),
        "theta": [round(float(value), 6) for value in verdict.champion_theta],
        "p_best": round(verdict.lead, 4),
        "verdict": verdict.verdict,
        "n_duels": verdict.n_duels,
        "provenance": provenance(verdict),
    }


def publish(responses, path=paths.CHAMPION):
    """Write both polarities' palettes for the applier, keeping whatever the file already
    holds for a polarity this log cannot yet decide. Returns what was written."""
    published = {}
    if path.exists():
        try:
            published = json.loads(path.read_text())
        except json.JSONDecodeError:
            published = {}
    for polarity in ("day", "night"):
        verdict = verdict_for(responses, polarity)
        if verdict is not None:
            published[polarity] = palette_of(verdict)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(published, indent=2, sort_keys=True) + "\n")
    return published
