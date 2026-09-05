"""The colour-discrimination trial generator: what the vision instrument shows next.

Four squares on one ground, three of one colour and one of another; the observer names the
odd one. Each probe trial is generated to maximise expected information about the observer
model in `theme.observer`, so every answer moves the whole threshold surface rather than one
pair's tally. This module is that generator, extracted from the vision notebook so the
same trials can be served by the web app -- which has the timing guarantees a notebook
cannot give -- and so the generator can be tested at all.

Everything here is a pure function of the trial number and the vision log before it,
exactly as `theme.schedule.trial_for` is of the aesthetics log: the recorder rebuilds the
trial from the log at record time rather than trusting what a page sent back.

Two generators share this code and differ in one declared way. The notebook's `v3` cycles
all three patch sizes, exhibit scale first. The app's `v4` retires the exhibit scale: 104 px
was declared converged on 2026-09-02 (68% intervals within +/-5% on every axis), and the one
number the programme still lacks is the small-field exponent, which only the 16 and 10 px
trials identify. Every row records which generator built it.
"""

import json
import random

import numpy as np

from . import observer as obs
from . import paths

# The palettes under evaluation, as literals. They used to be scraped from a matplotlib
# install and from the editor's extension directory at runtime, which made the trial
# generator depend on what happened to be installed on the machine taking the sitting. The
# palette label is provenance only -- the model reads the CAM16-UCS distance between two
# hexes and never the name -- so pinning the hexes costs nothing and makes a log replayable.
PALETTES = {
    "okabe-ito": ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    "cividis": ["#00224e", "#2a3f6d", "#575d6d", "#7d7c78", "#a59c74", "#d2c060", "#fee838"],
    "viridis": ["#440154", "#443983", "#31688e", "#21918c", "#35b779", "#90d743", "#fde725"],
    "batlow": ["#011959", "#144d62", "#3c6d56", "#828231", "#d29343", "#fdac9e", "#faccfa"],
    "tab10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    "Set1": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"],
    "horizon-day": ["#8a31b9", "#1d8991", "#f6661e", "#e84a72", "#da103f", "#989190"],
    "horizon-night": ["#a96ec9", "#24a2ad", "#e4a88a", "#e95378", "#d55170", "#4c4d53"],
}

#: Every within-palette pair: a candidate anchor, and the seed of every probe.
PAIRS = [
    (name, a, b) for name, hexes in PALETTES.items() for i, a in enumerate(hexes) for b in hexes[i + 1 :] if a != b
]

# The ground family spans the theme programme's candidate field in lightness. The first two
# labels are "day" and "night" because those are the labels the existing rows carry and the
# keys `observer.trial_features` falls back to for rows that predate `ground_hex`.
GROUND_LIST = [
    ("day", "#fdf0ed"),
    ("night", "#1c1e26"),
    ("selenized-light", "#fbf3db"),
    ("selenized-dark", "#103c48"),
    ("modus-light", "#ffffff"),
    ("modus-dark", "#000000"),
    ("github-dark", "#0d1117"),
]

#: Patch sizes: exhibit scale, and the glyph scale that decides the editor theme. Gap scales
#: with size (near-abutting, like adjacent glyphs) and rides with every response.
SIZES = (104, 16, 10)
GAPS = {104: 12, 16: 2, 10: 1}

#: The sizes the app serves: the exhibit scale is converged, the glyph scale is the question.
APP_SIZES = (16, 10)

#: Trials per (ground, size) block. Adaptation to a page is part of the measurement, so a
#: ground is held long enough to adapt to.
BLOCK = 16

#: Declared share of trials that are plain palette pairs rather than generated probes.
ANCHOR_SHARE = 0.05

#: How far a probe's base colour is moved off its palette colour, in CAM16-UCS units. What
#: makes a probe fresh: on a fixed base the information search kept converging on the same
#: offsets, and 489 probes produced only 332 distinct pairs.
BASE_JITTER = 2.5

#: Stamped on every response. A generator change is a stimulus change, and a stimulus change
#: that is not written down is indistinguishable later from a change in the observer.
NOTEBOOK_GENERATOR = "v3"
APP_GENERATOR = "v4"

#: The magnitudes swept per direction: coarse, then fine around the coarse winner. A
#: coarse-only grid measured ~28% of the achievable information lost when the threshold fell
#: between its steps.
COARSE_MAGNITUDES = np.geomspace(0.3, 90.0, 7)
FINE_SPAN = 2.5
FINE_STEPS = 8
MAX_MAGNITUDE = 110.0

#: Below this many responses the posterior is near flat and a top-k condensation picks an
#: arbitrary corner of the grid; a strided subset spans it evenly instead.
CONDENSE_AFTER = BLOCK
STRIDED_CELLS = 25_000


class Posterior:
    """The dense log-posterior over the observer grid, kept warm and cached beside the log.

    3.57M grid cells over every response costs minutes from cold (measured 227 s over 748
    responses on a busy machine), so the dense log-posterior is bootstrapped from a binary
    sidecar and updated per response (tens of milliseconds, equal to a from-scratch fit up
    to ~1e-4 log units of float accumulation). The sidecar is derived data and untracked.
    """

    def __init__(self, sidecar_dir=None):
        self.columns = obs.grid_columns()
        self.n_cells = len(self.columns[0])
        directory = sidecar_dir or paths.VISION_LOG.parent
        self.sidecar = directory / f"observer-logp-{obs.MODEL_VERSION}.npy"
        self.sidecar_meta = directory / f"observer-logp-{obs.MODEL_VERSION}.json"
        self._n = -1
        self._last_row = None
        self._logp = None

    def logp_for(self, responses):
        """The dense log-posterior over `responses`, incrementally where they extend what
        was folded in last time and from scratch otherwise.

        Keyed on content, not on length alone: a cache keyed by row count served a stale
        answer for different data of equal length more than once in this project, so the
        rows folded in so far must be the START of the rows asked for -- checked on the last
        one folded in, which is cheap and catches a different log of the same length.
        """
        extends = self._logp is not None and self._is_prefix_of(responses)
        if extends and self._n == len(responses):
            return self._logp
        if self._logp is None:
            self._logp = self._from_sidecar(responses)
        elif extends:
            self._logp = obs.add_loglik(self._logp, responses[self._n :])
        else:  # a different log, or one that shrank under an external edit: refit
            self._logp = obs.add_loglik(np.zeros(self.n_cells), responses, chunk=40_000)
        self._n = len(responses)
        self._last_row = responses[-1] if responses else None
        return self._logp

    def _is_prefix_of(self, responses):
        if self._n <= 0:
            return True
        return self._n <= len(responses) and responses[self._n - 1] == self._last_row

    def _from_sidecar(self, responses):
        stored_n = self._stored_length()
        logp, covered = None, 0
        if self.sidecar.exists() and 0 < stored_n <= len(responses):
            logp, covered = np.load(self.sidecar), stored_n
        if logp is None:
            logp = np.zeros(self.n_cells)
        if responses[covered:]:
            obs.add_loglik(logp, responses[covered:], chunk=40_000)
        # Never let a shorter log replace a longer log's sidecar. A test's scratch log, or a
        # truncated copy, would otherwise overwrite minutes of refit for the real log with
        # its own posterior -- and the next real sitting would silently start from zero.
        # The metadata is consulted whether or not the binary beside it exists: the binary
        # is gitignored and the metadata is not, so a worktree or a fresh clone has exactly
        # the metadata and no grid, and reading the length only when both were present let
        # an empty log's fit overwrite "n": 748 with "n": 0 (2026-09-05, in a worktree, by
        # the test suite). An empty log writes nothing: there is no work to cache.
        if responses and len(responses) >= stored_n:
            self.sidecar.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.sidecar, logp)
            self.sidecar_meta.write_text(json.dumps({"n": len(responses), "cells": self.n_cells}))
        return logp

    def _stored_length(self):
        """How many responses the sidecar on disk claims to cover: 0 when there is no
        metadata, or it describes a different grid."""
        if not self.sidecar_meta.exists():
            return 0
        meta = json.loads(self.sidecar_meta.read_text())
        if meta.get("cells") != self.n_cells:
            return 0
        return int(meta.get("n", 0))

    def condensed_for(self, responses):
        """(posterior weights, grid columns) over the cells that matter -- the generator's
        view."""
        logp = self.logp_for(responses)
        if len(responses) < CONDENSE_AFTER:
            index = np.arange(0, self.n_cells, max(1, self.n_cells // STRIDED_CELLS))
            weights = np.exp(logp[index] - logp[index].max())
            return weights / weights.sum(), [column[index] for column in self.columns]
        return obs.condense(logp, self.columns)

    def dense_for(self, responses):
        """The full-grid normalised posterior, for the interval readouts."""
        logp = self.logp_for(responses)
        weights = np.exp(logp - logp.max())
        return weights / weights.sum(), self.columns


#: The one posterior the instruments share in a process.
POSTERIOR = Posterior()


def block_for(n, sizes=SIZES):
    """(ground label, ground hex, patch size) for trial n.

    Sixteen-trial blocks cycle every ground, then rotate the patch size.
    """
    block = n // BLOCK
    label, hex_ = GROUND_LIST[block % len(GROUND_LIST)]
    return label, hex_, sizes[(block // len(GROUND_LIST)) % len(sizes)]


def odd_position_for(n):
    """Which of the four slots holds the odd square: a shuffled permutation per group of
    four trials, so balance is exact every four trials and position is decorrelated from
    position-within-block. Deterministic in n."""
    permutation = [0, 1, 2, 3]
    random.Random(0x5EED + n // 4).shuffle(permutation)
    return permutation[n % 4]


def _binary_entropy(q):
    return -(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12))


def _probe_directions(confusion_angle_deg):
    """The CAM16-UCS directions a probe is swept along: the axes, both blue-yellow
    diagonals, and the current estimate of the confusion axis."""
    diagonal = 1 / np.sqrt(2)
    rad = np.radians(confusion_angle_deg)
    return [
        np.array(v)
        for v in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (0, diagonal, diagonal),
            (0, -diagonal, diagonal),
            (0, np.cos(rad), np.sin(rad)),
            (0, -np.cos(rad), np.sin(rad)),
        ]
    ]


def _best_along(posterior, columns, base_ucs, base_hex, direction, magnitudes, ground_lightness, size):
    """(information gain, magnitude, hex) of the most informative stimulus on one direction."""
    candidates = np.array([base_ucs + direction * m for m in magnitudes])
    hexes = obs.ucs_to_hex(candidates)
    differences = (base_ucs[None, :] - obs.hex_to_ucs(hexes)).astype(np.float32)
    lightness = np.repeat(np.array([ground_lightness], dtype=np.float32), len(magnitudes))
    sizes = np.repeat(np.array([float(size)], dtype=np.float32), len(magnitudes))
    p_correct = obs.p_correct_cells(columns, differences, lightness, sizes)
    marginal = posterior @ p_correct
    gain = _binary_entropy(marginal) - posterior @ _binary_entropy(p_correct)
    for i, hex_ in enumerate(hexes):
        if hex_ == base_hex:  # a stimulus identical to its base carries no question
            gain[i] = -1.0
    best = int(np.argmax(gain))
    return float(gain[best]), float(magnitudes[best]), hexes[best]


def _most_informative_odd(rng, posterior, columns, seed_hex, ground_hex, size):
    """(base hex, odd hex): a jittered base and the odd colour that teaches the model most."""
    seed_ucs = obs.hex_to_ucs(seed_hex)[0]
    jitter = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
    base_ucs = seed_ucs + BASE_JITTER * jitter / max(float(np.linalg.norm(jitter)), 1e-9)
    base_hex = obs.ucs_to_hex(base_ucs[None, :])[0]
    base_ucs = obs.hex_to_ucs(base_hex)[0]
    ground_lightness = obs.hex_to_ucs(ground_hex)[0, 0] / 100.0
    confusion_angle = float((posterior * columns[0]).sum())
    best_gain, best_hex = -1.0, None
    for direction in _probe_directions(confusion_angle):
        gain, magnitude, hex_ = _best_along(
            posterior, columns, base_ucs, base_hex, direction, COARSE_MAGNITUDES, ground_lightness, size
        )
        if gain > best_gain:
            best_gain, best_hex = gain, hex_
        fine = np.geomspace(magnitude / FINE_SPAN, min(magnitude * FINE_SPAN, MAX_MAGNITUDE), FINE_STEPS)
        gain, _magnitude, hex_ = _best_along(
            posterior, columns, base_ucs, base_hex, direction, fine, ground_lightness, size
        )
        if gain > best_gain:
            best_gain, best_hex = gain, hex_
    return base_hex, best_hex


TRIAL_MEMO = {}
TRIAL_MEMO_KEEP = 8


def _history_key(n, history, generator, sizes):
    """Identify the history by its length and the CONTENT of its last row, not its
    timestamp: rows land seconds apart in a sitting but within the same second in a test
    suite, and two logs of equal length whose last rows differ are different histories. One
    row is cheap to serialise; the whole log is not, and does not need to be."""
    last = json.dumps(history[-1], sort_keys=True) if history else None
    return (n, len(history), last, generator, tuple(sizes))


def trial_for(n, responses, sizes=SIZES, generator=NOTEBOOK_GENERATOR, posterior=None):
    """The nth discrimination trial, generated to maximise expected information about the
    observer. Pure in (n, responses[:n], sizes, generator); memoised on all four.

    An anchor trial (a declared 5%) shows a plain palette pair, easy by construction, as a
    check against model misspecification. Every other trial is a probe: a jittered palette
    colour against the odd colour whose answer teaches the model most.
    """
    history = responses[:n]
    key = _history_key(n, history, generator, sizes)
    if key in TRIAL_MEMO:
        return TRIAL_MEMO[key]
    rng = random.Random(n * 2654435761 % (2**31))
    ground_label, ground_hex, size = block_for(n, sizes)
    palette, first, second = rng.choice(PAIRS)
    if rng.random() < ANCHOR_SHARE:
        if rng.random() < 0.5:
            first, second = second, first
        base, odd, kind = first, second, "anchor"
    else:
        weights, columns = (posterior or POSTERIOR).condensed_for(history)
        base, odd = _most_informative_odd(rng, weights, columns, first, ground_hex, size)
        kind = "probe"
    trial = {
        "kind": kind,
        "palette": palette,
        "base": base,
        "base_source": first,
        "odd_color": odd,
        "ground": ground_label,
        "ground_hex": ground_hex,
        "size_px": size,
        "gap_px": GAPS[size],
        # Recorded because it is a stimulus parameter, not because the model fits it.
        "layout": "row",
        "odd_position": odd_position_for(n),
        "generator": generator,
    }
    TRIAL_MEMO[key] = trial
    while len(TRIAL_MEMO) > TRIAL_MEMO_KEEP:
        TRIAL_MEMO.pop(next(iter(TRIAL_MEMO)))
    return trial


def build_entry(n, trial, choice, ts, timing=None):
    """The record for one answered discrimination trial.

    Everything the model reads plus everything needed to re-analyse the trial under a model
    that does not exist yet; this log has already been re-read under two observer models.
    `timing`, when the app recorded the answer, adds the reaction-time fields the notebook
    surface cannot provide -- the whole reason the arm moved into the app.
    """
    entry = {
        "ts": ts,
        "n": n,
        **{key: trial[key] for key in trial},
        "choice": choice,
        "correct": choice == trial["odd_position"],
    }
    if timing is not None:
        entry.update(timing)
    return entry
