"""The theme space, and how a point in it becomes a theme.

Nine axes in, a full palette of hexes out -- or a refusal, when no palette at that point
clears the floors. Two neighbours were split out of this module because they change for
different reasons than the space does, and both are re-exported below because callers
outside the colour layer (theme/model.py, theme/schedule.py, the analysis notebook) had
been importing them from here:

  thresholds.py   the observer-derived perceptual floors: DE_MIN and friends
  harmony.py      Ou & Luo (2006) two-colour harmony, transcribed from the paper

What stayed is what genuinely belongs to the space: the axes, the anchor colours, the
realization, the feasible pool, and the prior standardized over that pool. raw_prior in
particular reads theta by axis index and so cannot leave without taking the axis layout
with it, which is why the harmony model moved and the prior did not.
"""

import math
from itertools import combinations
from typing import NamedTuple

import numpy as np

from .color import (
    apca_lc,
    composite,
    composite_many,
    hex_to_rgb,
    rgb_to_hex,
    rgb_to_ucs,
    solve_j,
    ucs_to_rgb,
    wcag,
)
from .harmony import lab, ou_luo_pair
from .thresholds import DE_MIN, THRESH_DETAIL, VISION_FIT, VISION_LOG, VISION_N  # noqa: F401

# Theme space and its realization. Nine axes, each in [0, 1]; polarity (light page /
# dark page) is a block factor, not an axis — trials alternate in blocks and the model
# carries it as a tenth, binary coordinate so learning transfers between the two
# without conflating them.
AXES = [
    "ground lightness",
    "ground warmth",
    "accent hue rotation",
    "accent chroma",
    "body contrast",
    "hue spread",
    "comment recede",
    "find hue",
    "find salience",
]

# Horizon's own token colors anchor the accent hues — evolve, don't repaint. Night
# anchors carry alpha in the theme file and are composited onto their page before any
# appearance math (the rule that caught the 30%-alpha comments). Literals are ONE
# family on purpose: Horizon's day string #F6661E and number #F77D26 sit ~3 dE apart
# in CAM16-UCS — inside 2x the measured day threshold — so a string/number split would
# search a distinction the eye cannot resolve.
ANCHORS = {
    "day": {
        "keyword": "#8A31B9",
        "function": "#1D8991",
        "string": "#F6661E",
        "ground": "#FDF0ED",
    },
    "night": {
        "keyword": composite("#B877DB", 0.902, "#1C1E26"),
        "function": composite("#25B0BC", 0.902, "#1C1E26"),
        "string": composite("#FAB795", 0.902, "#1C1E26"),
        "ground": "#1C1E26",
    },
}
ROLE_ORDER = ("keyword", "function", "string")


def anchor_polar(polarity):
    """The anchor hues (degrees) and chromas of one polarity's accent roles."""
    ucs = rgb_to_ucs(hex_to_rgb([ANCHORS[polarity][role] for role in ROLE_ORDER]))
    chroma = np.linalg.norm(ucs[:, 1:], axis=1)
    hue = np.degrees(np.arctan2(ucs[:, 2], ucs[:, 1])) % 360
    return hue, chroma


ANCHOR_HM = {p: anchor_polar(p) for p in ("day", "night")}

# Realization and prior are pure functions of (theta, polarity); the caches make the
# per-trial local-refinement candidates (and every posterior call over the pool) pay
# for their appearance math exactly once per kernel.
REALIZE_CACHE = {}
PRIOR_CACHE = {}


def theta_key(theta, polarity):
    """The cache key for one candidate. Thetas within 1e-6 on every axis are one point:
    that is far finer than 8-bit quantization can express, so collapsing them cannot
    change a rendered theme, and it lets a re-proposed candidate hit the cache."""
    return (tuple(round(float(value), 6) for value in theta), polarity)


# NOTE (marimo name mangling): this code also exists as marimo cells in the analysis
# notebook, and there a cell-local (underscore) name referenced from inside an exported
# function resolves only if it is defined ABOVE that function in the cell — a later
# definition stays unmangled in the function body and NameErrors at call time under
# `marimo run`, invisibly to script execution. Nothing in this module is cell-local any
# more, but helpers still precede their callers here so the two stay transposable.
# The roles solve_j walks to the contrast bar, in the order their rows appear.
WALKED_ROLES = ("keyword", "function", "string", "ink", "comment", "punct")
#: APCA floors per walked role. Body tokens carry meaning; comments are context.
LC_FLOORS = np.array([60.0, 60.0, 60.0, 60.0, 45.0, 45.0])

#: How many times _walk_to_both_bars will raise a role's WCAG target and re-bisect to
#: get it over its APCA floor before giving up and letting the floors refuse the theme.
WALK_ATTEMPTS = 4


def _grounds(thetas, night):
    """The page colour for each theta: lightness within the polarity's family, warmth as a
    signed warm/cool axis."""
    lightness = 8.0 + 14.0 * thetas[:, 0] if night else 86.0 + 9.0 * thetas[:, 0]
    warmth = 2.0 * thetas[:, 1] - 1.0
    hue = np.where(warmth >= 0, math.radians(74.0), math.radians(256.0))
    chroma = np.abs(warmth) * 6.0
    ucs = np.column_stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)])
    return lightness, hue, ucs_to_rgb(ucs)


def _role_chromaticities(thetas, polarity, ground_hue):
    """The a', b' each walked role starts from, and the contrast ratio it must reach.

    Accents rotate and spread Horizon's own hues and scale their chroma; the neutral family
    (ink, comment, punctuation) takes the GROUND's hue at a whisper of chroma, so page and
    text agree in temperature. Lightness is not set here -- that is what solve_j walks.
    """
    anchor_hues, anchor_chroma = ANCHOR_HM[polarity]
    mean_hue = (
        math.degrees(math.atan2(np.sin(np.radians(anchor_hues)).mean(), np.cos(np.radians(anchor_hues)).mean())) % 360
    )
    rotation = (thetas[:, 2] - 0.5) * 120.0
    spread = 0.4 + 1.2 * thetas[:, 5]
    offsets = (anchor_hues - mean_hue + 180.0) % 360.0 - 180.0
    hues = (mean_hue + spread[:, None] * offsets[None, :] + rotation[:, None]) % 360.0
    chroma = (0.6 + 0.8 * thetas[:, 3])[:, None] * anchor_chroma[None, :]
    accent_ab = np.stack([chroma * np.cos(np.radians(hues)), chroma * np.sin(np.radians(hues))], axis=-1)

    neutral_direction = np.stack([np.cos(ground_hue), np.sin(ground_hue)], axis=-1)
    neutral_ab = neutral_direction[:, None, :] * np.array([1.5, 2.0, 1.5])[None, :, None]

    # 12:1 caps the body bar: the theme's native 18:1 is near-maximum contrast, where light
    # bleeds into the glyph edges and fine strokes appear to vibrate.
    body = 4.5 + 4.5 * thetas[:, 4]
    comment = np.maximum(4.5, body * (0.55 + 0.35 * thetas[:, 6]))
    punctuation = np.maximum(4.5, body * 0.75)
    ratios = np.minimum(np.column_stack([body, body, body, np.maximum(body, 5.5), comment, punctuation]), 12.0)
    return np.concatenate([accent_ab, neutral_ab], axis=1), ratios


def _walk_to_both_bars(role_ab, ground_rgb, ratios, night):
    """Walk every role's lightness until WCAG and APCA both hold.

    WCAG sets the first target; APCA is the stricter master on dark grounds (4.5:1 there is
    only Lc ~54), so rows short of their Lc floor walk further from the page until both
    bars hold. The 1.03 margin keeps bisection from converging a hair under.

    Batched across themes, which is the whole reason this is a separate function. The cost
    of colour-science is per CALL, not per colour -- measured, a call converting one colour
    costs 312 us and a call converting sixty-four costs 325 us -- so walking six roles for
    one theme at a time spent 3.8 s of a 4 s trial on Python-level validation inside the
    library. Iterating over all themes at once makes that one call per bisection step.

    Four attempts, and rows still short after the fourth are returned as they stand:
    _assemble's Lc floor is what then refuses the theme, so a role that cannot reach its
    bar surfaces as a refusal rather than as a quietly illegible colour. Measured over
    batches of 1 to 512 at both polarities, a batch uses 1 to 4 attempts; the stopping
    condition is shared across the batch, so a theme in company can be re-solved more
    times than it would be alone. That is only harmless because the extra solves use an
    unchanged target -- tested directly, since every batched caller here depends on it.
    """
    rows = role_ab.reshape(-1, 2)
    grounds = np.repeat(ground_rgb, role_ab.shape[1], axis=0)
    target = ratios.reshape(-1) * 1.03
    floors = np.tile(LC_FLOORS, len(role_ab))
    for attempt in range(WALK_ATTEMPTS):
        _lightness, rgb = solve_j(rows, grounds, target, lighter=night)
        short = np.abs(apca_lc(rgb, grounds)) < floors
        if not short.any() or attempt == WALK_ATTEMPTS - 1:
            break
        target = np.where(short, np.minimum(target * 1.18, 14.0), target)
    return rgb.reshape(*role_ab.shape[:2], 3), grounds.reshape(*role_ab.shape[:2], 3)


def _find_fills(thetas, ground_lightness, night):
    """The find highlight: a fill near the page's lightness whose loudness is the salience
    axis. Emitted with alpha, because that is how VSCode layers it; every constraint and
    every rendered pixel uses the composited result."""
    salience = thetas[:, 8]
    hue = 2.0 * math.pi * thetas[:, 7]
    chroma = 8.0 + 26.0 * salience
    lightness = ground_lightness + (4.0 + 14.0 * salience) * (1 if night else -1)
    return ucs_to_rgb(np.column_stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)]))


#: Where each colour sits in a theme's separation set, which is the order
#: `_quantize_and_measure` builds that set in. The pairwise separation floors are stated
#: between these. The first five are WALKED_ROLES[:5] in order, which is why the same
#: index names a colour in both -- punctuation is walked but has no separation owed.
KEYWORD, FUNCTION, STRING, INK, COMMENT, GROUND, FIND_CURRENT, FIND_OTHER = range(8)

#: Every pair of these owes 2x the discrimination threshold: the three roles that carry
#: meaning by hue, plus the body ink they have to stay clear of. Two of them confusable
#: is a page where the syntax highlighting is decoration. Doubled because discrimination
#: collapses toward glyph scale and the thresholds were measured on 104-px patches.
MEANING_ROLES = (KEYWORD, FUNCTION, STRING, INK)

#: How many of WALKED_ROLES are body text; the rest are comment and punctuation. This is
#: the span body_ratio is reported over.
BODY_ROLE_COUNT = 4

#: The WCAG ratio every role owes the page.
WCAG_FLOOR = 4.5

#: Slack on the WCAG floor, absorbing the last bits of a float the invariant test
#: recomputes independently from the same hexes. Deliberately NOT applied to the APCA or
#: separation floors, which are checked exactly, so that every check here is at least as
#: strict as the test asserting it rather than the other way round.
FLOOR_SLACK = 1e-6

#: What ink and a string owe a find fill they are sitting on. Under the 4.5:1 the page
#: itself owes, deliberately: a highlight is a transient state the eye is already pointed
#: at, and holding it to body-text contrast would forbid every fill loud enough to find.
#: What must not happen is a highlight hiding the token it was drawn to reveal.
FILL_FLOOR_INK, FILL_FLOOR_STRING = 4.0, 3.5

#: The two alphas VSCode layers a search hit at: the current match, then every other one.
FILL_ALPHA_CURRENT, FILL_ALPHA_OTHER = 0.85, 0.45


class _Measured(NamedTuple):
    """A batch of themes quantized to hex, with every floor already measured on it.

    Per-theme fields are indexed by position in the batch. Nothing downstream of this
    touches a continuous colour, which is the point.
    """

    role_hex: list  # per theme, one hex per WALKED_ROLES
    ground_hex: list
    fill_hex: list  # the uncomposited fill, which is what the theme file carries
    current_hex: list  # the fill at FILL_ALPHA_CURRENT over the page
    other_hex: list
    lc: np.ndarray  # (themes, roles) signed APCA against the page
    contrast: np.ndarray  # (themes, roles) WCAG against the page
    ink_on_fills: np.ndarray  # (themes, 2) WCAG of ink over [current, other]
    string_on_fills: np.ndarray
    separations: np.ndarray  # (themes, 8, 3) CAM16-UCS of the separated colours


def _quantize_and_measure(role_rgb, ground_rgb, fill_rgb):
    """Quantize a whole batch to hex, then measure every floor on the quantized values.

    Measured on the QUANTIZED colours -- the 8-bit values the page will actually write --
    never on the unrounded ones the bisection produced. Rounding to hex moves a colour by
    up to half a step, which is enough to cross a floor: a property test found a theme
    whose function and string tokens passed at Lc 60.27 and 60.06 and rendered at 59.89
    and 59.83. Both were shown. The floors are a promise about pixels, so they have to be
    measured on pixels. The fills are composited before anything is measured on them for
    the same reason: an alpha emitted into a theme file is contrast nobody has checked.

    Every measurement here is one array call for the whole batch instead of one per theme,
    which is what leaves the per-theme step below as pure packaging.
    """
    themes, roles = len(ground_rgb), len(WALKED_ROLES)
    role_hex = rgb_to_hex(role_rgb.reshape(-1, 3))
    ground_hex = rgb_to_hex(ground_rgb)
    fill_hex = rgb_to_hex(fill_rgb)
    current_hex = composite_many(fill_hex, FILL_ALPHA_CURRENT, ground_hex)
    other_hex = composite_many(fill_hex, FILL_ALPHA_OTHER, ground_hex)

    rendered_roles = hex_to_rgb(role_hex)
    rendered_grounds = np.repeat(hex_to_rgb(ground_hex), roles, axis=0)
    lc = apca_lc(rendered_roles, rendered_grounds).reshape(themes, roles)
    contrast = wcag(rendered_roles, rendered_grounds).reshape(themes, roles)

    # [current, other] per theme, interleaved, so a role repeated twice lines up with it.
    rendered_fills = hex_to_rgb([h for pair in zip(current_hex, other_hex, strict=True) for h in pair])
    on_fills = {
        role: wcag(np.repeat(rendered_roles[role::roles], 2, axis=0), rendered_fills).reshape(themes, 2)
        for role in (INK, STRING)
    }

    separations = rgb_to_ucs(
        hex_to_rgb(
            [
                colour_hex
                for i in range(themes)
                for colour_hex in (
                    *role_hex[i * roles : i * roles + COMMENT + 1],
                    ground_hex[i],
                    current_hex[i],
                    other_hex[i],
                )
            ]
        )
    ).reshape(themes, 8, 3)

    return _Measured(
        role_hex=[role_hex[i * roles : (i + 1) * roles] for i in range(themes)],
        ground_hex=ground_hex,
        fill_hex=fill_hex,
        current_hex=current_hex,
        other_hex=other_hex,
        lc=lc,
        contrast=contrast,
        ink_on_fills=on_fills[INK],
        string_on_fills=on_fills[STRING],
        separations=separations,
    )


def _contrast_floors_hold(lc, contrast):
    """Does one theme's text clear both contrast bars against its own page?

    Stated as "every role is at or above its floor" rather than "no role is below it".
    Those read the same for real numbers and differently for a NaN, which is what an
    out-of-range channel produces: this form refuses such a theme instead of passing it.
    """
    return bool((contrast >= WCAG_FLOOR - FLOOR_SLACK).all() and (np.abs(lc) >= LC_FLOORS).all())


def _separations_hold(gap, threshold):
    """Does one theme keep every pair of its colours as far apart as that pair is owed?

    The margin is not uniform and should not be. Comment and ink are MEANT to sit inside a
    full discrimination step: both are neutral text and a comment is a deliberate step
    quieter than body ink, so demanding more there would fight the figure-versus-ground
    rule the palette is built on. The italic carries the rest.
    """
    if any(gap(first, second) < 2 * threshold for first, second in combinations(MEANING_ROLES, 2)):
        return False
    if gap(COMMENT, INK) < threshold:
        return False
    # The highlight has to be findable against the page and separable from its siblings,
    # which is the whole point of the salience axis.
    return gap(FIND_CURRENT, GROUND) >= 1.5 * threshold and gap(FIND_CURRENT, FIND_OTHER) >= threshold


def _fills_readable(ink_on_fills, string_on_fills):
    """Does text survive sitting on either find fill? Same NaN-refusing form as above."""
    return bool((ink_on_fills >= FILL_FLOOR_INK).all() and (string_on_fills >= FILL_FLOOR_STRING).all())


def _assemble(measured, i, polarity):
    """Theme `i` of a measured batch, or None if it breaks a floor.

    Guard clauses and then packaging: the floors are hard constraints, never objectives,
    so there is nothing to trade off here -- only three ways to say no, and one dict.
    """
    if not _contrast_floors_hold(measured.lc[i], measured.contrast[i]):
        return None
    separations = measured.separations[i]

    def gap(first, second):
        return float(np.linalg.norm(separations[first] - separations[second]))

    if not _separations_hold(gap, DE_MIN[polarity]):
        return None
    if not _fills_readable(measured.ink_on_fills[i], measured.string_on_fills[i]):
        return None

    roles = dict(zip(WALKED_ROLES, measured.role_hex[i], strict=True))
    return {
        "ground": measured.ground_hex[i],
        **roles,
        "number": roles["string"],
        "variable": roles["ink"],
        "find_fill": measured.fill_hex[i],
        "find_current": measured.current_hex[i],
        "find_other": measured.other_hex[i],
        # Distance from everything the highlight has to win against, which is the right
        # measure for visual search rather than for discrimination.
        "salience": round(min(gap(FIND_CURRENT, other) for other in (GROUND, *MEANING_ROLES)), 2),
        "body_ratio": round(float(measured.contrast[i][:BODY_ROLE_COUNT].min()), 2),
    }


def _realize_batch(thetas, polarity):
    """The colour work for a batch of thetas: a list in the same order, None where refused.

    Batched because colour-science's cost is per CALL rather than per colour -- measured, a
    call converting one colour costs 312 us and a call converting sixty-four costs 325 us.
    Uncached; go through realize_many.
    """
    table = np.asarray(thetas, dtype=float).reshape(-1, 9)
    night = polarity == "night"
    ground_lightness, ground_hue, ground_rgb = _grounds(table, night)
    role_ab, ratios = _role_chromaticities(table, polarity, ground_hue)
    role_rgb, _walked_grounds = _walk_to_both_bars(role_ab, ground_rgb, ratios, night)
    fill_rgb = _find_fills(table, ground_lightness, night)
    measured = _quantize_and_measure(role_rgb, ground_rgb, fill_rgb)
    return [_assemble(measured, i, polarity) for i in range(len(table))]


#: How conspicuous a find highlight must be before a TIMED hunt may use it, in multiples
#: of the measured discrimination threshold.
#:
#: These are two different perceptual questions and the instrument was conflating them.
#: DE_MIN is a DISCRIMINATION threshold: can two patches be told apart, side by side, at
#: 104 px. Finding one highlighted token in a page of code is VISUAL SEARCH, which needs
#: conspicuity -- the target has to win against every distractor at a glance -- and that
#: takes many multiples of a discrimination step. realize() only ever required the current
#: highlight to sit 1.5x the threshold from the ground, so themes at ~2x came through, and
#: the active sampler sought exactly those out because an unexplored corner is where a
#: GP's variance is highest.
#:
#: 4x, from the logged hunts. Over 33 usable trials, salience correlates with log find time
#: at -0.43 (day) and -0.37 (night) -- more conspicuous really is faster -- and splitting at
#: the median gives 3489 ms against 2066 ms by day, 2897 against 2225 by night. A faint
#: highlight costs well over a second, which is not a measurement of the theme but of
#: patience. 4x excludes roughly the slowest quarter of what has been shown.
#:
#: Applied per ARM, deliberately, and not inside realize(). A quiet highlight is a
#: legitimate thing to PREFER, and theta 8 exists to find that trade-off, so duels must
#: keep exploring it. What a faint highlight destroys is the timed hunt, so that is where
#: the floor belongs.
CONSPICUITY_FLOOR = 4.0


def conspicuous_enough(theme, polarity):
    """Is this theme's find highlight loud enough for a timed hunt to mean anything?

    `salience` is the minimum CAM16-UCS distance from the current highlight to the ground
    and to every coloured role -- distance from everything it has to win against, which is
    the right measure for search rather than for discrimination.
    """
    return theme is not None and theme["salience"] >= CONSPICUITY_FLOOR * DE_MIN[polarity]


def realize_many(thetas, polarity):
    """Realize a batch of thetas; a list in the same order, None where refused.

    Cache first, then one batched call for whatever is left. Both halves matter and the
    measurement says why. Batching alone, ignoring the cache, made the search THREE TIMES
    SLOWER: a trial re-proposes most of its candidates from one sitting to the next, so the
    cache was already answering the majority of them, and recomputing a batch is slower
    than not computing at all. Caching alone leaves a cold sitting paying 312 us of
    library-validation overhead per theme, which was 3.8 s of a 4 s trial.
    """
    table = np.asarray(thetas, dtype=float).reshape(-1, 9)
    keys = [theta_key(row, polarity) for row in table]
    missing = [i for i, key in enumerate(keys) if key not in REALIZE_CACHE]
    if missing:
        for i, theme in zip(missing, _realize_batch(table[missing], polarity), strict=True):
            REALIZE_CACHE[keys[i]] = theme
    return [REALIZE_CACHE[key] for key in keys]


def realize_uncached(theta, polarity):
    """One theme, ignoring the cache. Defined through the batch path so there is exactly
    one implementation of the colour rules -- two copies of a constraint set drift, and a
    drifted floor is a stimulus nobody chose."""
    return _realize_batch(np.asarray(theta, dtype=float)[None, :], polarity)[0]


def realize(theta, polarity):
    """theta in [0,1]^9 -> a full, floor-satisfying theme (hexes + meta), or None when
    the hard constraints cannot be met. Floors are constraints, never objectives: WCAG
    4.5:1 and APCA |Lc| >= 60 for body tokens (comments >= 4.5:1, |Lc| >= 45), and
    pairwise CAM16-UCS separation >= 2x the measured 104-px threshold between any two
    colored roles and ink — doubled because discrimination collapses toward glyph
    scale; the comprehension probes measure the truth of that margin directly."""
    key = theta_key(theta, polarity)
    if key in REALIZE_CACHE:
        return REALIZE_CACHE[key]
    theme = realize_uncached(theta, polarity)
    REALIZE_CACHE[key] = theme
    return theme


def raw_prior(theta, polarity, theme):
    theta = np.asarray(theta, dtype=float)
    labs = lab([theme[role] for role in ROLE_ORDER] + [theme["ground"]])
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    harmony = float(np.mean([ou_luo_pair(labs[a], labs[b]) for a, b in pairs]))
    # Berlyne: pleasure peaks at intermediate complexity — interior optima on the
    # complexity axes, never a monotone pull to either wall.
    complexity = -1.2 * sum((float(theta[i]) - 0.55) ** 2 for i in (3, 4, 5))
    # Ecological-valence stand-in until specific loved colors are named: the stated warm
    # preference, gently.
    warmth = 0.5 * (float(theta[1]) - 0.5)
    return harmony + complexity + warmth


def _build_pool(pool_thetas):
    """The feasible pool per polarity, and the prior's location and scale over it.

    A fixed, deterministic candidate pool: the acquisition shops here (plus per-trial
    local refinements around the champion), the prior is standardized here, and
    infeasible corners are carved away by the floors rather than penalized.
    """
    pool, stats = {}, {}
    for polarity in ("day", "night"):
        feasible = []
        priors = []
        for theta in pool_thetas:
            theme = realize(theta, polarity)
            if theme is not None:
                feasible.append((theta, theme))
                priors.append(raw_prior(theta, polarity, theme))
        priors = np.array(priors)
        stats[polarity] = (float(priors.mean()), float(priors.std() + 1e-9))
        pool[polarity] = feasible
    return pool, stats


POOL_THETA = np.random.default_rng(0xA55).random((512, 9))
POOL, PRIOR_STATS = _build_pool(POOL_THETA)


def prior_mean(theta, polarity, theme=None):
    """Standardized prior utility (mean 0, sd 0.8 over the feasible pool) so the GP's
    signal variance, not the prior's arbitrary units, sets the scale."""
    key = theta_key(theta, polarity)
    if key in PRIOR_CACHE:
        return PRIOR_CACHE[key]
    if theme is None:
        theme = realize(theta, polarity)
    if theme is None:
        value = 0.0
    else:
        mean, sd = PRIOR_STATS[polarity]
        value = 0.8 * (raw_prior(theta, polarity, theme) - mean) / sd
    PRIOR_CACHE[key] = value
    return value
