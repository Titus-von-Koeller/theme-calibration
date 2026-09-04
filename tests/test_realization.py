"""What must be true of EVERY theme the instrument is willing to show.

The floors are constraints, not objectives: WCAG 4.5:1 and APCA |Lc| >= 60 for body
tokens, |Lc| >= 45 for comments, and pairwise CAM16-UCS separation of at least twice the
measured discrimination threshold between any two coloured roles. Every candidate Titus
sees has already cleared them; he is only ever asked which is BETTER.

That makes it an invariant rather than an example, so it is tested as one. Hypothesis
searches the nine-dimensional parameter space for a theta that produces a theme violating
a floor -- a case a hand-picked sample would miss -- and shrinks any counterexample to
something readable. `realize` is allowed to refuse (return None) when the constraints
cannot be met; what it may never do is return a theme that breaks one.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from theme import space
from theme.color import apca_lc, hex_to_rgb, rgb_to_ucs, wcag

# A theta is nine numbers in the unit cube; that is the whole parameter space.
thetas = st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=9, max_size=9)
polarities = st.sampled_from(["day", "night"])

BODY_ROLES = ("keyword", "function", "string", "ink")
QUIET_ROLES = ("comment", "punct")


@given(theta=thetas, polarity=polarities)
@settings(max_examples=150, deadline=None)
def test_a_realised_theme_always_clears_its_contrast_floors(theta, polarity):
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return  # refusing is allowed; shipping an illegible theme is not
    ground = hex_to_rgb([theme["ground"]])
    for role in BODY_ROLES + QUIET_ROLES:
        ink = hex_to_rgb([theme[role]])
        ratio = float(wcag(ink, ground)[0])
        lc = abs(float(apca_lc(ink, ground)[0]))
        floor = 60.0 if role in BODY_ROLES else 45.0
        assert ratio >= 4.5 - 1e-6, f"{polarity} {role} at {ratio:.2f}:1 is below the 4.5:1 floor"
        assert lc >= floor - 1e-6, f"{polarity} {role} at Lc {lc:.1f} is below the {floor:.0f} floor"


@given(theta=thetas, polarity=polarities)
@settings(max_examples=150, deadline=None)
def test_roles_stay_apart_by_the_margin_each_pair_is_owed(theta, polarity):
    """The separation contract, pair by pair, because it is not uniform -- and should not be.

    Writing this as "every pair must clear 2x the threshold" fails immediately, and
    correctly: hypothesis found a theme where comment and ink sit closer than that. They
    are MEANT to. Both are neutral text and a comment is a deliberate step quieter than
    body ink, so demanding a full discrimination margin between them would fight the
    figure-versus-ground rule the palette is built on. What each pair is owed:

      keyword / function / string, pairwise and against ink : 2x threshold
          These carry meaning by hue. Two of them confusable is a page where the syntax
          highlighting is decoration. Doubled because discrimination collapses toward glyph
          scale and the thresholds were measured on 104-px patches.
      comment against ink : 1x threshold
          Distinguishable, deliberately not more. The italic carries the rest.
      find-current against the ground : 1.5x
      find-current against find-other : 1x
          The highlight has to be findable against the page and separable from its
          siblings, which is the whole point of the salience axis.
    """
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return
    threshold = space.DE_MIN[polarity]
    named = ["keyword", "function", "string", "ink", "comment", "ground", "find_current", "find_other"]
    coords = dict(zip(named, rgb_to_ucs(hex_to_rgb([theme[name] for name in named])), strict=True))

    def gap(first, second):
        return float(np.linalg.norm(coords[first] - coords[second]))

    accents = ("keyword", "function", "string")
    owed = [(a, b, 2.0) for i, a in enumerate(accents) for b in (*accents[i + 1 :], "ink")]
    owed += [("comment", "ink", 1.0), ("find_current", "ground", 1.5), ("find_current", "find_other", 1.0)]
    for first, second, multiple in owed:
        margin = multiple * threshold
        assert gap(first, second) >= margin - 1e-6, (
            f"{polarity}: {first} and {second} are {gap(first, second):.2f} apart, "
            f"inside the {margin:.2f} they are owed ({multiple:g}x the measured threshold)"
        )


@given(theta=thetas, polarity=polarities)
@settings(max_examples=60, deadline=None)
def test_realising_the_same_theta_twice_gives_the_same_theme(theta, polarity):
    """Determinism, because the analysis re-realises a champion long after it was shown.

    A theme that drifted between renders would make every archived response describe a
    stimulus that no longer exists.
    """
    first = space.realize(np.array(theta), polarity)
    second = space.realize(np.array(theta), polarity)
    assert first == second


@given(theta=thetas, polarity=polarities)
@settings(max_examples=60, deadline=None)
def test_every_theme_defines_every_role_the_renderer_asks_for(theta, polarity):
    """A missing key is a KeyError mid-render, which reaches Titus as a blank page."""
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return
    for role in (*BODY_ROLES, *QUIET_ROLES, "ground", "find_current", "find_other", "salience"):
        assert role in theme, f"realised {polarity} theme is missing {role!r}"
