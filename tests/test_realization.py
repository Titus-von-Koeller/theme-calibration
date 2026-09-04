"""What must be true of EVERY theme the instrument is willing to show.

The floors are constraints, not objectives: WCAG 4.5:1 and APCA |Lc| >= 60 for body
tokens, |Lc| >= 45 for comments, and pairwise CAM16-UCS separation of at least twice the
measured discrimination threshold between any two coloured roles. Every candidate shown
has already cleared them; the only question ever put to the observer is which is BETTER.

That makes it an invariant rather than an example, so it is tested as one. Hypothesis
searches the nine-dimensional parameter space for a theta that produces a theme violating
a floor -- a case a hand-picked sample would miss -- and shrinks any counterexample to
something readable. `realize` is allowed to refuse (return None) when the constraints
cannot be met; what it may never do is return a theme that breaks one.

Every floor here is re-measured from the theme's own HEX strings, never from the
continuous colour the bisection produced. That is not incidental: a floor checked before
hex quantization was the one real bug this suite has caught, and re-measuring from the
hexes is what makes these tests able to catch its siblings. Rounding to 8 bits moves a
colour by up to half a step, which is enough to cross a bar.
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
    coords = _ucs_of(theme, named)

    accents = ("keyword", "function", "string")
    owed = [(a, b, 2.0) for i, a in enumerate(accents) for b in (*accents[i + 1 :], "ink")]
    owed += [("comment", "ink", 1.0), ("find_current", "ground", 1.5), ("find_current", "find_other", 1.0)]
    for first, second, multiple in owed:
        margin = multiple * threshold
        gap = _gap(coords, first, second)
        assert gap >= margin - 1e-6, (
            f"{polarity}: {first} and {second} are {gap:.2f} apart, "
            f"inside the {margin:.2f} they are owed ({multiple:g}x the measured threshold)"
        )


@given(theta=thetas, polarity=polarities)
@settings(max_examples=150, deadline=None)
def test_text_stays_readable_sitting_on_a_find_highlight(theta, polarity):
    """The fill floors, which no test asserted until now.

    A search hit paints a fill UNDER text that is still supposed to be read: the current
    match at 85% alpha, every other match at 45%. `realize` refuses a theme whose ink or
    string cannot survive either fill, and this is that refusal stated as an invariant.

    Both floors are below the 4.5:1 the page itself owes, deliberately -- a highlight is
    a transient state the eye is already pointed at, and holding it to body-text contrast
    would forbid every fill loud enough to find. What must not happen is a highlight that
    hides the token it was drawn to reveal.

    Measured on the COMPOSITED, quantized fill, because that is the pixel: an alpha
    emitted into the theme file is contrast nobody has checked until it is composited.
    """
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return
    fills = hex_to_rgb([theme["find_current"], theme["find_other"]])
    for role, floor in (("ink", 4.0), ("string", 3.5)):
        text = np.repeat(hex_to_rgb([theme[role]]), 2, axis=0)
        ratios = wcag(text, fills)
        for fill_name, ratio in zip(("find_current", "find_other"), ratios, strict=True):
            assert float(ratio) >= floor - 1e-6, (
                f"{polarity}: {role} on {fill_name} is {float(ratio):.2f}:1, under the {floor}:1 floor"
            )


@given(theta=thetas, polarity=polarities)
@settings(max_examples=120, deadline=None)
def test_the_reported_salience_is_the_distance_to_everything_it_competes_with(theta, polarity):
    """`salience` decides which themes a TIMED hunt may use, so it has to be the real number.

    conspicuous_enough() gates the hunt arm on it, which means a salience computed on the
    continuous colour rather than the rendered one would admit highlights the page never
    actually shows. Recomputed here from the theme's own hexes -- the only values a
    downstream reader has -- and compared against what realize() published.
    """
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return
    competitors = ["ground", "keyword", "function", "string", "ink"]
    coords = _ucs_of(theme, [*competitors, "find_current"])
    expected = min(_gap(coords, "find_current", other) for other in competitors)
    assert abs(theme["salience"] - expected) <= 0.005 + 1e-9, (
        f"{polarity}: published salience {theme['salience']} but the rendered colours give {expected:.4f}"
    )


@given(theta=thetas, polarity=polarities)
@settings(max_examples=120, deadline=None)
def test_the_reported_body_ratio_is_the_worst_rendered_body_contrast(theta, polarity):
    """`body_ratio` is what the analysis reads back to say how contrasty a theme was.

    It is the weakest of the four body tokens against the page, and it must be measured on
    the hexes for the same reason the floor is: half a quantization step is enough to move
    it. A theme whose theta asked for more contrast than its ground can deliver reports
    what it got, not what it asked for -- solve_j saturates silently, so this number is
    the only place that shows.
    """
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return
    ground = np.repeat(hex_to_rgb([theme["ground"]]), len(BODY_ROLES), axis=0)
    ratios = wcag(hex_to_rgb([theme[role] for role in BODY_ROLES]), ground)
    worst = float(ratios.min())
    assert abs(theme["body_ratio"] - worst) <= 0.005 + 1e-9, (
        f"{polarity}: published body_ratio {theme['body_ratio']} but the rendered colours give {worst:.4f}"
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
@settings(max_examples=40, deadline=None)
def test_a_theta_realises_the_same_alone_as_it_does_in_a_batch(theta, polarity):
    """Batch size must not be able to change a stimulus, and it is not obvious that it cannot.

    The APCA retry loop is coupled across the batch: it stops when NO row is still short
    of its Lc floor, so a theme sharing a batch with a stubborn one gets its bisection
    re-run more times than it would alone. That is only harmless because the extra runs
    use an unchanged target and so land on the same colour -- an assumption every batched
    caller here rests on, including the pool build, and one worth testing rather than
    arguing.

    Deliberately through the uncached batch entry point. Going through realize() would
    make the second call a cache hit and the assertion a tautology.
    """
    alone = space._realize_batch(np.array([theta]), polarity)[0]
    in_company = space._realize_batch(np.array([theta, [0.5] * 9, [0.0] * 9, [1.0] * 9]), polarity)[0]
    assert alone == in_company


@given(theta=thetas, polarity=polarities)
@settings(max_examples=60, deadline=None)
def test_every_theme_defines_every_role_the_renderer_asks_for(theta, polarity):
    """A missing key is a KeyError mid-render, which reaches the observer as a blank page."""
    theme = space.realize(np.array(theta), polarity)
    if theme is None:
        return
    for role in (*BODY_ROLES, *QUIET_ROLES, "ground", "find_current", "find_other", "salience"):
        assert role in theme, f"realised {polarity} theme is missing {role!r}"


def _ucs_of(theme, names):
    """The named colours of a theme in CAM16-UCS, converted from the theme's own hexes."""
    return dict(zip(names, rgb_to_ucs(hex_to_rgb([theme[name] for name in names])), strict=True))


def _gap(coords, first, second):
    return float(np.linalg.norm(coords[first] - coords[second]))
