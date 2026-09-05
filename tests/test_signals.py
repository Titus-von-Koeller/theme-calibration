"""Do the signals keep their conventions while clearing the palette's floors?

A signal has to be recognisable as the colour the world agreed on (a red error is red), be
legible on the measured paper by the same floors the body text clears, stay separable from
every syntax accent so a git decoration never reads as a keyword, and stay separable from
each other so a terminal program's six colours are six.
"""

import pytest

from theme.color import apca_lc, hex_to_rgb, ucs_dist, wcag
from theme.signals import LC_FLOOR, SIGNAL_HUES, WCAG_FLOOR, hue_of, signals_for
from theme.space import DE_MIN, separation_floor

PALETTES = {
    "day": {"ground": "#f9ecdd", "accents": ("#7f0179", "#004b64", "#7d2800")},
    "night": {"ground": "#222325", "accents": ("#ffa4cf", "#77c8ff", "#d5be7b")},
}

#: Gamut clipping bends a saturated hue on a light paper; the convention survives well
#: inside this, and a test that asked for less would fail on every light theme in the world.
HUE_TOLERANCE_DEG = 25.0


@pytest.fixture(params=["day", "night"])
def polarity(request):
    return request.param


@pytest.fixture
def signals(polarity):
    palette = PALETTES[polarity]
    return signals_for(palette["ground"], palette["accents"], polarity)


def test_every_signal_clears_the_text_floors_on_its_ground(polarity, signals):
    ground = hex_to_rgb(PALETTES[polarity]["ground"])
    for name, hex_ in signals.items():
        rendered = hex_to_rgb(hex_)
        assert wcag(rendered, ground)[0] >= WCAG_FLOOR, f"{polarity} {name} {hex_} under 4.5:1"
        assert abs(apca_lc(rendered, ground)[0]) >= LC_FLOOR, f"{polarity} {name} {hex_} under Lc {LC_FLOOR}"


def test_every_signal_keeps_its_conventional_hue(polarity, signals):
    for name, hex_ in signals.items():
        anchor = SIGNAL_HUES[name.removesuffix("_bright")]
        drift = (hue_of(hex_) - anchor + 180.0) % 360.0 - 180.0
        assert abs(drift) <= HUE_TOLERANCE_DEG, f"{polarity} {name} {hex_} drifted {drift:+.0f} deg from {anchor}"


def test_signals_stay_clear_of_every_syntax_accent(polarity, signals):
    """A signal that sits within the separation floor of a keyword or a string is a git
    decoration that reads as code. Lightness is what keeps them apart: signals sit at the
    floor, accents deeper."""
    floor = separation_floor(polarity)[0]
    for name, hex_ in signals.items():
        for accent in PALETTES[polarity]["accents"]:
            assert ucs_dist(hex_, accent)[0] >= floor, f"{polarity} {name} {hex_} within {floor:.1f} dE of {accent}"


def test_the_seven_signals_are_mutually_discriminable(polarity, signals):
    names = list(SIGNAL_HUES)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            distance = ucs_dist(signals[first], signals[second])[0]
            assert distance >= 2 * DE_MIN[polarity], f"{polarity} {first} and {second} are {distance:.1f} dE apart"


def test_bright_variants_exist_for_every_signal(signals):
    for name in SIGNAL_HUES:
        assert f"{name}_bright" in signals
