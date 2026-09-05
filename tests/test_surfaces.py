"""The page and border an applied theme derives from its ground.

Held to the steps read off the hand-chosen Horizon values, in both directions of
lightness, because the elevation system only reads as elevation if the page sits under
the paper and the border stands quietly between them.
"""

import pytest

from theme.color import hex_to_rgb, rgb_to_ucs, wcag
from theme.surfaces import BORDER_STEP, PAGE_STEP, derived_surfaces


@pytest.mark.parametrize("polarity", ["day", "night"])
def test_derived_surfaces_step_away_from_the_paper(polarity):
    ground = {"day": "#f9ecdd", "night": "#212224"}[polarity]
    surfaces = derived_surfaces(ground, polarity)
    lightness = rgb_to_ucs(hex_to_rgb([ground, surfaces["page"], surfaces["border"]]))[:, 0]
    assert lightness[1] == pytest.approx(lightness[0] - PAGE_STEP, abs=0.6), "the page sits one step below the paper"
    direction = -1 if polarity == "day" else 1
    assert lightness[2] == pytest.approx(lightness[1] + direction * BORDER_STEP, abs=0.8)
    # A quiet border, not a rule: visible against the page and nowhere near text contrast.
    assert 1.2 < wcag(hex_to_rgb(surfaces["border"]), hex_to_rgb(surfaces["page"]))[0] < 2.0
