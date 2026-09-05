"""Does the nearest-theta search find a palette that was actually in theme space?

Plant a theta, realize it, hand the palette back, and the search has to land within a hex
step of it -- the recovery test for an inverse that has no closed form.
"""

import numpy as np
import pytest

from theme.inverse import MATCHED_ROLES, nearest_theta, palette_distances
from theme.space import realize


@pytest.mark.parametrize("polarity", ["day", "night"])
def test_a_realized_palette_is_recovered_to_within_a_hex_step(polarity):
    planted = np.array([0.7, 0.6, 0.4, 0.5, 0.6, 0.5, 0.5, 0.5, 0.8])
    palette = realize(planted, polarity)
    assert palette is not None, "pick a planted theta the floors accept"
    theta, distance = nearest_theta(palette, polarity)
    assert distance < 0.6, f"RMS {distance:.2f} dE from a palette that IS in the space"
    recovered = realize(theta, polarity)
    assert palette_distances([recovered], palette)[0] == pytest.approx(distance)


def test_refused_themes_never_win():
    palette = {role: "#808080" for role in MATCHED_ROLES}
    distances = palette_distances([None, {**palette}], palette)
    assert np.isinf(distances[0]) and distances[1] == 0.0
