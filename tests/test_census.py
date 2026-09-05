"""Does the census find the foreign colour and forgive the blends?

A synthetic screenshot: the palette's paper, a block of ink, an antialiased edge between
them, a tinted border, and one loud badge in a colour the palette never produced. The
badge is the only thing the census may report.
"""

import numpy as np
from PIL import Image

from theme.census import census, known_colours
from theme.color import composite, hex_to_rgb

PALETTE = {
    "ground": "#f9ecdd",
    "page": "#efe2d3",
    "border": "#cebda6",
    "ink": "#474442",
    "comment": "#56524f",
    "punct": "#5a5754",
    "keyword": "#7f0179",
    "function": "#004b64",
    "string": "#7d2800",
    "find_fill": "#00d6ab",
    "signals": {"red": "#c83023"},
}


def _rgb(hex_):
    return (hex_to_rgb(hex_)[0] * 255).round().astype(np.uint8)


def test_the_census_reports_the_badge_and_nothing_else(tmp_path):
    canvas = np.tile(_rgb(PALETTE["ground"]), (400, 600, 1))
    canvas[50:150, 50:350] = _rgb(PALETTE["ink"])
    # An antialiased edge: the midpoint blend between ink and paper, one row deep.
    canvas[150:152, 50:350] = ((_rgb(PALETTE["ink"]).astype(int) + _rgb(PALETTE["ground"]).astype(int)) // 2).astype(
        np.uint8
    )
    canvas[200:204, :] = _rgb(composite(PALETTE["border"], 0x80 / 255, PALETTE["ground"]))
    canvas[300:340, 500:560] = _rgb("#e84a72")  # Horizon's pink badge
    canvas[350:380, 500:560] = _rgb(PALETTE["signals"]["red"])
    path = tmp_path / "shot.png"
    Image.fromarray(canvas).save(path)
    rows = census(path, PALETTE)
    assert [row[1] for row in rows] == ["#e84a72"], rows
    assert rows[0][7] is True, "a badge is loud"


def test_known_colours_include_the_signals_and_the_tints():
    known = known_colours(PALETTE)
    assert known["#c83023"] == "signal red"
    assert any(label.startswith("border tint") for label in known.values())
