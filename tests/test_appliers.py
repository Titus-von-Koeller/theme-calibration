"""The appliers write exactly their region and refuse anything else.

Plant a palette module shaped like loop-to-cluster's, write the region, and check what came
back: the measured values where the hand-written ones stood, the data palettes byte-identical,
a second write replacing in place, a hand-written owned name retired wherever it reappears,
and a refusal whenever the result would not be the module the notebooks import.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from theme.appliers import regions, viz
from theme.color import hex_to_rgb, wcag

ROOT = Path(__file__).resolve().parents[1]

PUBLISHED = {
    "day": {"ground": "#f9ecdd", "page": "#efe2d3", "border": "#cebda6", "ink": "#474442", "comment": "#56524f"},
    "night": {"ground": "#222325", "page": "#1b1c1e", "border": "#36383b", "ink": "#c0bfc0", "comment": "#a2a2a4"},
}

#: A palette module in the shape of loop-to-cluster's: data palettes, a hand-written ink
#: pair under its own comment, and a helper after them.
HAND_WRITTEN = '''"""The color constants."""

OKABE_ITO = {
    "orange": "#E69F00",
    "blue": "#0072B2",
}
BASE = OKABE_ITO["blue"]
RAMP = ["#00224e", "#434e6c", "#7d7c78", "#bcae6c", "#fee838"]

# Ink for text set on a known square fill.
INK_LIGHT = "#ffffff"
INK_DARK = "#15181d"


def tint(color, toward_white):
    return color
'''


def _module(source: str) -> dict:
    namespace: dict = {}
    exec(compile(source, "<palette>", "exec"), namespace)
    return namespace


def test_the_region_takes_the_place_of_the_hand_written_pair():
    written = viz.apply_to_source(HAND_WRITTEN, PUBLISHED)
    module = _module(written)
    assert module["INK_LIGHT"] == "#f9ecdd" and module["INK_DARK"] == "#222325", "the two papers are the ink pair"
    assert module["FURNITURE"]["day"] == {
        "paper": "#f9ecdd",
        "page": "#efe2d3",
        "ink": "#474442",
        "label": "#56524f",
        "grid": "#cebda6",
        "axis": "#cebda6",
    }
    assert (
        module["RAMP"] == _module(HAND_WRITTEN)["RAMP"] and module["OKABE_ITO"] == _module(HAND_WRITTEN)["OKABE_ITO"]
    )
    # Installed where the hand-written pair stood, under its comment, above the helper.
    comment = written.index("# Ink for text set on a known square fill.")
    assert comment < written.index("# >>> measured:viz-furniture") < written.index("def tint")
    assert written.count("INK_LIGHT =") == 1 and written.count("INK_DARK =") == 1


def test_a_second_write_replaces_the_region_in_place():
    once = viz.apply_to_source(HAND_WRITTEN, PUBLISHED)
    moved = {**PUBLISHED, "day": {**PUBLISHED["day"], "ground": "#f5e6d0"}}
    twice = viz.apply_to_source(once, moved)
    assert twice.count("# >>> measured:viz-furniture") == 1
    assert _module(twice)["INK_LIGHT"] == "#f5e6d0"
    assert _module(twice)["FURNITURE"]["day"]["paper"] == "#f5e6d0"
    assert viz.apply_to_source(once, PUBLISHED) == once, "an unchanged palette is a no-op"


def test_a_hand_written_owned_name_is_retired_on_every_write():
    """The applier's own lesson: a region owning a name retires it wherever it reappears,
    not only when the region is born."""
    once = viz.apply_to_source(HAND_WRITTEN, PUBLISHED)
    drifted = once + '\nINK_DARK = "#000000"\nFURNITURE = {\n    "day": {},\n}\n'
    twice = viz.apply_to_source(drifted, PUBLISHED)
    assert twice.count("INK_DARK =") == 1 and twice.count("FURNITURE =") == 1
    assert _module(twice)["INK_DARK"] == "#222325"


def test_a_palette_that_would_break_the_module_is_refused():
    broken = {**PUBLISHED, "day": {**PUBLISHED["day"], "ground": '#f9ec"dd'}}
    with pytest.raises(viz.RefusedWrite, match="does not import"):
        viz.apply_to_source(HAND_WRITTEN, broken)


def test_a_write_that_would_touch_a_data_palette_is_refused(monkeypatch):
    """Data marks are chosen by discriminability, never by the theme; the guard is
    structural."""
    body = [*viz.region_body(PUBLISHED), "RAMP = []"]
    monkeypatch.setattr(viz, "region_body", lambda published: body)
    with pytest.raises(viz.RefusedWrite, match="data palette RAMP"):
        viz.apply_to_source(HAND_WRITTEN, PUBLISHED)


def test_a_missing_polarity_is_refused():
    with pytest.raises(viz.RefusedWrite, match="needs both papers"):
        viz.apply_to_source(HAND_WRITTEN, {"day": PUBLISHED["day"]})


def test_retirement_removes_a_multi_line_assignment_whole():
    text = 'A = 1\nFURNITURE = {\n    "day": {},\n}\nB = 2\n'
    retired, at_line = regions.retire_assignments(text, "viz-furniture", ("FURNITURE",))
    assert retired == "A = 1\nB = 2\n" and at_line == 2


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is the environment's formatter")
def test_the_written_region_is_ruff_format_stable(tmp_path):
    """loop-to-cluster's pre-commit runs `ruff format --check`; a region the formatter would
    rewrite is a region that blocks every commit there."""
    target = tmp_path / "_palette.py"
    target.write_text(viz.apply_to_source(HAND_WRITTEN, PUBLISHED))
    result = subprocess.run(["ruff", "format", "--check", "--line-length", "119", str(target)], capture_output=True)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()


def _contrast(a: str, b: str) -> float:
    return float(wcag(hex_to_rgb(a), hex_to_rgb(b))[0])


def _along(stops: list[str], t: float) -> str:
    """Vega's piecewise-linear sRGB interpolation of a scheme, as a hex."""
    n = len(stops) - 1
    i = min(int(t * n), n - 1)
    a, b = hex_to_rgb(stops[i])[0], hex_to_rgb(stops[i + 1])[0]
    rgb = a + (b - a) * (t * n - i)
    return "#" + "".join(f"{round(v * 255):02x}" for v in rgb)


def test_the_ink_pair_keeps_the_viz_crossovers():
    """`_viz.py` switches ink at 0.48 of the cividis ramp and 0.71 of either polarity arm,
    calibrated against a white/near-black pair. The papers must keep those switches honest:
    the light ink must still win below the ramp's switch and the dark ink above it, and the
    same on the arms. When a published paper moves enough to fail this, `_viz.py`'s constants
    need re-measuring -- this is the test that says so."""
    ramp = ["#00224e", "#434e6c", "#7d7c78", "#bcae6c", "#fee838"]
    polarity = ["#8f3413", "#d95926", "#eaa886", "#e8e8e6", "#93bae9", "#2a78d6", "#173f6e"]
    light, dark = viz.ink_pair(PUBLISHED)
    for t in np.linspace(0.0, 0.46, 24):
        assert _contrast(light, _along(ramp, t)) > _contrast(dark, _along(ramp, t)), f"light ink loses at {t:.2f}"
    for t in np.linspace(0.52, 1.0, 24):
        assert _contrast(dark, _along(ramp, t)) > _contrast(light, _along(ramp, t)), f"dark ink loses at {t:.2f}"
    for arm in (0.5 - 0.74 / 2, 0.5 + 0.74 / 2):  # beyond 0.71 of either arm the light ink wins
        assert _contrast(light, _along(polarity, arm)) > _contrast(dark, _along(polarity, arm))
    assert _contrast(dark, _along(polarity, 0.5)) > _contrast(light, _along(polarity, 0.5))
