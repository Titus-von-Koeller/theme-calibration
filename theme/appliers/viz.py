"""Write the graph FURNITURE into loop-to-cluster's palette module from the published palette.

    pixi run python -m theme.appliers.viz            # show what would change, touch nothing
    pixi run python -m theme.appliers.viz --apply    # write it

A chart has two kinds of colour. Its furniture -- the canvas, the axis ink, tick labels,
gridlines, axis lines -- carries no data and is THEME colour: it derives from the one
published palette exactly as the editor's frame does, so a chart's canvas is the code paper,
its axis ink the measured ink, its gridlines the border tint (the method reef, "Widgets and
graphs"). Its data marks -- the cividis ramp, the Okabe-Ito set, the polarity pair -- carry
information and are NOT theme colour: they are chosen by discriminability under the fitted
observer at mark size, and this applier never touches them. The refusal below makes that
structural: a write that would change any data palette is refused, not warned about.

The target is `notebooks/pytorch-basics/_palette.py` in loop-to-cluster, the torch-free
constants every notebook there draws from. The region it owns holds `FURNITURE`, keyed by
polarity because a notebook is read on either and only the notebook knows which
(`mo.app_meta().theme`), and the ink pair `INK_LIGHT` / `INK_DARK` for text set on a known
data fill: the two papers. On a coloured fill the ink is the paper, as it is on a button
(the controls rule), and a census then classifies it as in-system where pure white and a
near-black were foreign. Measured against the ramps (2026-09-05): the day paper on the dark
end and the night paper on the light end move the ramp's ink crossover from 0.500 to 0.495
and the polarity crossovers from 0.65/0.71 to 0.66/0.72, so `_viz.py`'s constants (0.48,
0.71) stand; the contrast at the crossover falls from 4.2:1 to 3.6:1, which no near-neutral
pair clears there either (the floor at a mid-grey fill is out of reach for anything but
pure black).

Same marker discipline as the dotfiles applier, and the same refusal: settings.jsonc must
still parse, this module must still import. The result is compiled and executed in a fresh
namespace before anything is written; if that raises, if an owned name is missing, or if a
data palette differs from the file's own, nothing is written and the reason is printed.
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

from .. import paths
from .regions import splice

#: Where loop-to-cluster keeps the constants every notebook draws from.
TARGET = Path.home() / "src/loop-to-cluster/notebooks/pytorch-basics/_palette.py"

REGION = "viz-furniture"

#: The names the region defines; a hand-written assignment to any of them is retired.
OWNED = ("INK_LIGHT", "INK_DARK", "FURNITURE")

#: The names a write may never change: the data palettes and the scheme names beside them.
DATA_PALETTES = ("OKABE_ITO", "BASE", "ACCENT", "SEQUENTIAL_SCHEME", "DIVERGING_SCHEME", "RAMP", "POLARITY")

#: Furniture role -> palette role, one polarity. The canvas is the code paper and the page
#: under it the notebook page, so a chart sits on the same ground as the code beside it;
#: titles and axis labels take the ink; tick labels take the comment step, the same step the
#: applier gives every secondary text; gridlines and axis lines are the border tint, a quiet
#: frame in the paper's own material rather than a grey rule.
FURNITURE_ROLES = {
    "paper": "ground",
    "page": "page",
    "ink": "ink",
    "label": "comment",
    "grid": "border",
    "axis": "border",
}


class RefusedWrite(Exception):
    """The result would not be the module the notebooks import; nothing was written."""


def furniture(palette: dict) -> dict:
    """One polarity's graph furniture from its published palette."""
    return {role: palette[source] for role, source in FURNITURE_ROLES.items()}


def ink_pair(published: dict) -> tuple[str, str]:
    """(light ink, dark ink) for text on a known data fill: the day and the night paper."""
    return published["day"]["ground"], published["night"]["ground"]


def region_body(published: dict) -> list[str]:
    """The lines between the markers. Multi-line dicts with trailing commas, so `ruff format`
    leaves them exactly as written and the pre-commit hook in loop-to-cluster stays green."""
    light, dark = ink_pair(published)
    lines = [
        "# Written by theme-calibration (`python -m theme.appliers.viz`) from the published palette;",
        "# never by hand. The ink on a coloured data fill is the paper -- day's on the dark end of a",
        "# ramp, night's on the light end -- as it is on a button. The data palettes above are chosen",
        "# by discriminability and are not furniture: the applier refuses any write that changes them.",
        f'INK_LIGHT = "{light}"',
        f'INK_DARK = "{dark}"',
        "",
        "# Graph FURNITURE, keyed by polarity: a chart's canvas is the code paper, the page under it",
        "# the notebook page, its axis ink the measured ink, tick labels the comment step, gridlines",
        "# and axis lines the border tint. Only the notebook knows which polarity it is read on",
        "# (mo.app_meta().theme), so both are here.",
        "FURNITURE = {",
    ]
    for polarity in ("day", "night"):
        lines.append(f'    "{polarity}": {{')
        lines += [f'        "{role}": "{colour}",' for role, colour in furniture(published[polarity]).items()]
        lines.append("    },")
    lines.append("}")
    return lines


def apply_to_source(text: str, published: dict) -> str:
    """The module's source with the region written; refuses rather than return a broken
    module."""
    missing = [polarity for polarity in ("day", "night") if polarity not in published]
    if missing:
        raise RefusedWrite(f"the palette publishes no {' or '.join(missing)}; the ink pair needs both papers")
    new = splice(text, REGION, region_body(published), OWNED)
    refuse_unless_faithful(text, new)
    return new


def _namespace(source: str) -> dict:
    namespace: dict = {}
    exec(compile(source, "_palette.py", "exec"), namespace)  # the module we are about to write
    return namespace


def refuse_unless_faithful(old: str, new: str) -> None:
    """The new source must import, define every owned name, and leave every data palette as
    the old source had it."""
    try:
        before, after = _namespace(old), _namespace(new)
    except Exception as exc:  # any failure to import is the refusal
        raise RefusedWrite(f"the result does not import: {exc!r}") from exc
    for name in OWNED:
        if name not in after:
            raise RefusedWrite(f"the result does not define {name}")
    for name in DATA_PALETTES:
        if name in before and before[name] != after.get(name):
            raise RefusedWrite(f"the write would change the data palette {name}; data marks are not furniture")
    foreign = {k for k in after if not k.startswith("__")} - {k for k in before if not k.startswith("__")} - set(OWNED)
    if foreign:
        raise RefusedWrite(f"the write would define names the module never had: {sorted(foreign)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the file (default: diff only)")
    parser.add_argument("--champion", type=Path, default=paths.CHAMPION, help="the published palette")
    parser.add_argument("--target", type=Path, default=TARGET, help="the palette module to write")
    args = parser.parse_args(argv)

    if not args.champion.exists():
        print(f"no published champion at {args.champion}; run `pixi run publish` first", file=sys.stderr)
        return 1
    if not args.target.exists():
        print(f"no palette module at {args.target}", file=sys.stderr)
        return 1
    published = json.loads(args.champion.read_text())
    for polarity in ("day", "night"):
        if polarity in published:
            p = published[polarity]
            print(f"{polarity:6s} paper {p['ground']}  ink {p['ink']}  border {p['border']}  verdict {p['verdict']}")

    old = args.target.read_text()
    try:
        new = apply_to_source(old, published)
    except RefusedWrite as refusal:
        print(f"\nREFUSING TO WRITE: {refusal}.", file=sys.stderr)
        print("Nothing was changed.", file=sys.stderr)
        return 2
    if new == old:
        print(f"\n{args.target.name} already matches the published palette.")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(old.splitlines(True), new.splitlines(True), "_palette.py", "_palette.py (measured)")
    )
    if not args.apply:
        print("\n(dry run -- pass --apply to write)")
        return 0
    args.target.write_text(new)
    print(f"\nwrote {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
