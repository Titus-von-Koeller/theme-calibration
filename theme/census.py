"""A colour census of a screenshot against the published palette: what still speaks
another theme?

    pixi run census -- shot.png [--polarity day]

Coherence was judged by squinting until 2026-09-05, and squinting misses exactly what
this finds: the badge in Horizon's pink, the workbench text a step darker than the ink,
the notebook card frame baked as a literal. Every pixel colour that covers more than a
declared share of the screenshot is classified as one of the palette's colours, a derived
surface, one of their alpha composites, an antialiasing blend between two of them, or
FOREIGN -- and the foreign ones are the work list. Two model-free renders (grayscale,
saturation-only) are written beside the screenshot for the eye's second opinion; the table
is the measurement.

The second table is the observer's. A population colour-deficiency filter was here first
and was removed the same day (Titus: it biases the judgement toward a spectrum position he
has not been measured to hold). Instead the census asks the ONE observer model in
`theme.observer` -- fitted from his own discrimination log -- which pairs of colours on
this screen his eyes tell apart worst. As the vision arm accumulates trials the fit sharpens
and this table tightens on its own; nothing here encodes an assumption about his eyes that
the log did not put there. Predictions are made at the patch size the log has measured; the
glyph-scale prediction waits on the size exponent, and the table says which regime it is in.

Takes a PNG rather than taking the screenshot, so the keyhole (or anyone) supplies the
pixels and this stays a pure function of an image and a palette.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import observer, paths
from .color import composite, hex_to_rgb, rgb_to_ucs
from .thresholds import REFERENCE_SIZE_PX, size_is_identified

#: A colour under this share of the pixels is noise (antialiasing edges, icons) unless it
#: is loud; a loud colour is reported from a far smaller share, because a badge is small.
QUIET_SHARE = 0.0004
LOUD_SHARE = 0.00003
LOUD_CHROMA = 14.0

#: How close a pixel colour must sit to a known colour to count as that colour: about a hex
#: step in CAM16-UCS.
MATCH_DE = 1.6

#: How close, in 8-bit sRGB units, a pixel must sit to the straight segment between two known
#: colours to count as an antialiasing blend of them. Blends happen in the renderer's RGB,
#: not in a perceptual space -- the midpoint of ink and paper sits 12 dE off the UCS segment
#: between them and dead on the sRGB one -- so this test is made where the blending is.
BLEND_RGB = 6.0

#: The alphas the applier writes tints at, so their composites are known colours too.
TINT_ALPHAS = (0x99, 0x80, 0xB3, 0x66, 0x4D, 0xD9, 0x73)

#: The P(correct) under which the observer's pairwise check reports a pair as hard to tell
#: apart (chance in the four-alternative task is 0.25).
CONFUSION_P = 0.90


def known_colours(palette):
    """{hex: name} for everything the palette puts on screen, composites included."""
    known = {}
    for role in ("ground", "page", "border", "ink", "comment", "punct", "keyword", "function", "string", "find_fill"):
        known[palette[role].lower()] = role
    for name, hex_ in palette.get("signals", {}).items():
        known[hex_.lower()] = f"signal {name}"
    for alpha in TINT_ALPHAS:
        for base in ("ground", "page"):
            known.setdefault(
                composite(palette["border"], alpha / 255, palette[base]).lower(), f"border tint on {base}"
            )
        known.setdefault(composite(palette["find_fill"], alpha / 255, palette["ground"]).lower(), "find tint")
        known.setdefault(composite(palette["function"], alpha / 255, palette["page"]).lower(), "accent tint on page")
    return known


def _blend_distance(point, ends):
    """Distance from `point` to the nearest segment between any two of `ends`, in the
    units `ends` are given in."""
    best = np.inf
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            a, b = ends[i], ends[j]
            direction = b - a
            length = float(direction @ direction)
            t = 0.0 if length == 0 else float(np.clip(((point - a) @ direction) / length, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (a + t * direction))))
    return best


def classify(hex_, known, known_ucs, known_rgb):
    """(verdict, label, distance): in-system by identity, by blend, or foreign."""
    point = rgb_to_ucs(hex_to_rgb(hex_))[0]
    distances = np.linalg.norm(known_ucs - point, axis=1)
    nearest = int(np.argmin(distances))
    label = list(known.values())[nearest]
    if distances[nearest] <= MATCH_DE:
        return "in", label, float(distances[nearest])
    # Antialiased glyph edges are blends of an ink and its ground, made in the renderer's
    # sRGB; the segment test runs there, over every pair of known colours.
    blend = _blend_distance(hex_to_rgb(hex_)[0] * 255.0, known_rgb)
    if blend <= BLEND_RGB:
        return "blend", "an antialiasing blend of two known colours", blend
    return "foreign", f"nearest {label}", float(distances[nearest])


def census(png_path, palette):
    """Rows of (verdict, hex, pixels, share, label, distance, box, loud) worth reading, loud
    first."""
    image = Image.open(png_path).convert("RGB")
    pixels = np.asarray(image)
    height, width, _ = pixels.shape
    flat = pixels.reshape(-1, 3).astype(np.int64)
    codes = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    counts = collections.Counter(codes.tolist())
    total = height * width
    known = known_colours(palette)
    known_ucs = rgb_to_ucs(hex_to_rgb(list(known)))
    known_rgb = hex_to_rgb(list(known)) * 255.0
    rows = []
    for code, count in counts.most_common():
        share = count / total
        if share < LOUD_SHARE:
            break
        hex_ = f"#{code:06x}"
        ucs = rgb_to_ucs(hex_to_rgb(hex_))[0]
        loud = float(np.hypot(ucs[1], ucs[2])) >= LOUD_CHROMA
        if share < QUIET_SHARE and not loud:
            continue
        verdict, label, distance = classify(hex_, known, known_ucs, known_rgb)
        if verdict != "foreign":
            continue
        ys, xs = np.nonzero(codes.reshape(height, width) == code)
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        rows.append((verdict, hex_, count, share, label, distance, box, loud))
    rows.sort(key=lambda row: (not row[7], -row[2]))
    return rows


def colours_that_mean_something(palette, foreign_rows):
    """The colours whose confusion would matter: every palette role and signal, plus whatever
    the census found foreign. Antialiasing variants of one colour are left out on purpose --
    a pair one hex step apart is at chance for every observer and says nothing."""
    roles = ("ground", "page", "border", "ink", "comment", "punct", "keyword", "function", "string", "find_fill")
    hexes = [palette[role].lower() for role in roles]
    hexes += [hex_.lower() for name, hex_ in palette.get("signals", {}).items() if not name.endswith("_bright")]
    hexes += [row[1] for row in foreign_rows]
    return list(dict.fromkeys(hexes))


def observer_confusions(hexes, ground_hex, fit=None, size_px=None):
    """Pairs among `hexes` the fitted observer tells apart worst on this ground, sorted by
    predicted P(correct) in the four-alternative task; (pair, p, size used, regime note)."""
    fit = fit or observer.fit(paths.VISION_LOG)
    if size_px is None:
        size_px = REFERENCE_SIZE_PX
        regime = f"at the measured {REFERENCE_SIZE_PX:.0f} px; the glyph-scale prediction waits on the size exponent"
        if size_is_identified(fit):
            regime = f"at {REFERENCE_SIZE_PX:.0f} px (the size exponent is identified; pass --size to predict a glyph)"
    else:
        regime = f"at {size_px:.0f} px" + (
            "" if size_is_identified(fit) else " -- scaled by the exponent's PRIOR, not data"
        )
    pairs = []
    for i, first in enumerate(hexes):
        for second in hexes[i + 1 :]:
            p = observer.discriminability(fit, first, second, ground_hex, size_px)
            if p < CONFUSION_P:
                pairs.append((first, second, float(p)))
    pairs.sort(key=lambda row: row[2])
    return pairs, size_px, regime


def write_filters(png_path):
    """Grayscale and saturation-only renders beside the screenshot: model-free views that
    show structure without hue and hue without structure."""
    image = Image.open(png_path).convert("RGB")
    stem = Path(png_path).with_suffix("")
    image.convert("L").save(f"{stem}-gray.png")
    linear = np.asarray(image) / 255.0
    high, low = linear.max(-1), linear.min(-1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-9), 0.0)
    gray = np.asarray(image.convert("L"))[..., None] // 3 + 150
    stray = np.where((saturation > 0.18)[..., None], np.asarray(image), gray)
    Image.fromarray(stray.astype(np.uint8)).save(f"{stem}-saturation.png")
    return [f"{stem}-gray.png", f"{stem}-saturation.png"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("png", type=Path)
    parser.add_argument("--polarity", choices=("day", "night"), default="day")
    parser.add_argument("--champion", type=Path, default=paths.CHAMPION)
    parser.add_argument("--no-filters", action="store_true")
    parser.add_argument("--size", type=float, help="predict discriminability at this patch size in px")
    args = parser.parse_args(argv)
    palette = json.loads(args.champion.read_text())[args.polarity]
    rows = census(args.png, palette)
    print(f"{args.png.name}: {len(rows)} foreign colours (loud ones first)")
    for _verdict, hex_, count, share, label, distance, box, loud in rows:
        flag = "LOUD " if loud else "     "
        print(f"  {flag}{hex_} {count:8d} px {100 * share:6.3f}%  {label} ({distance:.1f} dE)  at {box}")
    if paths.VISION_LOG.exists():
        hexes = colours_that_mean_something(palette, rows)
        pairs, size_px, regime = observer_confusions(hexes, palette["ground"], size_px=args.size)
        print(f"pairs among the {len(hexes)} colours that carry meaning here which your fitted observer")
        print(f"tells apart worst, {regime}:")
        for first, second, p in pairs[:12]:
            print(f"  {first} vs {second}  P(correct) {p:.2f}")
        if not pairs:
            print(f"  none under P(correct) {CONFUSION_P} at {size_px:.0f} px")
    if not args.no_filters:
        for path in write_filters(args.png):
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
