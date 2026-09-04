"""Fresh, unfamiliar code stimuli for the aesthetics instrument — one snippet per trial.

calibrate-aesthetics.py times how long Titus takes to find a named function in a page of
code and reads that time as legibility. It cannot, while the corpus is four snippets from
his own repo: of 116 recorded trials `tensor-ops` was drawn 46 times, and by a page's
fortieth showing the answer's position is recalled rather than found. The log already
shows the damage — across the twenty comprehension probes no theme axis separates from
reaction time (median |r| 0.12 over the nine axes, largest 0.48, none of it surviving
nine comparisons at n = 20), while find-hunt time falls steadily with trial index
(r = -0.47, n = 17). A practice curve on a memorized corpus is what was being measured.

So every trial gets code the instrument has never shown. Two generators, because neither
does both jobs:

- **procedural**, from a small grammar. The only way to hold the token-role mix, the line
  count and the nesting depth *identical* across trials, so a reaction-time difference is
  a difference between themes rather than between pages. Identifiers are invented from
  morpheme lists, which makes novelty structural instead of hoped for: a permutation of
  the seed chooses the names, so seeds under ~17 million apart cannot yield the same page.
- **obscure stdlib**, sliced out of modules he is unlikely to have opened. Real code
  carries the irregular line lengths, comment placement and naming a grammar smooths away,
  and a legibility number measured only on synthetic text has no claim on the editor.

Torch-free and stdlib-only on purpose: the instrument serves under `marimo run`, which
instantiates in a worker thread, and importing torch from a non-main thread can die
mid-import — the same constraint that split _palette.py out of _viz.py.

The returned dict is a drop-in for one of the instrument's own snippet records (`spans`
carry the same role tokenization; `fn_ids` and `ident_ids` index into them), plus the
`target` / `hash` / `role_counts` keys a fresh corpus needs. `fn_ids` holds exactly one
span — the comprehension answer, an identifier occurring exactly once in the page — so the
task has one right click and no position the eye can learn.
"""

import ast
import builtins
import hashlib
import importlib
import inspect
import io
import keyword
import random
import textwrap
import tokenize
from functools import lru_cache
from math import gcd

# The instrument's card is a fixed-width `<pre>` with `overflow:hidden` — a line past the
# card wraps or is clipped, and either one changes the stimulus instead of the theme.
MAX_WIDTH = 100
# Two duel cards share the band, so each gets roughly 80 columns at 14 px — pass
# `max_width` to hold pages inside one card rather than trusting the ceiling above.
DUEL_WIDTH = 64  # a duel's column: short enough that both pages sit near the centre
# of the screen. Two 80-column pages aligned inward still spanned 58% of a 2560px
# panel and read as two edge-ward blocks; 64 brings the pair into the middle where an
# 8K screen is viewed straight on (Titus). It is a ceiling, not a target -- real code
# lines are shorter than it more often than not.
# Nesting past this reads as a different task, and a page indented off the left margin is
# mostly whitespace; both are stimulus changes wearing a theme's clothes.
MAX_NESTING = 4
MAX_INDENT = 24
# Slack for the extraction pass alone: a stdlib line this long is usually a wrapped
# expression whose dedent brings it back under MAX_WIDTH, so it is worth parsing before
# the final width filter rejects whatever is still too wide.
EXTRACT_MAX_WIDTH = 119

KINDS = ("procedural", "stdlib")

# Roles as `_role_spans` labels them. `punct` is deliberately unplanned: bracket and
# operator counts follow from the statements rather than being chosen.
PLANNED_ROLES = ("comment", "string", "number", "keyword", "function", "variable")

# The role mix every procedural trial is built to, so two themes are judged on identical
# token statistics. Centered on what the grammar actually reaches at 14 lines (measured),
# not on a guess: a plan the search cannot hit would silently widen trial-to-trial
# variance in the very quantity this holds constant.
DEFAULT_ROLES = {
    "comment": 2,
    "string": 3,
    "number": 6,
    "keyword": 9,
    "function": 4,
    "variable": 20,
}

# Total absolute deviation from the plan that still counts as the same stimulus class.
ROLE_TOLERANCE = 4

DEFAULT_LINES = 14

# Ecological validity is the minority report: real code every third trial keeps the
# measurement anchored to what an editor shows, while the controlled pages carry the
# statistical weight.
AUTO_STDLIB_EVERY = 3

# Domain vocabulary that would make a page familiar: the four the confound turns on
# (torch, tensor, model, neural) plus the rest of this program's working nouns. Matched as
# substrings, so `models` and `submodel` are caught by `model`.
_FORBIDDEN = (
    "torch",
    "tensor",
    "model",
    "neural",
    "cuda",
    "pytorch",
    "autograd",
    "dataloader",
    "dataset",
    "matmul",
    "logits",
    "relu",
    "softmax",
    "sigmoid",
    "backprop",
    "epoch",
    "optimizer",
)

# Identifiers from the instrument's retired corpus, checked on generated pages only. These
# are ordinary English words that appear innocently in stdlib source, where they carry no
# familiarity at all; an invented page must simply not stumble onto one.
_REPO_VOCAB = (
    "train_loop",
    "loss_fn",
    "loss",
    "pred",
    "batch",
    "flatten",
    "forward",
    "linear",
    "sequential",
    "toward_white",
    "tint",
    "palette",
    "channels",
    "agg",
    "snippet",
    "theme",
)

# Invented morphemes, drawn from joinery, landscape and bookbinding — plausible as code,
# and nowhere near the vocabulary of this repo or of anything he reads.
_QUALIFIERS = (
    "prime",
    "outer",
    "lateral",
    "median",
    "coastal",
    "tandem",
    "hollow",
    "quiet",
    "brittle",
    "distal",
    "vernal",
    "sable",
    "candid",
    "boreal",
    "civic",
    "molten",
    "granite",
    "amber",
    "ferrous",
    "tidal",
    "supine",
    "leeward",
    "russet",
    "opaline",
)

_NOUNS = (
    "ledger",
    "beacon",
    "harbor",
    "lantern",
    "quarry",
    "ripple",
    "thicket",
    "meadow",
    "cistern",
    "trellis",
    "furrow",
    "kestrel",
    "marmot",
    "juniper",
    "sextant",
    "bellows",
    "cobble",
    "shingle",
    "vellum",
    "spindle",
    "flume",
    "wicket",
    "plinth",
    "gantry",
    "coping",
    "louver",
    "mullion",
    "purlin",
    "corbel",
    "tessera",
    "quoin",
    "spandrel",
    "voussoir",
    "reglet",
    "chamfer",
    "escutcheon",
    "finial",
    "gudgeon",
    "haunch",
    "keeper",
)

_ROLE_SUFFIX = (
    "",
    "_index",
    "_table",
    "_span",
    "_walk",
    "_pass",
    "_ring",
    "_slate",
    "_frame",
    "_count",
    "_mark",
    "_seam",
    "_gate",
    "_lane",
    "_yard",
    "_stub",
)

_VERBS = (
    "gather",
    "sift",
    "braid",
    "winnow",
    "burnish",
    "unspool",
    "tally",
    "hoist",
    "temper",
    "sluice",
    "kindle",
    "furl",
    "graft",
    "mottle",
    "cleave",
    "stipple",
    "plait",
    "harrow",
    "scarf",
    "dovetail",
    "rabbet",
    "kerf",
    "swage",
    "anneal",
    "quench",
    "reeve",
    "flute",
    "chase",
)

# Comment prose, invented and deliberately dull: a sentence with a claim in it invites
# reading for the claim, which is not the thing being timed.
_PROSE = (
    "the keeper is rebuilt whenever the seam width changes",
    "callers hold the gate open until the last span is drained",
    "the count is advisory; the yard recomputes it on demand",
    "empty lanes are kept so the index stays dense",
    "order here follows the slate, not the input",
    "the table is small enough to copy rather than share",
    "a stale mark is cheaper to discard than to repair",
    "widths are stored as thirds to avoid rounding twice",
    "the walk stops at the first closed wicket",
    "trailing separators are tolerated for hand-written input",
    "this branch exists only for the single-entry case",
    "the ring is sized once and never grown",
)

# Short literals: keys, labels and separators that read as configuration.
_KEYS = (
    "width",
    "depth",
    "kind",
    "label",
    "origin",
    "extent",
    "offset",
    "weight",
    "flag",
    "shade",
    "grain",
    "pitch",
)

_TEXTS = (
    "loose",
    "seated",
    "half-lap",
    "mitered",
    "square",
    "raked",
    "flush",
    "banded",
    "sprung",
    "keyed",
    "coped",
    "dressed",
)

_SEPS = (", ", " | ", "; ", " / ", " - ")

# Names that cannot be a click task: everything already in the reader's fingers.
_NOT_A_TARGET = frozenset(dir(builtins)) | {"self", "cls", "args", "kwargs", "None", "True", "False"}

_INTS = ("0", "1", "2", "3", "4", "7", "8", "12", "16", "31", "64", "128")
_ALT_NUMS = ("0.5", "1.5", "2.25", "0.125", "6", "9", "11", "24", "48")
_THIRD_NUMS = ("5", "10", "13", "20", "36", "72")


def _role_spans(code):
    """Role spans via the stdlib tokenizer: text, role, and the token's line and column.

    Duplicated from the instrument's own cell rather than shared, because a marimo cell's
    underscore-prefixed helper is cell-local and cannot be imported. The semantics have to
    stay identical — the instrument colors and click-targets these spans — so `_render`
    below reconstructs the source out of them and `_finish` refuses any page whose
    reconstruction is not the source. That is the same walk the instrument's renderer
    performs, so it fails here rather than as mangled code on a card.
    """
    spans = []
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    prev_sig = None
    for i, tok in enumerate(toks):
        typ, txt, (sr, sc), (er, ec), _ = tok
        if typ in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        if typ == tokenize.COMMENT:
            role = "comment"
        elif typ == tokenize.STRING or typ in (
            getattr(tokenize, "FSTRING_START", -1),
            getattr(tokenize, "FSTRING_MIDDLE", -2),
            getattr(tokenize, "FSTRING_END", -3),
        ):
            role = "string"
        elif typ == tokenize.NUMBER:
            role = "number"
        elif typ == tokenize.OP:
            role = "punct"
        elif typ == tokenize.NAME:
            if keyword.iskeyword(txt):
                role = "keyword"
            elif prev_sig in ("def", "class"):
                role = "function"
            else:
                nxt = next((t2 for t2 in toks[i + 1 :] if t2.type not in (tokenize.NL, tokenize.NEWLINE)), None)
                role = "function" if (nxt is not None and nxt.string == "(") else "variable"
        else:
            role = "variable"
        spans.append({"text": txt, "role": role, "sr": sr, "sc": sc, "er": er, "ec": ec})
        if typ == tokenize.NAME or (typ == tokenize.OP and txt in "()[]{}.,:"):
            prev_sig = txt
    return spans


def _render(code, spans):
    """The instrument's renderer reduced to plain text: spans in order, gaps filled from
    the raw line. Comparing this against the source is the one check that proves a page
    will draw correctly — f-string and multi-line-string token positions are where a
    span-addressed renderer goes wrong, and it goes wrong silently."""
    lines = code.split("\n")
    out = []
    row, col = 0, 0
    for s in spans:
        r, c = s["sr"] - 1, s["sc"]
        while row < r:
            out.append("\n")
            row, col = row + 1, 0
        if c > col:
            out.append(lines[r][col:c])
        out.append(s["text"])
        row, col = s["er"] - 1, s["ec"]
    return "".join(out)


def _role_counts(spans):
    counts = dict.fromkeys((*PLANNED_ROLES, "punct"), 0)
    for s in spans:
        counts[s["role"]] = counts.get(s["role"], 0) + 1
    return counts


def _role_error(counts, plan):
    return sum(abs(counts.get(role, 0) - want) for role, want in plan.items())


def _forbidden_hit(code, generated):
    low = code.lower()
    for word in _FORBIDDEN:
        if word in low:
            return word
    if generated:
        for word in _REPO_VOCAB:
            if word in low:
                return word
    return None


def _widest(code):
    return max((len(line) for line in code.split("\n")), default=0)


def _nesting(code):
    """Block nesting, from the tokenizer's own indent stack rather than from leading
    spaces: a continuation line aligned under an open bracket sits 40 columns in without
    being nested at all, and counting columns called such a page eleven deep."""
    depth = level = 0
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.INDENT:
            level += 1
            depth = max(depth, level)
        elif tok.type == tokenize.DEDENT:
            level -= 1
    return depth


def _max_indent(code):
    """Leftmost column the page's text starts at, at its worst. A stimulus that is mostly
    whitespace is a different stimulus, however shallow its nesting."""
    body = [line for line in code.split("\n") if line.strip()]
    return max(((len(line) - len(line.lstrip())) for line in body), default=0)


# ---------------------------------------------------------------------------
# Invented names
# ---------------------------------------------------------------------------

_VAR_SPACE = len(_QUALIFIERS) * len(_NOUNS) * len(_ROLE_SUFFIX)
_FN_SPACE = len(_VERBS) * len(_NOUNS)

# A page's carrier name and its first call name come from one index into the product of
# the two name spaces, so seeds closer together than this cannot render the same page.
# Everything else about a page also varies with the seed; this is the part that is
# guaranteed rather than merely likely.
SEED_PERIOD = _VAR_SPACE * _FN_SPACE

# The seed is scattered across that product before being split, because the split alone is
# not enough: taken raw, trial 0 through trial 15 differ only in a name's suffix and every
# one of them calls `gather_ledger`, and a recurring name is the familiarity this module
# exists to remove. Coprime with SEED_PERIOD (= 2**15 * 3 * 5**2 * 7), so the scatter is a
# permutation and the guarantee above survives it.
_SCATTER = 2654435761


def _var_name(i):
    i %= _VAR_SPACE
    i, s = divmod(i, len(_ROLE_SUFFIX))
    i, n = divmod(i, len(_NOUNS))
    return f"{_QUALIFIERS[i % len(_QUALIFIERS)]}_{_NOUNS[n]}{_ROLE_SUFFIX[s]}"


def _fn_name(i):
    v, n = divmod(i % _FN_SPACE, len(_NOUNS))
    return f"{_VERBS[v]}_{_NOUNS[n]}"


class _NameBook:
    """Every name a page will use, drawn before the structure search runs.

    Fixing the names first is what makes that search safe: swapping one statement for
    another re-costs the role mix without renaming anything, so the page's click targets
    stay put across iterations. The first two names are a permutation of the seed, which is
    where the freshness guarantee lives; the rest are drawn without replacement so no page
    carries two identical identifiers by accident.
    """

    def __init__(self, seed, rng):
        self.rng = rng
        self._used = set()
        fi, vi = divmod(seed * _SCATTER % SEED_PERIOD, _VAR_SPACE)
        self.carrier = self._claim(_var_name(vi))
        self.first_call = self._claim(_fn_name(fi))
        self.entry = self._draw(_fn_name, _FN_SPACE)
        self.cls = "".join(p.capitalize() for p in self._draw(_var_name, _VAR_SPACE).split("_"))
        self._vars = [self._draw(_var_name, _VAR_SPACE) for _ in range(14)]
        self._fns = [self.first_call] + [self._draw(_fn_name, _FN_SPACE) for _ in range(9)]
        self._loops = rng.sample(("row", "item", "part", "node", "step", "edge"), 4)

    def _claim(self, name):
        self._used.add(name)
        return name

    def _draw(self, maker, space):
        """Drawn until unique. Whether the winner also avoids being a substring of another
        name is settled downstream, where the click target is picked: a name that contains
        the target disqualifies the target, not itself."""
        candidate = maker(self.rng.randrange(space))
        for _ in range(64):
            if candidate not in self._used:
                break
            candidate = maker(self.rng.randrange(space))
        return self._claim(candidate)

    def var(self, i):
        return self._vars[i % len(self._vars)]

    def fn(self, i):
        return self._fns[i % len(self._fns)]

    def loop(self, i):
        return self._loops[i % len(self._loops)]


class _SlotNames:
    """The literal material one statement slot may use, drawn once so that a template swap
    reuses it and the search explores structure alone."""

    def __init__(self, names, rng, i):
        self.var = names.var(i)
        self.var2 = names.var(i + 7)
        self.fn = names.fn(i)
        self.loop = names.loop(i)
        self.field = names.var(i + 3).rsplit("_", 1)[-1]
        self.key = rng.choice(_KEYS)
        self.key2 = rng.choice([k for k in _KEYS if k != self.key])
        self.text = rng.choice(_TEXTS)
        self.text2 = rng.choice([t for t in _TEXTS if t != self.text])
        self.label = rng.choice(_KEYS)
        self.sep = rng.choice(_SEPS)
        self.prose = rng.choice(_PROSE)
        self.num = rng.choice(_INTS)
        self.num2 = rng.choice(_ALT_NUMS)
        self.num3 = rng.choice(_THIRD_NUMS)


# ---------------------------------------------------------------------------
# The grammar: one-line statements, and the block shapes that hold them
# ---------------------------------------------------------------------------

# Every statement is exactly one line, so a swap during the search cannot change the page's
# height. A comment may sit anywhere a statement may, including the first line of a block
# body — what a block cannot survive is a body of comments *alone*, so each region keeps at
# least one real statement. Reads go through the carrier rather than a neighboring slot's
# name, so every name a page mentions is a name that page defines: an assert on an
# identifier that appears nowhere else sends the reader looking for it, on the clock.
_STATEMENTS = {
    "num": lambda s, c: f"{s.var} = {s.num}",
    "str": lambda s, c: f'{s.var} = "{s.text}"',
    "call": lambda s, c: f"{s.var} = {s.fn}({c})",
    "call_num": lambda s, c: f"{s.var} = {s.fn}({c}, {s.num})",
    "call_kw": lambda s, c: f"{s.var} = {s.fn}({c}, {s.key}={s.num})",
    "dict": lambda s, c: f'{s.var} = {{"{s.key}": {s.num}, "{s.key2}": "{s.text}"}}',
    "list": lambda s, c: f"{s.var} = [{s.num}, {s.num2}, {s.num3}]",
    "strlist": lambda s, c: f'{s.var} = ["{s.text}", "{s.text2}"]',
    "comp": lambda s, c: f"{s.var} = [{s.fn}({s.loop}) for {s.loop} in {c} if {s.loop} != {s.num}]",
    "dcomp": lambda s, c: f"{s.var} = {{{s.loop}: {s.fn}({s.loop}) for {s.loop} in {c}}}",
    "index": lambda s, c: f"{s.var} = {c}[{s.num}]",
    "add": lambda s, c: f"{s.var} = {c} + {s.num}",
    "is_none": lambda s, c: f"{s.var} = {c} is not None",
    "ternary": lambda s, c: f"{s.var} = {s.num} if {c} else {s.num2}",
    "assertion": lambda s, c: f"assert {c} != {s.num}",
    "join": lambda s, c: f'{s.var} = "{s.sep}".join({s.fn}({c}))',
    "unpack": lambda s, c: f"{s.var}, {s.var2} = {s.fn}({c})",
    "attr": lambda s, c: f"{s.var} = {c}.{s.field}",
    "fstring": lambda s, c: f'{s.var} = f"{s.label} {{{c}}}"',
    "lam": lambda s, c: f"{s.var} = lambda {s.loop}: {s.loop} * {s.num}",
    "contains": lambda s, c: f'{s.var} = "{s.key}" in {c}',
    "negate": lambda s, c: f"{s.var} = not {c}",
    "comment": lambda s, c: f"# {s.prose}",
}

_BODY_IDS = tuple(k for k in _STATEMENTS if k != "comment")


class _Slot:
    """A statement line waiting for a template: its indentation, and which of the shape's
    regions it belongs to — the region is what has to keep one real statement."""

    __slots__ = ("depth", "region")

    def __init__(self, depth, region):
        self.depth, self.region = depth, region


def _split(total, minimums, rng):
    """Deal `total` statement lines over the shape's regions, honoring each minimum. The
    minimums are what keeps every block body non-empty; the remainder is dealt at random
    so the target's line does not settle at a position the eye can learn."""
    counts = list(minimums)
    for _ in range(max(0, total - sum(minimums))):
        counts[rng.randrange(len(counts))] += 1
    return counts


def _slots(region, depth, count):
    return [_Slot(depth, region) for _ in range(count)]


def _shape_loop_branch(nb, n):
    """def, then for, then if."""
    pre, blk, inner, post = _split(n, (0, 1, 1, 0), nb.rng)
    return [
        f"def {nb.entry}({nb.var(0)}, {nb.var(1)}):",
        f'    """{nb.rng.choice(_PROSE).capitalize()}."""',
        f"    {nb.carrier} = {nb.first_call}({nb.var(0)})",
        *_slots("pre", 1, pre),
        f"    for {nb.loop(0)} in {nb.carrier}:",
        *_slots("block", 2, blk),
        f"        if {nb.loop(0)} != {nb.var(1)}:",
        *_slots("inner", 3, inner),
        *_slots("post", 1, post),
        f"    return {nb.carrier}",
    ]


def _shape_guarded(nb, n):
    """def, then try/except, with the loop inside the guarded body."""
    pre, blk, inner, post = _split(n, (0, 1, 1, 1), nb.rng)
    return [
        f"def {nb.entry}({nb.var(0)}, {nb.var(1)}):",
        f'    """{nb.rng.choice(_PROSE).capitalize()}."""',
        f"    {nb.carrier} = {nb.first_call}({nb.var(0)})",
        *_slots("pre", 1, pre),
        "    try:",
        *_slots("block", 2, blk),
        f"        for {nb.loop(1)} in {nb.carrier}:",
        *_slots("inner", 3, inner),
        "    except ValueError:",
        *_slots("post", 2, post),
        f"    return {nb.carrier}",
    ]


def _shape_context(nb, n):
    """def, then with, then for."""
    pre, blk, inner, post = _split(n, (0, 1, 1, 0), nb.rng)
    return [
        f"def {nb.entry}({nb.var(0)}, {nb.var(1)}):",
        f'    """{nb.rng.choice(_PROSE).capitalize()}."""',
        f"    {nb.carrier} = {nb.first_call}({nb.var(0)})",
        *_slots("pre", 1, pre),
        f"    with {nb.fn(2)}({nb.var(1)}) as {nb.var(4)}:",
        *_slots("block", 2, blk),
        f"        for {nb.loop(2)} in {nb.carrier}:",
        *_slots("inner", 3, inner),
        *_slots("post", 1, post),
        f"    return {nb.var(4)}",
    ]


def _shape_record(nb, n):
    """A class with annotated fields, one method, one loop."""
    pre, inner, post = _split(n, (0, 1, 0), nb.rng)
    return [
        f"class {nb.cls}:",
        f'    """{nb.rng.choice(_PROSE).capitalize()}."""',
        f"    {nb.var(5)}: int = {nb.rng.choice(('2', '8', '16', '64'))}",
        f'    {nb.var(6)}: str = "{nb.rng.choice(_TEXTS)}"',
        "",
        f"    def {nb.entry}(self, {nb.var(0)}):",
        f"        {nb.carrier} = {nb.first_call}({nb.var(0)})",
        *_slots("pre", 2, pre),
        f"        for {nb.loop(3)} in {nb.carrier}:",
        *_slots("inner", 3, inner),
        *_slots("post", 2, post),
        f"        return {nb.carrier}",
    ]


def _shape_while_branch(nb, n):
    """def, then while, then if."""
    pre, blk, inner, post = _split(n, (0, 1, 1, 0), nb.rng)
    return [
        f"def {nb.entry}({nb.var(0)}, {nb.var(1)}, {nb.var(3)}):",
        f'    """{nb.rng.choice(_PROSE).capitalize()}."""',
        f"    {nb.carrier} = {nb.first_call}({nb.var(0)})",
        *_slots("pre", 1, pre),
        f"    while {nb.carrier} and {nb.var(1)} > {nb.var(3)}:",
        *_slots("block", 2, blk),
        f"        if {nb.var(3)} in {nb.carrier}:",
        *_slots("inner", 3, inner),
        *_slots("post", 1, post),
        f"    return {nb.carrier}",
    ]


# Each shape reaches indentation depth three and states how many lines it spends on its own
# frame, so `lines` can be honored exactly whichever shape a seed draws.
_SHAPES = (
    ("loop-branch", _shape_loop_branch, 6, 2),
    ("guarded", _shape_guarded, 7, 3),
    ("context", _shape_context, 6, 2),
    ("record", _shape_record, 9, 1),
    ("while-branch", _shape_while_branch, 6, 2),
)


def _assemble(rows, picks, slot_names, carrier):
    out, k = [], 0
    for row in rows:
        if isinstance(row, _Slot):
            out.append("    " * row.depth + _STATEMENTS[picks[k]](slot_names[k], carrier))
            k += 1
        else:
            out.append(row)
    return "\n".join(out) + "\n"


def _generate(seed, n_lines, plan, attempt, max_width):
    """One procedural page, built to the plan by local search over statement choices.

    Search rather than arithmetic, because a statement's role vector is whatever the
    tokenizer says it is and not what a hand-written table claims: the page is
    re-tokenized after every swap, so the plan is met against the same counter the
    instrument's colors come from.
    """
    rng = random.Random((seed * _SCATTER + attempt * 40503) % (2**61 - 1))
    nb = _NameBook(seed, rng)
    shape_name, shape, frame, floor = _SHAPES[(seed // 7 + attempt) % len(_SHAPES)]
    rows = shape(nb, max(floor, n_lines - frame))
    slots = [r for r in rows if isinstance(r, _Slot)]
    slot_names = [_SlotNames(nb, rng, i) for i in range(len(slots))]
    # Comments are pinned rather than searched: they are the one role the search reliably
    # trades away (a comment line is a statement line spent on no other role), and comment
    # contrast is a theme axis in its own right, so the count has to be exact.
    spare = {}
    for s in slots:
        spare[s.region] = spare.get(s.region, 0) + 1
    pinned = set()
    for i in rng.sample(range(len(slots)), len(slots)):
        if len(pinned) >= plan.get("comment", 0):
            break
        if spare[slots[i].region] > 1:
            pinned.add(i)
            spare[slots[i].region] -= 1
    allowed = [("comment",) if i in pinned else _BODY_IDS for i in range(len(slots))]
    picks = [rng.choice(a) for a in allowed]

    def cost(candidate):
        code = _assemble(rows, candidate, slot_names, nb.carrier)
        if _widest(code) > max_width:
            return 10**6, code
        try:
            spans = _role_spans(code)
        except tokenize.TokenError, IndentationError, SyntaxError:
            return 10**6, code
        return _role_error(_role_counts(spans), plan), code

    # Coordinate descent, one slot fully explored at a time: a random-swap walk gets the
    # mean error low and leaves a tail of pages several tokens off the plan, and a tail is
    # exactly what a controlled stimulus cannot have.
    best, best_code = cost(picks)
    for _ in range(4):
        if best == 0:
            break
        opening = best
        for i in rng.sample(range(len(picks)), len(picks)):
            for option in allowed[i]:
                if option == picks[i]:
                    continue
                trial = list(picks)
                trial[i] = option
                err, code = cost(trial)
                if err < best:
                    picks, best, best_code = trial, err, code
            if best == 0:
                break
        if best == opening:
            break
    return best_code, f"generated: {shape_name} shape, {n_lines} lines, seed {seed}"


# ---------------------------------------------------------------------------
# The stdlib corpus
# ---------------------------------------------------------------------------

# Modules Titus is unlikely to have opened. The obvious candidates for that are the ones
# PEP 594 retired, so six of them (sunau, sndhdr, chunk, uu, cgitb, pipes) are simply gone
# before 3.14 and their same-spirit survivors stand in — the list is long because how many
# distinct blocks the corpus holds is exactly how long the freshness guarantee lasts.
STDLIB_MODULES = (
    "wave",
    "colorsys",
    "difflib",
    "tabnanny",
    "email.header",
    "email.quoprimime",
    "email.base64mime",
    "email.charset",
    "email.iterators",
    "mailbox",
    "imaplib",
    "quopri",
    "nturl2path",
    "netrc",
    "plistlib",
    "sched",
    "shelve",
    "stringprep",
    "codeop",
    "symtable",
    "pyclbr",
    "filecmp",
    "fileinput",
    "getopt",
    "cmd",
    "mimetypes",
    "poplib",
    "pickletools",
    "wsgiref.handlers",
    "wsgiref.headers",
    "modulefinder",
    "py_compile",
)


@lru_cache(maxsize=1)
def _module_sources():
    """The importable subset, with its source and parse tree.

    Guarded per module because which of these exist is a property of the running
    interpreter rather than of the list above: obscurity and removal have the same cause,
    so a seventh module could go in 3.15 without this file noticing.
    """
    out = []
    for name in STDLIB_MODULES:
        try:
            src = inspect.getsource(importlib.import_module(name))
            tree = ast.parse(src)
        except ImportError, OSError, SyntaxError, TypeError:
            continue
        out.append((name, src, tree))
    return tuple(out)


def _candidate_ranges(tree, n_lines):
    """Line ranges worth slicing: whole functions and classes near the requested length,
    and contiguous runs of the statements inside them.

    Statement boundaries rather than arbitrary line windows, because a block cut mid-`else`
    does not parse. A run that happens to carry a bare `return` does not either, and is
    caught by the compile downstream instead of by a rule here that would have to
    enumerate every such case.
    """
    lo, hi = n_lines - 2, n_lines + 2
    ranges = set()
    holders = [tree]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            holders.append(node)
            first = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            if lo <= node.end_lineno - first + 1 <= hi:
                ranges.add((first, node.end_lineno))
    for holder in holders:
        body = getattr(holder, "body", [])
        for i, head in enumerate(body):
            start = head.decorator_list[0].lineno if getattr(head, "decorator_list", None) else head.lineno
            for tail in body[i:]:
                span = tail.end_lineno - start + 1
                if span > hi:
                    break
                if span >= lo:
                    ranges.add((start, tail.end_lineno))
    return sorted(ranges)


@lru_cache(maxsize=4)
def _stdlib_blocks(n_lines, max_width):
    """Every extractable block, in a fixed order: module name, first line, code.

    Order is module order and then line order, so the corpus is the same list on every
    run: a seed names the same block when the instrument replays a logged trial as it did
    in the sitting that recorded it.
    """
    blocks = []
    for name, src, tree in _module_sources():
        lines = src.split("\n")
        for start, end in _candidate_ranges(tree, n_lines):
            raw = lines[start - 1 : end]
            if any(len(line) > EXTRACT_MAX_WIDTH for line in raw):
                continue
            code = textwrap.dedent("\n".join(raw)).rstrip() + "\n"
            if _widest(code) > max_width or code.startswith((" ", "\t")):
                continue
            if _forbidden_hit(code, generated=False):
                continue
            try:
                compile(code, "<stdlib-block>", "exec")
                block_tree = ast.parse(code)
            except SyntaxError, ValueError:
                continue
            # A run of bare statements out of the middle of a method is valid Python and
            # still the wrong stimulus: it starts mid-thought, and the reader spends the
            # clock reconstructing context instead of finding a name. A block has to read
            # as a definition, which is also what puts a function name on the page.
            if not any(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) for n in block_tree.body):
                continue
            blocks.append((name, start, code))
    return tuple(blocks)


@lru_cache(maxsize=4)
def _viable_blocks(n_lines, max_width):
    """Blocks that already pass every check a returned page must pass, deduplicated by
    content.

    Filtering here rather than retrying in `snippet` is what makes freshness hold: a retry
    loop walks a rejected block's neighbor, which is some other seed's block, and two
    seeds then share a page. Validity does not depend on the seed — only which name
    becomes the target does — so one probe settles it for all of them.
    """
    out, seen = [], set()
    for name, start, code in _stdlib_blocks(n_lines, max_width):
        if code in seen or _finish(code, "stdlib", 0, "", DEFAULT_ROLES, False, max_width) is None:
            continue
        seen.add(code)
        out.append((name, start, code, _role_counts(_role_spans(code))))
    return tuple(out)


@lru_cache(maxsize=8)
def _stdlib_pool(n_lines, plan_key, max_width):
    """The blocks a plan admits, kept in corpus order.

    A plan narrows the pool instead of reordering it, so the stride `snippet` walks it with
    stays a bijection: within any window of `len(pool)` consecutive seeds, no block
    repeats. The band is wide on purpose — real code cannot be held to a synthetic role
    mix, and this exists so a caller asking for comment-heavy pages gets them, not to make
    the stdlib arm match the procedural one, which it cannot and should not.
    """
    plan = dict(plan_key)
    band = sum(plan.values()) // 2 + 6
    blocks = _viable_blocks(n_lines, max_width)
    pool = [(name, start, code) for name, start, code, counts in blocks if _role_error(counts, plan) <= band]
    return tuple(pool) or tuple((name, start, code) for name, start, code, _c in blocks)


def _stride(m):
    """A stride coprime with the pool size, so successive seeds walk the whole pool before
    any block comes round a second time."""
    return next((s for s in (1597, 1201, 997, 811, 613, 419, 307, 211, 101, 37, 7) if gcd(s, m) == 1), 1)


# ---------------------------------------------------------------------------
# Assembly and the public entry point
# ---------------------------------------------------------------------------

# Content hashes issued this session, so the instrument's non-repetition assertion has a
# second witness. Observational only: resolving a collision here by perturbing the output
# would make a page depend on the order trials were generated in, and the instrument
# replays its log by regenerating each page from its seed.
_ISSUED = {}
_COLLISIONS = []


def _nameable(text, floor=4):
    """Whether a token can carry a click task. `self`, `s`, `b` and `tuple` all fail: a
    one-letter name is found by shape rather than read, and a builtin is already known by
    heart — neither times the thing the instrument is timing. The find hunt raises the
    floor, because its own loop variables (`part`, `item`, `node`) clear the lower one."""
    return len(text) >= floor and text not in _NOT_A_TARGET and not keyword.iskeyword(text)


def _weak_target_ids(spans):
    """Span indices that make a poor comprehension answer.

    A name on a `def` or `class` line sits at a fixed place near the top, and teaching its
    position is the confound this module removes. A name after a dot is a method — `join`,
    `write`, `startswith` — so it is vocabulary the reader already has, and it recurred on
    fifteen of two hundred generated pages, which is a corpus of four wearing a disguise.
    """
    out = set()
    for i, s in enumerate(spans):
        if s["role"] == "function" and i and spans[i - 1]["text"] in ("def", "class", "."):
            out.add(i)
    return out


def _finish(code, kind, seed, provenance, plan, generated, max_width=MAX_WIDTH, target_kind=None):
    """Validate a candidate page and derive its two click targets, or return None.

    The comprehension answer is a call name from the body, and the find-hunt identifier is
    the page's most repeated name, since that task highlights every occurrence and asks
    which one is current. Both are decided here rather than by the instrument's random
    pick over every function-role token, so "occurs exactly once" is a property of the
    trial instead of a hope about the page.
    """
    if not code.strip() or _widest(code) > max_width or _forbidden_hit(code, generated):
        return None
    if _max_indent(code) > MAX_INDENT:
        return None
    try:
        compile(code, f"<{kind}-{seed}>", "exec")
        spans = _role_spans(code)
    except SyntaxError, ValueError, tokenize.TokenError, IndentationError:
        return None
    if _render(code, spans) != code.rstrip("\n"):
        return None
    counts = _role_counts(spans)
    if sum(1 for n in counts.values() if n) < 3 or _nesting(code) > MAX_NESTING:
        return None

    weak = _weak_target_ids(spans)
    once = [
        i
        for i, s in enumerate(spans)
        if s["role"] == "function" and code.count(s["text"]) == 1 and _nameable(s["text"])
    ]
    # Preference, not requirement: a stdlib block whose only single-occurrence function
    # name is its own `def` is still a usable page, and refusing it would shrink the
    # corpus for the sake of a rule that only matters when there is a choice.
    candidates = [i for i in once if i not in weak] or once
    if not candidates:
        return None

    # A def-site target is a much easier find than a call-site one -- it sits at a line
    # start, at a predictable indent, and a page holds only one or two of them -- so
    # mixing the two kinds puts a large step in the task's difficulty and reaction time
    # measures which kind was drawn rather than how the theme reads (measured: 12 of 60
    # probe pages handed out a def-site target). Call sites are preferred, and the kind is
    # reported either way so a caller can require one and an analysis can check.
    def _is_def_site(_i):
        _line = code.split("\n")[spans[_i]["sr"] - 1]
        return _line.lstrip().startswith("def ") and f"def {spans[_i]['text']}" in _line

    _calls = [i for i in candidates if not _is_def_site(i)]
    _pool = _calls or candidates
    if target_kind == "call" and not _calls:
        return None
    if target_kind == "def":
        _defs = [i for i in candidates if _is_def_site(i)]
        if not _defs:
            return None
        _pool = _defs
    target_id = _pool[(seed // 3) % len(_pool)]
    target = spans[target_id]["text"]
    target_is_def = _is_def_site(target_id)

    # Longer wins a tie in occurrence count: the find hunt highlights every match and asks
    # which is current, so the useful stimulus is a name read rather than recognized.
    tally = {}
    for s in spans:
        if s["role"] in ("variable", "function") and s["text"] != target:
            tally[s["text"]] = tally.get(s["text"], 0) + 1
    ranked = sorted(
        ((n, len(t), t) for t, n in tally.items() if 2 <= n <= 8 and _nameable(t, floor=5)),
        reverse=True,
    )
    if not ranked:
        return None
    ident = ranked[0][2]

    digest = hashlib.blake2b(code.encode(), digest_size=6).hexdigest()
    body = code.rstrip("\n").split("\n")
    return {
        "id": f"{'gen' if kind == 'procedural' else 'std'}-{seed:05d}-{digest}",
        "provenance": provenance,
        "code": code,
        "spans": spans,
        # Exactly one span, so the instrument's `random.choice(fn_ids)` cannot land on an
        # ambiguous name: whatever it draws is the target below.
        "fn_ids": [target_id],
        "ident": ident,
        "ident_ids": [i for i, s in enumerate(spans) if s["text"] == ident],
        "target": target,
        "target_id": target_id,
        "target_kind": "def" if target_is_def else "call",
        "kind": kind,
        "seed": seed,
        "hash": digest,
        "role_counts": counts,
        "role_plan": dict(plan),
        "role_error": _role_error(counts, plan),
        "n_lines": len(body),
        "max_width": _widest(code),
        "depth": _nesting(code),
        "max_indent": _max_indent(code),
    }


def snippet(
    seed: int,
    kind: str | None = None,
    lines: int = DEFAULT_LINES,
    roles: dict | None = None,
    *,
    max_width: int = MAX_WIDTH,
    target_kind: str | None = None,
) -> dict:
    """One code page for one trial, determined entirely by `seed`.

    `kind` is "procedural", "stdlib", or None to let the seed choose — one stdlib page
    every third trial. `lines` is exact for procedural pages and within two of the request
    for stdlib ones, whose blocks are real code. `roles` overrides DEFAULT_ROLES:
    procedural pages are built to it, stdlib pages filtered toward it. `max_width` is the
    `target_kind` may require "call" or "def" for the comprehension target: a def-site
    target is found far faster than a call-site one, so a probe that wants to measure the
    theme rather than the draw asks for "call". Leaving it None still prefers call sites
    whenever the page offers a choice.

    column ceiling no line may cross; DUEL_WIDTH is the number to pass when two cards
    share the band, since the ceiling above is what one card can hold, not two.

    Returns a drop-in for the instrument's snippet record — `id`, `provenance`, `code`,
    `spans`, `fn_ids`, `ident`, `ident_ids` — plus `target` (the identifier `fn_ids` names,
    occurring exactly once in `code`), `target_id`, `hash`, `role_counts`, `role_plan`,
    `role_error`, `kind`, `seed`, and the page's measured `n_lines`, `max_width`, `depth`
    and `max_indent`. `fresh` is False when this session already issued this page under a
    different seed, which is how a wrapped corpus announces itself — see `corpus_stats`.

    Determinism is the whole contract: the instrument regenerates a logged trial from its
    seed, so nothing here may depend on what was generated before it.
    """
    if kind is None:
        kind = "stdlib" if seed % AUTO_STDLIB_EVERY == AUTO_STDLIB_EVERY - 1 else "procedural"
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    plan = dict(DEFAULT_ROLES if roles is None else roles)
    lines = max(8, int(lines))
    max_width = min(int(max_width), MAX_WIDTH)

    out = None
    if kind == "procedural":
        # Attempts change the shape as well as the random stream, because a plan can be
        # unreachable inside one shape's frame and comfortable inside another's — five
        # comments do not fit a class-shaped page at fourteen lines. The best attempt wins
        # rather than the first, so an unreachable plan degrades by a token or two instead
        # of by whichever shape the seed happened to draw.
        want_comments = plan.get("comment", 0)

        def rank(page):
            # Comments rank ahead of the total: a comment is a whole line spent, so a
            # shape that cannot fit the asked-for number is the wrong shape, however well
            # it does on the other five roles.
            return abs(page["role_counts"]["comment"] - want_comments), page["role_error"]

        for attempt in range(len(_SHAPES)):
            code, provenance = _generate(seed, lines, plan, attempt, max_width)
            page = _finish(code, kind, seed, provenance, plan, True, max_width, target_kind)
            if page is not None and (out is None or rank(page) < rank(out)):
                out = page
            if out is not None and rank(out) == (0, 0):
                break
    else:
        # One block, no retry: the pool is pre-filtered to blocks that cannot fail, so the
        # seed maps onto it bijectively and two seeds cannot be handed the same page.
        pool = _stdlib_pool(lines, tuple(sorted(plan.items())), max_width)
        name, start, code = pool[(seed * _stride(len(pool))) % len(pool)]
        end = start + len(code.rstrip("\n").split("\n")) - 1
        where = f"{name} L{start}-{end} (CPython stdlib)"
        out = _finish(code, kind, seed, where, plan, False, max_width, target_kind)
    if out is None and target_kind and kind == "stdlib":
        # A required target kind can rule out the one stdlib block a seed maps to, and a
        # raise would cost a trial. Fall back to the procedural generator for this seed:
        # the constraint is kept, the page is still unique, and only the mix between the
        # two generators shifts -- which the record's `kind` field already reports.
        return snippet(
            seed,
            kind="procedural",
            lines=lines,
            roles=roles,
            max_width=max_width,
            target_kind=target_kind,
        )
    if out is None:
        raise RuntimeError(f"no valid {kind} page for seed {seed} at {lines} lines")

    prior = _ISSUED.setdefault(out["hash"], seed)
    out["fresh"] = prior == seed
    if prior != seed:
        _COLLISIONS.append((prior, seed, out["hash"]))
    return out


def corpus_stats(
    sample: int = 128,
    lines: int = DEFAULT_LINES,
    roles: dict | None = None,
    *,
    max_width: int = MAX_WIDTH,
) -> dict:
    """What the corpus can supply, and what it does supply over `sample` seeds.

    Read this before trusting the freshness claim. The stdlib arm's guarantee is one block
    per seed across `stdlib_period` consecutive seeds — a number measured here rather than
    asserted anywhere in this file, since it depends on which modules the interpreter
    still ships.
    """
    plan = dict(DEFAULT_ROLES if roles is None else roles)
    pool = _stdlib_pool(lines, tuple(sorted(plan.items())), max_width)
    present = {name for name, _src, _tree in _module_sources()}
    per_module = {}
    for name, _start, _code in pool:
        per_module[name] = per_module.get(name, 0) + 1
    # Role error is reported per kind because averaging them says nothing: the procedural
    # arm is built to the plan and the stdlib arm cannot be, so one number would hide both.
    errors = {kind: [] for kind in KINDS}
    widths, depths, kinds = [], [], {}
    for seed in range(sample):
        page = snippet(seed, lines=lines, roles=plan, max_width=max_width)
        errors[page["kind"]].append(page["role_error"])
        widths.append(page["max_width"])
        depths.append(page["depth"])
        kinds[page["kind"]] = kinds.get(page["kind"], 0) + 1
    return {
        "kinds": KINDS,
        "stdlib_modules": sorted(present),
        "stdlib_missing": [m for m in STDLIB_MODULES if m not in present],
        "stdlib_blocks": len(_stdlib_blocks(lines, max_width)),
        "stdlib_viable": len(_viable_blocks(lines, max_width)),
        "stdlib_period": len(pool),
        "stdlib_per_module": dict(sorted(per_module.items())),
        "procedural_period": SEED_PERIOD,
        "role_plan": plan,
        "role_tolerance": ROLE_TOLERANCE,
        "sample": sample,
        "sampled_kinds": kinds,
        "role_error_max": {k: max(v) for k, v in errors.items() if v},
        "role_error_mean": {k: round(sum(v) / len(v), 2) for k, v in errors.items() if v},
        "max_width": max_width,
        "max_width_seen": max(widths),
        "depth_seen": (min(depths), max(depths)),
        "session_issued": len(_ISSUED),
        "session_collisions": list(_COLLISIONS),
    }
