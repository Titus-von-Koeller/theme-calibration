"""The stimulus layer: the code pages a trial shows, and their HTML.

Two jobs, and they meet only at `render_card`:

- choosing the page. `snippet_for` asks the generator in `codegen` for a fresh page at the
  width and length a trial wants, relaxing the request in a declared order when the
  generator cannot supply that shape, and falling back to a familiarity CONTROL page as a
  last resort.
- drawing it. `render_card` turns a page plus a candidate theme into the HTML for one card,
  in one of the three arrangements `SURFACES` names.

Token roles are not decided here. `tokenize_roles` is re-exported from `codegen`, which
owns the one role table both the generator and this renderer read; a second copy would
drift, and the drift would show up as a miscoloured or unclickable token rather than as a
failure.
"""

import html

from . import codegen
from .codegen import role_spans as tokenize_roles

#: The column ceiling for two cards sharing the band, re-exported so a caller arranging a
#: duel does not have to know the generator owns the number.
DUEL_WIDTH = codegen.DUEL_WIDTH

__all__ = [
    "CONTROL",
    "DUEL_WIDTH",
    "OUTPUT_TAIL",
    "PROSE",
    "PROSE_TAIL",
    "READING_PX",
    "SOURCES",
    "SURFACES",
    "render_card",
    "snippet_for",
    "tokenize_roles",
]

# Stimuli that are real code, embedded verbatim: code representative of the corpus under
# study, not lorem ipsum. Embedded rather than read at render time so the stimulus set is
# stable across sessions; each record carries the snippet id.
SOURCES = {
    "train-loop": (
        "07-optimization-loop.py",
        """def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = (loss.item(), batch * 64 + len(X))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
""",
        "loss",
    ),
    "build-model": (
        "05-build-model.py",
        """class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
""",
        "nn",
    ),
    "tint": (
        "_palette.py",
        '''def tint(color, toward_white):
    """The palette hue mixed toward a card's white, as a literal hex.

    For renderers that cannot take scheme names (graphviz, raw CSS):
    fills stay derived from the constants above instead of hand-tuned
    hexes appearing per notebook.
    """
    channels = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * toward_white):02x}" for c in channels)
''',
        "color",
    ),
    "tensor-ops": (
        "02-tensors.py",
        """y2 = tensor_2.matmul(tensor_2.T)
torch.matmul(tensor_2, tensor_2.T, out=y3)
z1 = tensor_2 * tensor_2
z2 = tensor_2.mul(tensor_2)
# This computes the element-wise product; z1, z2 will match
torch.mul(tensor_2, tensor_2, out=z3)
t1 = torch.cat([tensor_2, tensor_2, tensor_2], dim=1)
agg = tensor_2.sum()
agg_item = agg.item()
""",
        "tensor_2",
    ),
}


def _control_pages():
    """The four embedded sources as snippet records.

    One page per trial, never the same twice: `codegen` writes it. These four stay as the
    cold-start corpus and as a familiarity CONTROL -- a page the reader knows is the
    reference against which a fresh page's reaction time is read -- but they are no longer
    the corpus.

    `fresh` is False on every one of them, and that is the point: a control page has been
    seen before by construction, so a trial that lands on one is answered partly from
    memory. The response log reads `fresh` to decide whether a reaction time is a find
    time at all, and it defaults the flag to True when absent, so a record that omitted it
    would be silently counted as a first showing.
    """
    pages = []
    for snippet_id, (provenance, code, ident) in SOURCES.items():
        spans = tokenize_roles(code)
        pages.append(
            {
                "id": snippet_id,
                "provenance": provenance,
                "code": code,
                "spans": spans,
                "fn_ids": [i for i, span in enumerate(spans) if span["role"] == "function"],
                "ident": ident,
                "ident_ids": [i for i, span in enumerate(spans) if span["text"] == ident],
                "hash": f"control-{snippet_id}",
                "kind": "control",
                "fresh": False,
            }
        )
    return pages


# Built at import, below the function that builds it: the order is load-bearing under a
# reactive notebook runtime, where a name referenced inside an exported function resolves
# only if it is defined above that function. A plain script run shares one namespace and
# never mangles, so the wrong order passes every check here and fails only when served.
CONTROL = _control_pages()

#: Pages already chosen this session, keyed by the full request. The instrument, the
#: recorder and the analysis all resolve the same trial to the same page without
#: regenerating it -- which matters because generating one costs hundreds of milliseconds.
SNIP_MEMO = {}

# Width and length are PREFERENCES; freshness is the requirement. The generator cannot
# promise every shape for every seed -- a narrow 28-line page is a tall order, and asking
# for 64 columns alone lost half the seeds -- so the request relaxes in this declared
# order: hold the narrow column and shorten, then widen a step and shorten again.
_LINE_RELAXATIONS = (0, -4, -8)
_WIDTH_RELAXATIONS = (0, 8, 16)


def _shape_ladder(width, lines):
    """The (max_width, lines) requests to try, in the order they are given up.

    Width is the outer loop, so every length is tried at the narrow column before the
    column is widened at all: a page that overflows its card is clipped by the card's
    `overflow:hidden`, and a clipped stimulus is a different stimulus, whereas a page four
    lines short is the same stimulus with less of it.

    A None in either position means "do not constrain this", which hands the generator its
    own default -- MAX_WIDTH for the column, DEFAULT_LINES for the length. The unbounded
    rungs are last for that reason: they are the widest ONE card can hold, not the widest
    this card can hold, so reaching them is how a duel ends up with a page wider than the
    64 columns two cards share.

    Measured over 120 duel seeds at (width 64, lines 28): the ladder relaxes the column on
    35 of them and the widest page returned is 72, so it reaches the +8 rung but not the
    unbounded one, and 72 is still inside the ~80 columns a duel card holds at 14px --
    nothing was clipped. The exposure is latent rather than active: nothing here caps the
    unbounded rung at what the card can show, so a seed that fell through to it could
    return 100 columns and be clipped in silence. The length relaxes far more often -- 83
    of the 120 came back short of 28, and 32 of those at the generator's own 14 -- and
    that is by design, since a short page is the same stimulus with less of it.

    None of this reaches the response log: `max_width` and `n_lines` are on the record
    that `snippet_for` returns, but theme/responses.py does not write them, so an analysis
    cannot currently separate a 14-line duel page from a 28-line one.
    """
    widths = [int(width) + step for step in _WIDTH_RELAXATIONS] + [None] if width else [None]
    line_counts = [int(lines) + step for step in _LINE_RELAXATIONS] + [None] if lines else [None]
    return [(w, n) for w in widths for n in line_counts]


def snippet_for(seed, width=None, target_kind=None, lines=None):
    """The page for this trial seed: fresh procedural or obscure-stdlib code.

    width is the column ceiling: two duel cards side by side hold about eighty columns
    at 14px, and the stimulus <pre> is overflow:hidden, so a wider line would be
    silently clipped -- a clipped stimulus is a different stimulus. Line count and role
    mix stay at the generator's calibrated default: freshness alone makes the
    comprehension probe hard now that no page is ever shown twice, and a longer page
    would trade away the identical-role-statistics property that lets two reaction
    times be compared at all.

    Neither preference is a guarantee. The returned record carries the page's measured
    `max_width` and `n_lines`, and a caller that needs the request honoured exactly must
    check them; the ladder relaxes rather than raising, because a raise costs a trial.
    """
    request = (int(seed), width, target_kind, lines)
    if request in SNIP_MEMO:
        return SNIP_MEMO[request]
    page = None
    for max_width, line_count in _shape_ladder(width, lines):
        wanted = {}
        if max_width:
            wanted["max_width"] = max_width
        if line_count:
            wanted["lines"] = line_count
        if target_kind:
            wanted["target_kind"] = target_kind
        try:
            page = dict(codegen.snippet(int(seed), **wanted))
        except ValueError, RuntimeError:
            # The two ways the generator declines: an unusable request (ValueError) and no
            # page it can build for this seed at this shape (RuntimeError). Anything else
            # is a bug in the generator and must not be relaxed away -- a swallowed
            # TypeError would degrade every trial to a control page with nothing said.
            continue
        page.setdefault("ident", page.get("target"))
        break
    if page is None:
        # Every shape failed. A control page is code the reader has already seen, which
        # turns a find task into a memory test, so it is the last resort and it says so:
        # `fresh` is False on it. Copied because the record is handed out to callers that
        # annotate it, and CONTROL must stay the pristine reference.
        page = dict(CONTROL[int(seed) % len(CONTROL)])
    SNIP_MEMO[request] = page
    return page


PROSE_TAIL = (
    "The consumer holds the lock only while it copies out, so a slow reader delays the "
    "next fill rather than corrupting the one in flight."
)
OUTPUT_TAIL = "queue depth 3  drained 1284  blocked 0.4%  last fill 2.1 ms"
PROSE = (
    "A buffer is filled once per frame and drained by the consumer thread; the queue "
    "length bounds how far the two can drift apart before a reader blocks."
)

# The three surfaces the corpus under study is read on. A theme is one theme, but it is
# *seen* in three arrangements, and the one that wins on a bare code page need not win
# where prose and code interleave. Surface is a stimulus factor, not a theme axis: utility
# stays defined over the theme, and the surface is logged so a later analysis can test for
# a surface-by-theme interaction rather than assuming there is none.
#
#   editor    a page of code with a line of prose above it -- the plain editor
#   panel     the assistant chat surface: serif turns, a raised code card between
#             them, the proportions of an assistant answer
#   notebook  the marimo/VSCode notebook: a centred prose column at the measured 42rem
#             reading measure, then a raised code card, then an output block
SURFACES = ("editor", "panel", "notebook")

# And the size each one is ACTUALLY read at, from the reader's own editor settings: the
# global editor.fontSize is unset so ordinary editors sit at VSCode's default 14, notebook
# code cells are customised to 16, and the chat panel's code follows the editor at 14.
#
# This matters more than it looks. Duels ran at 12 and 13px on the reasoning that a full
# screen wants small type -- but 12 and 13 are sizes this reader never reads code at, so a
# preference measured there was being applied to reading at 14 and 16. Contrast
# sensitivity falls with glyph scale, which is exactly why the colour floors are doubled
# against the 104px threshold, so "measure at one size, apply at another" is not a free
# assumption in a colour experiment. Each surface is now shown at its true size, which
# also stops size and surface from being independently varied for no reason: in a real
# working day they covary, and it is the real pairing whose theme is wanted.
READING_PX = {"editor": 14, "panel": 14, "notebook": 16}

# Machine text sits on a raised card a step off the page, the grammar the applied theme
# uses: flat tinted panel means aside, raised card means artifact. The editor surface has
# no card -- a page of code IS the page.
_RAISED_SURFACES = ("panel", "notebook")

_SERIF = "'IBM Plex Serif',serif"
_MONO = "'IosevkaLigated Nerd Font Mono',monospace"

# Layout of the prose block above the code, per surface: reading measure, then margin. The
# notebook centres its column at the measured 42rem; the other surfaces run flush left.
_PROSE_LAYOUT = {
    "editor": ("34em", "margin:0 0 14px 0"),
    "panel": ("34em", "margin:0 0 14px 0"),
    "notebook": ("42rem", "margin:0 auto 14px auto"),
}

# The step off the page, in the ground's own hue and never toward grey: lighter on a dark
# ground, darker on a light one, with the border taking a larger step than the fill. As
# (fill, edge) channel deltas.
_CARD_STEP_ON_DARK = (12, 26)
_CARD_STEP_ON_LIGHT = (-10, -22)
_CARD_SHADOW = "0 1px 3px -1px rgba(0,0,0,.35), 0 5px 14px -6px rgba(0,0,0,.28)"

# Sum of the ground's channels below which it counts as dark. 384 is mid-grey summed over
# three channels, so the test is "darker than 50% grey" without a luminance conversion --
# the card step is a nudge, not a measured contrast, and the floors that ARE measured live
# in the colour layer.
_DARK_GROUND_SUM = 384


def _channels(hex_color):
    """The three 8-bit channels of a `#rrggbb` colour."""
    digits = hex_color.lstrip("#")
    return [int(digits[i : i + 2], 16) for i in (0, 2, 4)]


def _shifted(channels, delta):
    """`channels` moved by `delta`, clamped to the byte range, back as `#rrggbb`."""
    return "#" + "".join(f"{max(0, min(255, value + delta)):02x}" for value in channels)


def _mixed(ground, other, toward_other):
    """`ground` blended `toward_other` of the way to `other`, as `#rrggbb`."""
    base, tint = _channels(ground), _channels(other)
    return "#" + "".join(f"{round(base[i] * (1 - toward_other) + tint[i] * toward_other):02x}" for i in range(3))


def _raised_card(theme):
    """The (opening, closing) tags of a card raised one step off the ground."""
    channels = _channels(theme["ground"])
    dark = sum(channels) < _DARK_GROUND_SUM
    fill_step, edge_step = _CARD_STEP_ON_DARK if dark else _CARD_STEP_ON_LIGHT
    fill = _shifted(channels, fill_step)
    edge = _shifted(channels, edge_step)
    return (
        f'<div style="background:{fill};border:1px solid {edge};border-radius:4px;'
        f'padding:12px 14px;box-shadow:{_CARD_SHADOW};overflow:hidden">'
    ), "</div>"


def _serif_block(ink, text, measure, margin):
    """One block of prose: the reading typeface at 17px, held to `measure`."""
    return (
        f'<div style="font-family:{_SERIF};font-size:17px;line-height:1.6;'
        f'color:{ink};max-width:{measure};{margin}">{html.escape(text)}</div>'
    )


def _span_style(theme, span, span_id, find_current, highlighted):
    """The inline style for one token: its role colour, plus the find layer if it is part
    of one. Colour lookup only -- no markup, so the theme's role table is read in exactly
    one place."""
    style = f"color:{theme[span['role']]}"
    if span["role"] == "comment":
        style += ";font-style:italic"
    if span_id in highlighted:
        fill = theme["find_current"] if span_id == find_current else theme["find_other"]
        style += f";background:{fill};border-radius:2px"
    return style


def _code_spans(theme, snippet, find_current, task):
    """The page's tokens as coloured spans, with the whitespace between them.

    Walks the spans in order and fills each gap from the raw line, which is the same walk
    the generator's own round-trip check performs -- so a page whose span offsets do not
    address its source has already been refused upstream rather than reaching a card as
    mangled code. Gap text is escaped along with token text: it is whitespace on every page
    the generator will pass, and escaping it costs nothing on those, but an unescaped gap
    would be a hole in a renderer that emits HTML.
    """
    source_lines = snippet["code"].split("\n")
    highlighted = set(snippet["ident_ids"]) if find_current is not None else set()
    parts = []
    row, col = 0, 0
    for span_id, span in enumerate(snippet["spans"]):
        span_row, span_col = span["sr"] - 1, span["sc"]
        while row < span_row:
            parts.append("\n")
            row, col = row + 1, 0
        if span_col > col:
            parts.append(html.escape(source_lines[span_row][col:span_col]))
        style = _span_style(theme, span, span_id, find_current, highlighted)
        target = f' data-tid="{span_id}"' if task else ""
        parts.append(f'<span style="{style}"{target}>{html.escape(span["text"])}</span>')
        row, col = span["er"] - 1, span["ec"]
    return "".join(parts)


def _code_block(theme, snippet, code_px, find_current, task):
    """The stimulus itself: the page of code, at the true editor pixel size."""
    return (
        f'<pre style="font-family:{_MONO};font-size:{code_px}px;'
        f'line-height:1.5;margin:0;white-space:pre;overflow:hidden;color:{theme["punct"]}">'
        f"{_code_spans(theme, snippet, find_current, task)}"
        "</pre>"
    )


# The diff shown on the chat surface. Both backgrounds are DERIVED, not searched: the theme
# already carries a cool role colour and a warm one, and mixing each into the ground keeps
# added/removed on the cool/warm polarity that survives colour-vision deficiency while
# adding no dimension to a nine-dimensional space that is already the binding constraint on
# convergence. Line text stays the code ink -- a diff recolours the field, never the code.
_DIFF_MIX = 0.16
_DIFF_LINES = (
    ("-", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table)", "removed"),
    ("+", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table, 12)", "added"),
    (" ", "    with sift_gantry(opaline_voussoir_walk) as vernal_cistern_gate:", None),
    ("+", "        prime_mullion_stub = 128", "added"),
)


def _diff_block(theme, code_px, card_open, card_close):
    """A second card holding a diff, because an assistant turn is mostly diffs and their
    colours are part of what gets read all day."""
    fields = {
        "added": _mixed(theme["ground"], theme["function"], _DIFF_MIX),
        "removed": _mixed(theme["ground"], theme["string"], _DIFF_MIX),
        None: None,
    }
    rows = []
    for mark, text, field in _DIFF_LINES:
        style = f"display:block;padding:0 6px;color:{theme['punct']}"
        if fields[field]:
            style += f";background:{fields[field]}"
        rows.append(
            f'<span style="{style}"><span style="color:{theme["comment"]}">{mark}</span>{html.escape(text)}</span>'
        )
    return (
        f'{card_open}<div style="font-family:{_SERIF};font-size:13px;'
        f'color:{theme["comment"]};margin:0 0 6px 0">edited codegen.py</div>'
        f'<pre style="font-family:{_MONO};'
        f"font-size:{code_px}px;line-height:1.5;margin:0;white-space:pre;"
        f'overflow:hidden">' + "".join(rows) + f"</pre>{card_close}"
    )


def _output_block(theme, code_px):
    """A notebook cell is code plus its output, so the output block is part of the
    stimulus: mono, one step of ink below the code, on the page rather than the card."""
    return (
        f'<pre style="font-family:{_MONO};'
        f"font-size:{code_px}px;line-height:1.5;margin:8px 0 0 0;"
        f'color:{theme["comment"]};white-space:pre;overflow:hidden">'
        f"{html.escape(OUTPUT_TAIL)}</pre>"
    )


def render_card(theme, snippet, code_px, find_current=None, task=False, prose=True, surface="editor"):
    """One candidate page as HTML: prose in IBM Plex Serif 17px, code in Iosevka at the
    true editor pixel size, on the candidate ground. find_current=None hides the find
    layer; an int marks that occurrence as the current match, the rest as plain
    highlights. task=True makes every span a click target (data-tid), visually inert.
    surface selects the arrangement (see SURFACES above).

    The three arrangements differ only in what surrounds the code block, so this reads as
    the arrangement rather than as a branch per property: an unknown surface falls back to
    the editor's layout, which is the bare page.
    """
    card_open, card_close = _raised_card(theme) if surface in _RAISED_SURFACES else ("", "")
    blocks = []
    if prose:
        measure, margin = _PROSE_LAYOUT.get(surface, _PROSE_LAYOUT["editor"])
        blocks.append(_serif_block(theme["ink"], PROSE, measure, margin))
    blocks.append(card_open + _code_block(theme, snippet, code_px, find_current, task) + card_close)
    if surface == "panel":
        blocks.append(_diff_block(theme, code_px, card_open, card_close))
        if prose:
            # An assistant turn continues after the code: the second serif block is what
            # makes this the chat surface rather than a card on a page.
            blocks.append(_serif_block(theme["ink"], PROSE_TAIL, "34em", "margin:12px 0 0 0"))
    elif surface == "notebook":
        blocks.append(_output_block(theme, code_px))
    return "".join(blocks)
