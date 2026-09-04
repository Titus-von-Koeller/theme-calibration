"""The stimulus layer: the code pages a trial shows, and their HTML.

Extracted verbatim from the notebook this instrument used to be; cell-local names lost
their leading underscore.
"""

import html
import io
import keyword
import tokenize

from . import codegen

# Stimuli are real code, embedded verbatim from this repo's own notebooks (07's
# training loop, 05's model, _palette's tint) — the code Titus actually reads, not
# lorem ipsum. Embedded rather than read at render time so the stimulus set is stable
# across sessions; each record carries the snippet id.
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


def tokenize_roles(code):
    """Role spans via the stdlib tokenizer: (text, role, line, col). Definition and
    call names are `function`; control words `keyword`; strings and numbers are the
    one literal family; dotted-name reads and everything else recede as variable/punct."""
    _spans = []
    _toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    _prev_sig = None
    for _i, _tok in enumerate(_toks):
        _typ, _txt, (_sr, _sc), (_er, _ec), _ = _tok
        if _typ in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        if _typ == tokenize.COMMENT:
            _role = "comment"
        elif _typ == tokenize.STRING or _typ in (
            getattr(tokenize, "FSTRING_START", -1),
            getattr(tokenize, "FSTRING_MIDDLE", -2),
            getattr(tokenize, "FSTRING_END", -3),
        ):
            _role = "string"
        elif _typ == tokenize.NUMBER:
            _role = "number"
        elif _typ == tokenize.OP:
            _role = "punct"
        elif _typ == tokenize.NAME:
            if keyword.iskeyword(_txt):
                _role = "keyword"
            elif _prev_sig in ("def", "class"):
                _role = "function"
            else:
                _nxt = next(
                    (_t2 for _t2 in _toks[_i + 1 :] if _t2.type not in (tokenize.NL, tokenize.NEWLINE)),
                    None,
                )
                _role = "function" if (_nxt is not None and _nxt.string == "(") else "variable"
        else:
            _role = "variable"
        _spans.append({"text": _txt, "role": _role, "sr": _sr, "sc": _sc, "er": _er, "ec": _ec})
        if _typ == tokenize.NAME or (_typ == tokenize.OP and _txt in "()[]{}.,:"):
            _prev_sig = _txt
    return _spans


# One page per trial, never the same twice: _codegen writes it. The four embedded
# sources above stay as the cold-start and as a familiarity CONTROL -- a page he knows
# is the reference against which a fresh page's reaction time is read -- but they are
# no longer the corpus. Memoized per seed so the widget, the recorder and the analysis
# cell all resolve the same page without regenerating it.
# CONTROL is built BEFORE snippet_for on purpose: a cell-local name referenced
# inside an exported function resolves only if it is defined above that function,
# and only under `marimo run`/`edit` -- a script run shares one namespace and
# never mangles, so the wrong order passes every check and fails only when served.
CONTROL = []
for sid, (prov, code, ident) in SOURCES.items():
    sp = tokenize_roles(code)
    CONTROL.append(
        {
            "id": sid,
            "provenance": prov,
            "code": code,
            "spans": sp,
            "fn_ids": [_i for _i, _s in enumerate(sp) if _s["role"] == "function"],
            "ident": ident,
            "ident_ids": [_i for _i, _s in enumerate(sp) if _s["text"] == ident],
            "hash": f"control-{sid}",
            "kind": "control",
            # Seen before by construction. `theme.responses` defaults a missing `fresh`
            # to True, so omitting it here logged a reused page as a first showing -- and
            # a remembered page turns a find task into a memory test.
            "fresh": False,
        }
    )

# The loop's names were cell-local in the notebook and are throwaways here: dropped so
# `code`, `ident` and the rest are not part of this module's surface.
del sid, prov, code, ident, sp

SNIP_MEMO = {}


def snippet_for(seed, width=None, target_kind=None, lines=None):
    """The page for this trial seed: fresh procedural or obscure-stdlib code.

    width is the column ceiling: two duel cards side by side hold about eighty columns
    at 14px, and the stimulus <pre> is overflow:hidden, so a wider line would be
    silently clipped -- a clipped stimulus is a different stimulus. Line count and role
    mix stay at the generator's calibrated default: freshness alone makes the
    comprehension probe hard now that no page is ever shown twice, and a longer page
    would trade away the identical-role-statistics property that lets two reaction
    times be compared at all.
    """
    _key = (int(seed), width, target_kind, lines)
    if _key in SNIP_MEMO:
        return SNIP_MEMO[_key]
    # Width and length are PREFERENCES; freshness is the requirement. The generator
    # cannot promise every shape for every seed -- a narrow 28-line page is a tall
    # order, and asking for 64 columns alone lost half the seeds -- so the request
    # relaxes in a declared order: hold the narrow column and shorten, then widen a
    # step and shorten again, and only if every combination fails fall back to a
    # control page, which is code he has already seen and therefore the last resort.
    _lines_ladder = [int(lines), int(lines) - 4, int(lines) - 8, None] if lines else [None]
    _width_ladder = [int(width), int(width) + 8, int(width) + 16, None] if width else [None]
    _s = None
    for _w in _width_ladder:
        for _ln in _lines_ladder:
            try:
                _kw = {}
                if _w:
                    _kw["max_width"] = _w
                if _ln:
                    _kw["lines"] = _ln
                if target_kind:
                    _kw["target_kind"] = target_kind
                _s = dict(codegen.snippet(int(seed), **_kw))
                _s.setdefault("ident", _s.get("target"))
                break
            except ValueError, RuntimeError:
                # The two ways the generator declines: an unusable request (ValueError)
                # and no page it can build for this seed at this shape (RuntimeError).
                # Anything else is a bug in the generator and must not be relaxed away.
                continue
        if _s is not None:
            break
    if _s is None:
        # Copied: the record is handed to callers that annotate it, and CONTROL has to
        # stay the pristine reference for every later trial that falls back to it.
        _s = dict(CONTROL[int(seed) % len(CONTROL)])
    SNIP_MEMO[_key] = _s
    return _s


PROSE_TAIL = (
    "The consumer holds the lock only while it copies out, so a slow reader delays the "
    "next fill rather than corrupting the one in flight."
)
OUTPUT_TAIL = "queue depth 3  drained 1284  blocked 0.4%  last fill 2.1 ms"
PROSE = (
    "A buffer is filled once per frame and drained by the consumer thread; the queue "
    "length bounds how far the two can drift apart before a reader blocks."
)

# The three surfaces Titus actually reads. A theme is one theme, but it is *seen* in
# three arrangements, and the one that wins on a bare code page need not win where
# prose and code interleave. Surface is a stimulus factor, not a theme axis: utility
# stays defined over the theme, and the surface is logged so a later analysis can test
# for a surface-by-theme interaction rather than assuming there is none.
#
#   editor    a page of code with a line of prose above it -- the plain editor
#   panel     the Claude Code chat surface: serif turns, a raised code card between
#             them, the proportions of an assistant answer
#   notebook  the marimo/VSCode notebook: a centred prose column at the measured 42rem
#             reading measure, then a raised code card, then an output block
SURFACES = ("editor", "panel", "notebook")

# And the size he ACTUALLY reads each one at, from his own settings.jsonc: the global
# editor.fontSize is unset so ordinary editors sit at VSCode's default 14, notebook code
# cells are customised to 16, and the chat panel's code follows the editor at 14.
#
# This matters more than it looks. Duels ran at 12 and 13px on the reasoning that a full
# screen wants small type -- but 12 and 13 are sizes he never reads code at, so a
# preference measured there was being applied to reading at 14 and 16. Contrast
# sensitivity falls with glyph scale, which is exactly why the colour floors are doubled
# against the 104px threshold, so "measure at one size, apply at another" is not a free
# assumption in a colour experiment. Each surface is now shown at its true size, which
# also stops size and surface from being independently varied for no reason: in his real
# day they covary, and it is the real pairing whose theme is wanted.
READING_PX = {"editor": 14, "panel": 14, "notebook": 16}


def render_card(theme, snippet, code_px, find_current=None, task=False, prose=True, surface="editor"):
    """One candidate page as HTML: prose in IBM Plex Serif 17px, code in Iosevka at the
    true editor pixel size, on the candidate ground. find_current=None hides the find
    layer; an int marks that occurrence as the current match, the rest as plain
    highlights. task=True makes every span a click target (data-tid), visually inert.
    surface selects the arrangement (see SURFACES above)."""
    _lines = snippet["code"].split("\n")
    _cursor = {}
    _out = []
    _card_open, _card_close = "", ""
    if surface in ("panel", "notebook"):
        # Machine text sits on a raised card a step off the page, the grammar the
        # applied theme uses: flat tinted panel means aside, raised card means
        # artifact. The step is taken in the ground's own hue, never toward grey.
        _g = theme["ground"].lstrip("#")
        _rgb = [int(_g[_k : _k + 2], 16) for _k in (0, 2, 4)]
        _dark = sum(_rgb) < 384
        _step = 12 if _dark else -10
        _card_bg = "#" + "".join(f"{max(0, min(255, _v + _step)):02x}" for _v in _rgb)
        _edge = "#" + "".join(f"{max(0, min(255, _v + (26 if _dark else -22))):02x}" for _v in _rgb)
        _shadow = "0 1px 3px -1px rgba(0,0,0,.35), 0 5px 14px -6px rgba(0,0,0,.28)"
        _card_open = (
            f'<div style="background:{_card_bg};border:1px solid {_edge};border-radius:4px;'
            f'padding:12px 14px;box-shadow:{_shadow};overflow:hidden">'
        )
        _card_close = "</div>"
    if prose:
        _measure = "42rem" if surface == "notebook" else "34em"
        _centre = "margin:0 auto 14px auto" if surface == "notebook" else "margin:0 0 14px 0"
        _out.append(
            f"<div style=\"font-family:'IBM Plex Serif',serif;font-size:17px;line-height:1.6;"
            f'color:{theme["ink"]};max-width:{_measure};{_centre}">{html.escape(PROSE)}</div>'
        )
    _out.append(_card_open)
    _out.append(
        f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;font-size:{code_px}px;"
        f'line-height:1.5;margin:0;white-space:pre;overflow:hidden;color:{theme["punct"]}">'
    )
    _find_ids = set(snippet["ident_ids"]) if find_current is not None else set()
    for _i, _s in enumerate(snippet["spans"]):
        _r, _c = _s["sr"] - 1, _s["sc"]
        _pr, _pc = _cursor.get("r", 0), _cursor.get("c", 0)
        while _pr < _r:
            _out.append("\n")
            _pr, _pc = _pr + 1, 0
        if _c > _pc:
            # Escaped like the token text: whitespace on every page the generator will
            # pass, so this changes no byte there, and a renderer that emits HTML should
            # not have a path that does not escape.
            _out.append(html.escape(_lines[_r][_pc:_c]))
        _style = f"color:{theme[_s['role']]}"
        if _s["role"] == "comment":
            _style += ";font-style:italic"
        if _i in _find_ids:
            _fill = theme["find_current"] if _i == find_current else theme["find_other"]
            _style += f";background:{_fill};border-radius:2px"
        _tid = f' data-tid="{_i}"' if task else ""
        _out.append(f'<span style="{_style}"{_tid}>{html.escape(_s["text"])}</span>')
        _cursor = {"r": _s["er"] - 1, "c": _s["ec"]}
    _out.append("</pre>")
    _out.append(_card_close)
    if surface == "panel":
        # The diff card, because a Claude Code turn is mostly diffs and their colours
        # are part of what he reads all day. Both backgrounds are DERIVED, not searched:
        # the theme already carries a cool role colour and a warm one, and mixing each
        # into the ground keeps added/removed on the cool/warm polarity that survives
        # colour-vision deficiency while adding no dimension to a nine-dimensional
        # space that is already the binding constraint on convergence. Line text stays
        # the code ink -- a diff recolours the field, never the code.
        def _mix(_hex, _t):
            _a = theme["ground"].lstrip("#")
            _b = _hex.lstrip("#")
            return "#" + "".join(
                f"{round(int(_a[_k : _k + 2], 16) * (1 - _t) + int(_b[_k : _k + 2], 16) * _t):02x}" for _k in (0, 2, 4)
            )

        _add_bg, _del_bg = _mix(theme["function"], 0.16), _mix(theme["string"], 0.16)
        _sign = theme["comment"]
        _diff = [
            ("-", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table)", _del_bg),
            ("+", "    ferrous_voussoir_mark = stipple_plinth(ferrous_bellows_table, 12)", _add_bg),
            (" ", "    with sift_gantry(opaline_voussoir_walk) as vernal_cistern_gate:", None),
            ("+", "        prime_mullion_stub = 128", _add_bg),
        ]
        _rows = []
        for _mark, _text, _bg in _diff:
            _style = f"display:block;padding:0 6px;color:{theme['punct']}"
            if _bg:
                _style += f";background:{_bg}"
            _rows.append(
                f'<span style="{_style}"><span style="color:{_sign}">{_mark}</span>{html.escape(_text)}</span>'
            )
        _out.append(
            f"{_card_open}<div style=\"font-family:'IBM Plex Serif',serif;font-size:13px;"
            f'color:{theme["comment"]};margin:0 0 6px 0">edited codegen.py</div>'
            f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;"
            f"font-size:{code_px}px;line-height:1.5;margin:0;white-space:pre;"
            f'overflow:hidden">' + "".join(_rows) + f"</pre>{_card_close}"
        )
    if surface == "panel" and prose:
        # An assistant turn continues after the code: the second serif block is what
        # makes this the chat surface rather than a card on a page.
        _out.append(
            f"<div style=\"font-family:'IBM Plex Serif',serif;font-size:17px;line-height:1.6;"
            f'color:{theme["ink"]};max-width:34em;margin:12px 0 0 0">'
            f"{html.escape(PROSE_TAIL)}</div>"
        )
    if surface == "notebook":
        # A notebook cell is code plus its output, so the output block is part of the
        # stimulus: mono, one step of ink below the code, on the page rather than the card.
        _out.append(
            f"<pre style=\"font-family:'IosevkaLigated Nerd Font Mono',monospace;"
            f"font-size:{code_px}px;line-height:1.5;margin:8px 0 0 0;"
            f'color:{theme["comment"]};white-space:pre;overflow:hidden">'
            f"{html.escape(OUTPUT_TAIL)}</pre>"
        )
    return "".join(_out)


DUEL_WIDTH = codegen.DUEL_WIDTH
