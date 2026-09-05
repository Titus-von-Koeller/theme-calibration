"""Marked regions in a hand-edited Python file: the discipline every applier keeps.

An applier writes measured values into a file a person also edits, so it owns exactly the
lines between two marker comments -- `# >>> measured:<name>` and `# <<< measured:<name>`, the
same markers the dotfiles applier writes into settings.jsonc with `//` -- and touches nothing
outside them. Two rules, both earned there:

- The region is installed once, where the applier says, and replaced in place ever after, so
  the file's own ordering and the comments around the region survive every write.
- A region that owns names retires the hand-written definitions of those names on EVERY
  write, over the whole file, not only when it is installed. The dotfiles applier dropped
  superseded keys only at a region's birth; when the region's key set grew a day later the
  file held two values for one key, and the editor used whichever it read last.

Retirement works on the syntax tree, not on line prefixes: a top-level assignment to an owned
name is removed whole, however many lines its value spans.
"""

import ast
import re

BEGIN = "# >>> measured:{name}"
END = "# <<< measured:{name}"


def region_pattern(name: str) -> re.Pattern:
    begin, end = re.escape(BEGIN.format(name=name)), re.escape(END.format(name=name))
    return re.compile(rf"(?P<indent>[ \t]*){begin}[^\n]*\n(?P<body>.*?)(?P<tail>[ \t]*){end}[^\n]*\n?", re.DOTALL)


def has_region(text: str, name: str) -> bool:
    return region_pattern(name).search(text) is not None


def _region_line_span(text: str, name: str) -> tuple[int, int] | None:
    """1-based (first, last) line of the region's markers, or None when not installed."""
    match = region_pattern(name).search(text)
    if match is None:
        return None
    first = text.count("\n", 0, match.start()) + 1
    last = text.count("\n", 0, match.end() - 1) + 1
    return first, last


def _owned_assignments(text: str, owned: tuple[str, ...]) -> list[ast.Assign]:
    """Top-level assignments whose targets are all owned names."""
    return [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in owned for t in node.targets)
    ]


def retire_assignments(text: str, name: str, owned: tuple[str, ...]) -> tuple[str, int | None]:
    """Drop every hand-written assignment to an owned name outside the region.

    Returns the text and the 1-based line where the first retired assignment stood (in the
    returned text), which is where a region being installed belongs: the reader finds the
    measured values exactly where the hand-written ones were.
    """
    span = _region_line_span(text, name)
    doomed: set[int] = set()
    for node in _owned_assignments(text, owned):
        if span and span[0] <= node.lineno <= span[1]:
            continue
        doomed.update(range(node.lineno, node.end_lineno + 1))
    if not doomed:
        return text, None
    lines = text.splitlines(True)
    kept = [line for i, line in enumerate(lines, 1) if i not in doomed]
    return "".join(kept), min(doomed)


def splice(text: str, name: str, body: list[str], owned: tuple[str, ...]) -> str:
    """The region with `body` inside its markers, installed if absent, hand-written owned
    names retired wherever they stand."""
    text, first_retired = retire_assignments(text, name, owned)
    if not has_region(text, name):
        text = _install(text, name, at_line=first_retired)
    match = region_pattern(name).search(text)
    indent = match.group("indent")
    region = (
        f"{indent}{BEGIN.format(name=name)}\n"
        + "".join(f"{indent}{line}\n" if line else "\n" for line in body)
        + f"{indent}{END.format(name=name)}\n"
    )
    return text[: match.start()] + region + text[match.end() :]


def _install(text: str, name: str, at_line: int | None) -> str:
    """An empty region before `at_line` (1-based), or at the end of the file."""
    lines = text.splitlines(True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    empty = f"{BEGIN.format(name=name)}\n{END.format(name=name)}\n"
    if at_line is None:
        return "".join(lines) + ("\n" if lines and lines[-1].strip() else "") + empty
    index = at_line - 1
    return "".join(lines[:index]) + empty + "".join(lines[index:])
