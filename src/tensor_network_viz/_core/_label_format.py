"""Mini-markup for tensor node names -> Matplotlib mathtext (before drawing)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

_MATTEXT_ROMAN_ESCAPE_RE = re.compile(r"([#%&_{}~^\\$ ])")


def _escape_for_mathtext_roman(s: str) -> str:
    """Escape text placed inside ``\\mathrm{...}`` for matplotlib mathtext."""

    def repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        if ch == " ":
            return r"\ "
        return "\\" + ch

    return _MATTEXT_ROMAN_ESCAPE_RE.sub(repl, s)


def _wrap_roman(s: str) -> str:
    if not s:
        return ""
    return r"\mathrm{" + _escape_for_mathtext_roman(s) + "}"


def _find_balanced(s: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """Index *after* closing bracket, or *None* if unbalanced. *start* is index of *open_ch*."""
    depth = 0
    i = start
    n = len(s)
    while i < n:
        c = s[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _parse_script_atom(s: str, i: int) -> tuple[str | None, int]:
    """Parse one subscript/superscript atom; *i* is the first index after ``_`` or ``^``."""
    n = len(s)
    if i >= n:
        return None, i
    c = s[i]
    if c == "(":
        end = _find_balanced(s, i, "(", ")")
        if end is None:
            return None, i
        return s[i + 1 : end - 1], end
    if c == "{":
        end = _find_balanced(s, i, "{", "}")
        if end is None:
            return None, i
        return s[i + 1 : end - 1], end
    if c.isalnum():
        j = i + 1
        while j < n and s[j].isalnum():
            j += 1
        return s[i:j], j
    return None, i


def _inner_to_mathtext(s: str) -> str:
    """Build mathtext for a substring that may contain scripts (no outer $)."""
    i = 0
    n = len(s)
    parts: list[str] = []
    while i < n:
        if s[i] not in "^_":
            start = i
            while i < n and s[i] not in "^_":
                i += 1
            chunk = s[start:i]
            if chunk:
                parts.append(_wrap_roman(chunk))
            continue
        op = s[i]
        idx_after_op = i + 1
        atoms: list[str] = []
        j = idx_after_op
        while True:
            raw_inner, j2 = _parse_script_atom(s, j)
            if raw_inner is None:
                break
            atoms.append(_inner_to_mathtext(raw_inner))
            j = j2
            if j < n and s[j] == op:
                j += 1
                continue
            break
        if not atoms:
            parts.append("\\" + s[i])
            i += 1
            continue
        merged = r"\ ".join(atoms)
        parts.append(f"{op}{{{merged}}}")
        i = j
    return "".join(parts)


def _plain_segment_needs_mathtext(seg: str) -> bool:
    """True if *seg* needs mathtext (sub/sup markup or ``$``, special in mpl plain text)."""
    if "$" in seg:
        return True
    n = len(seg)
    i = 0
    while i < n:
        if seg[i] not in "^_":
            i += 1
            continue
        idx_after = i + 1
        atom, _ = _parse_script_atom(seg, idx_after)
        if atom is not None:
            return True
        i += 1
    return False


def _plain_segment_to_display(seg: str) -> str:
    if not seg:
        return seg
    if not _plain_segment_needs_mathtext(seg):
        return seg
    return "$" + _inner_to_mathtext(seg) + "$"


def _iter_dollar_runs(
    s: str,
) -> list[tuple[Literal["plain", "math"], str]] | None:
    buf: list[str] = []
    math = False
    i = 0
    n = len(s)
    runs: list[tuple[Literal["plain", "math"], str]] = []
    while i < n:
        if s[i] == "$":
            if i + 1 < n and s[i + 1] == "$":
                buf.append("$")
                i += 2
                continue
            kind: Literal["plain", "math"] = "math" if math else "plain"
            runs.append((kind, "".join(buf)))
            buf = []
            math = not math
            i += 1
            continue
        buf.append(s[i])
        i += 1
    runs.append(("math" if math else "plain", "".join(buf)))
    if math:
        return None
    return runs


@lru_cache(maxsize=4096)
def format_tensor_node_label(text: str) -> str:
    """
    Format a tensor node name or bond / index edge label for Matplotlib (mini-markup + ``$...$``).

    Plain segments (outside ``$...$``):

    - ``x_a``: subscript ``a`` (TeX-style; multi-letter runs group as one atom).
    - ``x_(ab)``: subscript the parenthesized text.
    - ``x_{ab}``: explicit braced subscript.
    - ``x_a_b``: two consecutive subscripts merged with a visible space.
    - ``x^a``, ``x^(ab)``, ``x^a^b``: superscripts, with the same grouping rules.
    - A ``_`` or ``^`` not followed by a valid atom stays literal.

    Dollar runs toggle mathtext: content between single ``$`` pairs is passed through
    unchanged (your LaTeX / mathtext). A literal ``$`` in plain text is written as ``$$``.
    An odd number of unescaped ``$`` leaves the whole string in plain mini-markup mode.
    """
    if not text:
        return text
    runs = _iter_dollar_runs(text)
    if runs is None:
        return _plain_segment_to_display(text)
    if len(runs) == 1 and runs[0][0] == "plain":
        return _plain_segment_to_display(runs[0][1])
    parts: list[str] = []
    for kind, seg in runs:
        if kind == "math":
            if seg:
                parts.append("$" + seg + "$")
        else:
            parts.append(_plain_segment_to_display(seg))
    return "".join(parts)
