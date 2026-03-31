"""
pivotal.errors — shared error types and display formatting.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PivotalError:
    """A structured error from the Pivotal parser or validator."""
    message: str
    error_type: str = "Error"          # "Error" | "Warning" | "Syntax Error" | "Validation Error"
    line: Optional[int] = None
    column: Optional[int] = None
    source_line: Optional[str] = None  # the raw source line (no line-number prefix)
    suggestion: Optional[str] = None   # "did you mean X?" or other hint


def _make_suggestion(name: str, candidates: list[str]) -> Optional[str]:
    """Return a 'did you mean X?' string if a close match exists, else None."""
    if not candidates:
        return None
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return None


def _pointer_line(column: int, length: int = 1) -> str:
    """Return a caret pointer string aligned to column (1-based)."""
    col0 = max(column - 1, 0)
    return " " * col0 + "^" * max(length, 1)


def format_error_text(err: PivotalError, source_code: str = "") -> str:
    """Format a PivotalError as plain text."""
    lines = source_code.splitlines() if source_code else []

    # Header: "Pivotal Error (line N): message"
    if err.line is not None:
        header = f"Pivotal {err.error_type} (line {err.line}): {err.message}"
    else:
        header = f"Pivotal {err.error_type}: {err.message}"

    parts = [header]

    if err.suggestion:
        parts.append(f"  -> {err.suggestion}")

    # Source snippet with pointer
    src_line = err.source_line
    if src_line is None and err.line is not None and 1 <= err.line <= len(lines):
        src_line = lines[err.line - 1]

    if src_line is not None and err.line is not None:
        parts.append("")
        parts.append(f"  {err.line} | {src_line}")
        if err.column is not None:
            # offset = "  N | " prefix width
            prefix_len = len(f"  {err.line} | ")
            parts.append(" " * prefix_len + _pointer_line(err.column))

    return "\n".join(parts)


def format_error_html(err: PivotalError, source_code: str = "") -> str:
    """Format a PivotalError as an HTML string for JupyterLab display."""
    import html as _html
    lines = source_code.splitlines() if source_code else []

    if err.line is not None:
        header = f"Pivotal {err.error_type} (line {err.line}): {_html.escape(err.message)}"
    else:
        header = f"Pivotal {err.error_type}: {_html.escape(err.message)}"

    parts = [
        '<div style="font-family:monospace;padding:8px 12px;border-left:3px solid #e05252;'
        'background:#fff5f5;border-radius:3px;margin:4px 0;line-height:1.6">',
        f'<span style="color:#c0392b;font-weight:bold">{header}</span>',
    ]

    if err.suggestion:
        parts.append(
            f'<br><span style="color:#555">&nbsp;&nbsp;\u2192 {_html.escape(err.suggestion)}</span>'
        )

    src_line = err.source_line
    if src_line is None and err.line is not None and 1 <= err.line <= len(lines):
        src_line = lines[err.line - 1]

    if src_line is not None and err.line is not None:
        prefix = f"  {err.line} | "
        escaped_src = _html.escape(src_line)
        parts.append(f'<br><code style="color:#333">{_html.escape(prefix)}{escaped_src}</code>')
        if err.column is not None:
            pointer = " " * len(prefix) + _pointer_line(err.column)
            parts.append(f'<br><code style="color:#e05252">{_html.escape(pointer)}</code>')

    parts.append("</div>")
    return "".join(parts)


def display_error(err: PivotalError, source_code: str = "") -> None:
    """Display a PivotalError — HTML in JupyterLab, plain text elsewhere."""
    try:
        from IPython.display import display, HTML
        import IPython
        ip = IPython.get_ipython()
        if ip is not None:
            display(HTML(format_error_html(err, source_code)))
            return
    except Exception:
        pass
    print(format_error_text(err, source_code))
