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


# ---------------------------------------------------------------------------
# Runtime error translation (Phase 4)
# ---------------------------------------------------------------------------

def _translate_runtime_error(exc: BaseException) -> Optional[PivotalError]:
    """
    Translate a known Python backend exception into a PivotalError.
    Returns None if the exception is not a recognised pattern — the caller
    should then display the original traceback unchanged.
    """
    import re
    etype = type(exc)
    msg = str(exc)

    # KeyError — usually a missing column in Pandas/Polars
    if etype is KeyError:
        col = msg.strip("'\"[] ")
        return PivotalError(
            message=f"Column '{col}' not found - check your column names",
            error_type="Runtime Error",
        )

    # NameError — usually a table that wasn't loaded
    if etype is NameError:
        m = re.search(r"name '(\w+)' is not defined", msg)
        if m:
            name = m.group(1)
            return PivotalError(
                message=f"'{name}' is not defined - was the table loaded in a previous cell?",
                error_type="Runtime Error",
            )

    # FileNotFoundError — bad path in a load statement
    if etype is FileNotFoundError:
        path = getattr(exc, 'filename', None) or msg
        return PivotalError(
            message=f"File not found: {path}",
            error_type="Runtime Error",
            suggestion="Check the file path and make sure the file exists",
        )

    # TypeError — type mismatch in filter/assign expressions
    if etype is TypeError:
        if 'not supported between' in msg or 'unsupported operand' in msg:
            return PivotalError(
                message="Type mismatch in expression - check the column types",
                error_type="Runtime Error",
                suggestion=msg,
            )

    # ValueError from Pandas — often a bad cast or merge issue
    if etype is ValueError:
        if 'You are trying to merge' in msg or 'merge' in msg.lower():
            return PivotalError(
                message="Merge failed - check that the join key columns have compatible types",
                error_type="Runtime Error",
                suggestion=msg,
            )

    return None


def _is_eval_error(tb_text: str) -> bool:
    """Return True if the traceback suggests the error came from a Pandas eval expression."""
    eval_markers = ('DataFrame.eval', 'df.eval', 'numexpr', '.eval(', 'eval_expr')
    return any(m in tb_text for m in eval_markers)


def _format_traceback_html(tb_text: str) -> str:
    """Wrap a traceback string in a collapsible <details> block."""
    import html as _html
    return (
        '<details style="margin-top:6px">'
        '<summary style="cursor:pointer;color:#888;font-size:0.9em">'
        'Show full traceback</summary>'
        '<pre style="background:#f8f8f8;padding:8px;margin-top:4px;'
        'font-size:0.82em;overflow-x:auto;color:#555;border-radius:3px">'
        f'{_html.escape(tb_text)}'
        '</pre></details>'
    )


def display_error_with_traceback(err: PivotalError, source_code: str,
                                 tb_text: str) -> None:
    """Display a PivotalError with an expandable traceback section."""
    try:
        from IPython.display import display, HTML
        import IPython
        ip = IPython.get_ipython()
        if ip is not None:
            html_body = format_error_html(err, source_code)
            # Inject the collapsible traceback before the closing </div>
            html_body = html_body[:-6] + _format_traceback_html(tb_text) + '</div>'
            display(HTML(html_body))
            return
    except Exception:
        pass
    # Plain text fallback
    print(format_error_text(err, source_code))
    print('\nFull traceback:')
    print(tb_text)


def run_cell_with_error_filter(shell, combined: str, source_code: str) -> object:
    """
    Run *combined* Python code via IPython's run_cell, intercepting tracebacks.

    - Known error patterns (KeyError, NameError, etc.) → friendly message only.
    - Unrecognised errors → friendly generic message with expandable traceback.

    Returns the IPython ExecutionResult.
    """
    import sys
    import traceback as _tb

    _captured_exc_info = [None]
    _captured_tb_text = [None]
    original_showtb = shell.showtraceback

    def _capturing_showtb(*args, **kwargs):
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            _captured_exc_info[0] = exc_info
            _captured_tb_text[0] = ''.join(_tb.format_exception(*exc_info))

    shell.showtraceback = _capturing_showtb
    try:
        result = shell.run_cell(combined)
    finally:
        shell.showtraceback = original_showtb

    if result.error_in_exec is not None:
        friendly = _translate_runtime_error(result.error_in_exec)
        tb_text = _captured_tb_text[0] or ''

        if friendly:
            # Recognised pattern — clean message, no traceback needed.
            display_error(friendly, source_code)
        else:
            # Unrecognised — show a friendly wrapper with expandable traceback.
            if tb_text and _is_eval_error(tb_text):
                msg = "Error in assign expression - check column names and expression syntax"
                suggestion = str(result.error_in_exec)
            else:
                msg = "An error occurred while executing your Pivotal script"
                suggestion = str(result.error_in_exec)

            err = PivotalError(
                message=msg,
                error_type="Runtime Error",
                suggestion=suggestion if suggestion else None,
            )
            display_error_with_traceback(err, source_code, tb_text)

    return result
