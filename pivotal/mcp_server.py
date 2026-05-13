"""MCP server exposing Pivotal verification and comparison tools."""
from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .dsl_parser import DSLParser


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SYNTAX_PATH = _REPO_ROOT / "PIVOTAL.md"
_PUBLIC_DOCS_BASE_URL = "https://nealbob.github.io/pivotal-py"
_FULL_SYNTAX_MAX_CHARS = 25000

_DOC_FILES = (
    "PIVOTAL.md",
    "README.md",
    "docs/syntax/index.md",
    "docs/syntax/command-reference.md",
    "docs/syntax/data-quality.md",
    "docs/syntax/data-sources.md",
    "docs/syntax/filtering.md",
    "docs/syntax/functions.md",
    "docs/syntax/grouping.md",
    "docs/syntax/joining.md",
    "docs/syntax/missing-data.md",
    "docs/syntax/output.md",
    "docs/syntax/pipeline-control.md",
    "docs/syntax/python-interop.md",
    "docs/syntax/reshaping.md",
    "docs/syntax/saving.md",
    "docs/syntax/selection.md",
    "docs/syntax/sorting.md",
    "docs/syntax/transformation.md",
    "docs/syntax/values.md",
    "docs/syntax/window-functions.md",
    "docs/jupyter.md",
)

_TOPIC_ALIASES = {
    "melt": ("unpivot", "reshaping", "pivot"),
    "reshape": ("reshaping", "pivot", "unpivot"),
    "reshaping": ("reshaping", "pivot", "unpivot"),
    "wide": ("pivot", "reshaping"),
    "long": ("unpivot", "melt", "reshaping"),
    "spread": ("pivot", "reshaping"),
    "gather": ("unpivot", "melt", "reshaping"),
    "wavg": ("wmean", "weighted mean", "aggregation"),
    "weighted average": ("wmean", "weighted mean", "aggregation"),
    "values": ("scalar", "dict", "list", "config"),
    "scalar": ("scalar", "values", "config"),
    "dict": ("dict", "dictionary", "config", "values"),
    "dictionary": ("dict", "dictionary", "config", "values"),
    "config": ("dict", "scalar", "list", "values"),
    "loops": ("for", "function", "assert", "check"),
    "pipeline control": ("for", "function", "assert", "check"),
}

_CORE_GUIDANCE = (
    "Pivotal is an indented, pipeline-oriented DSL for data transformation. "
    "It compiles to Python by default using pandas DataFrames, and can also "
    "target polars, duckdb, or SQL. Core conventions: `with table` sets the "
    "active DataFrame; `with source as name` creates a derived copy and makes "
    "it active; indented statements mutate the active table in place. `load`, "
    "`save`, and `delete` are standalone top-level commands. Table and column "
    "names are bare identifiers; strings use quotes; Python runtime variables "
    "and callables use `:` such as `:threshold` or `:clean_func(col)`. "
    "Canonical example:\n"
    "with sales\n"
    "    filter amount > 100\n"
    "    revenue = price * quantity\n"
    "\n"
    "with sales as report\n"
    "    group by region\n"
    "        agg sum revenue as total_revenue\n"
    "    sort total_revenue desc\n"
)

_VERIFICATION_GUIDANCE = (
    "Before giving Pivotal code to a user, compile it with pivotal_compile at "
    "minimum to catch syntax and code-generation errors. When input data or "
    "expected outputs are available, prefer pivotal_run or pivotal_compare for "
    "stronger verification. For polished presentation, use pivotal_highlight "
    "after the code compiles successfully. "
)

_HIGHLIGHT_CSS = """
.pvt-code-block {
  position: relative;
  margin: 0;
}
.pvt-code-block .pvt-code {
  box-sizing: border-box;
  margin: 0;
  padding: 1rem 5.5rem 1rem 1rem;
  overflow: auto;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  background: #f8fafc;
  color: #111827;
  font: 0.875rem/1.6 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  white-space: pre;
}
.pvt-copy-button {
  position: absolute;
  top: 0.625rem;
  right: 0.625rem;
  z-index: 1;
  border: 1px solid #cbd5e1;
  border-radius: 0.375rem;
  padding: 0.35rem 0.65rem;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  font: 0.75rem/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.pvt-copy-button:hover {
  background: #f1f5f9;
}
.pvt-copy-button.pvt-copy-success {
  border-color: #0f766e;
  background: #ccfbf1;
  color: #0f766e;
}
.pvt-keyword { color: #7c3aed; font-weight: 600; }
.pvt-clause { color: #2563eb; font-weight: 500; }
.pvt-builtin { color: #0f766e; }
.pvt-variable { color: #9333ea; }
.pvt-name { color: #111827; }
.pvt-string { color: #b45309; }
.pvt-number { color: #0369a1; }
.pvt-operator { color: #be123c; }
.pvt-punctuation { color: #64748b; }
.pvt-comment { color: #64748b; font-style: italic; }
.pvt-constant { color: #0891b2; }
""".strip()

_COPY_BUTTON_SCRIPT = """
(function(button) {
  var block = button.closest('.pvt-code-block');
  var code = block && block.querySelector('.pvt-code');
  var text = code ? code.innerText : '';
  function markCopied() {
    var original = button.getAttribute('data-label') || button.textContent;
    button.setAttribute('data-label', original);
    button.textContent = 'Copied!';
    button.classList.add('pvt-copy-success');
    window.setTimeout(function() {
      button.textContent = original;
      button.classList.remove('pvt-copy-success');
    }, 2000);
  }
  function fallbackCopy() {
    var textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    markCopied();
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(markCopied, fallbackCopy);
  } else {
    fallbackCopy();
  }
})(this);
""".strip()


def _render_highlighted_code_block(highlighted_html: str, *, include_css: bool) -> str:
    style = f"<style>\n{_HIGHLIGHT_CSS}\n</style>\n" if include_css else ""
    return (
        f"{style}"
        '<div class="pvt-code-block">'
        '<button class="pvt-copy-button" type="button" '
        f'onclick="{html.escape(_COPY_BUTTON_SCRIPT, quote=True)}">Copy</button>'
        f'<pre class="pvt-code">{highlighted_html}</pre>'
        "</div>"
    )


def _load_input_files(input_files: Optional[Mapping[str, str]]) -> dict[str, Any]:
    import pandas as pd

    inputs: dict[str, Any] = {}
    for name, path_text in (input_files or {}).items():
        if not name.isidentifier():
            raise ValueError(f"Invalid input table name: {name}")
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"Input file not found: {path}")

        lower = path.name.lower()
        if lower.endswith(".csv"):
            inputs[name] = pd.read_csv(path)
        elif lower.endswith(".parquet"):
            inputs[name] = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input file type for {path}; use CSV or Parquet")
    return inputs


def _read_text_file(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def _trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        max_chars = 1
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "\n\n[truncated]", True


def _relative_doc_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _public_doc_url(relative_path: str) -> str:
    if relative_path == "PIVOTAL.md":
        return f"{_PUBLIC_DOCS_BASE_URL}/syntax/command-reference/"
    if relative_path == "README.md":
        return _PUBLIC_DOCS_BASE_URL
    if relative_path.startswith("docs/") and relative_path.endswith(".md"):
        page = relative_path.removeprefix("docs/").removesuffix(".md")
        if page.endswith("/index"):
            page = page.removesuffix("/index")
        return f"{_PUBLIC_DOCS_BASE_URL}/{page}/"
    return _PUBLIC_DOCS_BASE_URL


def _iter_doc_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for relative in _DOC_FILES:
        path = (_REPO_ROOT / relative).resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _resolve_doc_path(path_text: str) -> Optional[Path]:
    normalized = path_text.replace("\\", "/").strip().lstrip("/")
    for path in _iter_doc_paths():
        relative = _relative_doc_path(path)
        if normalized in {relative, path.name, relative.removeprefix("docs/")}:
            return path
    return None


def _slugify_heading(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text.lower())
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def _syntax_heading_level(line: str) -> Optional[int]:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes and len(stripped) > hashes and stripped[hashes] == " ":
        return hashes
    return None


def _extract_heading_text(line: str) -> str:
    level = _syntax_heading_level(line)
    if level is None:
        return ""
    return line.lstrip()[level + 1:].strip()


def _document_sections(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        level = _syntax_heading_level(line)
        if level is not None:
            headings.append((idx, level, _extract_heading_text(line)))

    sections: list[dict[str, Any]] = []
    for pos, (start, level, heading) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _ in headings[pos + 1:]:
            if next_level <= level:
                end = next_start
                break
        body = "\n".join(lines[start:end])
        sections.append({
            "heading": heading,
            "level": level,
            "start_line": start + 1,
            "end_line": end,
            "content": body,
        })
    return sections


def _topic_terms(topic: str) -> list[str]:
    topic_lower = topic.lower().strip()
    terms = [topic_lower]
    terms.extend(_TOPIC_ALIASES.get(topic_lower, ()))
    return list(dict.fromkeys(term.lower() for term in terms if term))


def _term_count(text: str, term: str) -> int:
    if re.fullmatch(r"[a-z0-9_]+", term):
        return len(re.findall(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text))
    return text.count(term)


def _section_score(section: Mapping[str, Any], terms: Sequence[str]) -> int:
    heading = str(section["heading"]).lower()
    content = str(section["content"]).lower()
    score = 0
    matched = False
    for term in terms:
        if term == heading or term == _slugify_heading(heading):
            score += 120
            matched = True
        heading_count = _term_count(heading, term)
        content_count = _term_count(content, term)
        if heading_count:
            score += 80
            matched = True
        if content_count:
            score += 10 + content_count
            matched = True
    if matched:
        score += max(0, 6 - int(section["level"]))
    return score


def _best_matching_sections(
    docs: Sequence[Path],
    topic: str,
    *,
    max_results: int = 5,
) -> tuple[list[dict[str, Any]], list[str]]:
    terms = _topic_terms(topic)
    matches: list[dict[str, Any]] = []
    for path in docs:
        relative = _relative_doc_path(path)
        for section in _document_sections(path):
            # The H1 section usually wraps the whole page; prefer specific child sections.
            if int(section["level"]) == 1:
                heading_lower = str(section["heading"]).lower()
                if not any(_term_count(heading_lower, term) for term in terms):
                    continue
            score = _section_score(section, terms)
            if score <= 0:
                continue
            matches.append({
                "score": score,
                "source": str(path),
                "path": relative,
                "url": _public_doc_url(relative),
                "heading": section["heading"],
                "level": section["level"],
                "start_line": section["start_line"],
                "end_line": section["end_line"],
                "content": section["content"],
            })

    matches.sort(
        key=lambda item: (
            item["score"],
            -int(item["level"]),
            -(item["end_line"] - item["start_line"]),
        ),
        reverse=True,
    )
    return matches[:max_results], terms


def get_pivotal_docs_index() -> dict[str, Any]:
    """Return available Pivotal documentation pages and headings."""
    documents: list[dict[str, Any]] = []
    for path in _iter_doc_paths():
        relative = _relative_doc_path(path)
        sections = _document_sections(path)
        title = sections[0]["heading"] if sections else path.stem
        documents.append({
            "path": relative,
            "source": str(path),
            "url": _public_doc_url(relative),
            "title": title,
            "headings": [
                {
                    "level": section["level"],
                    "heading": section["heading"],
                    "line": section["start_line"],
                }
                for section in sections
            ],
        })
    return {"ok": True, "documents": documents}


def get_pivotal_docs(
    topic: Optional[str] = None,
    path: Optional[str] = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Return Pivotal documentation by topic or allowlisted local docs path."""
    if path:
        doc_path = _resolve_doc_path(path)
        if doc_path is None:
            return {
                "ok": False,
                "topic": topic,
                "path": path,
                "content": "",
                "truncated": False,
                "message": f"Unknown Pivotal docs path '{path}'. Call pivotal_docs_index for valid paths.",
            }
        text = doc_path.read_text(encoding="utf-8")
        content, truncated = _trim_text(text, max_chars)
        relative = _relative_doc_path(doc_path)
        return {
            "ok": True,
            "topic": topic,
            "path": relative,
            "source": str(doc_path),
            "url": _public_doc_url(relative),
            "content": content,
            "truncated": truncated,
        }

    if not topic:
        return get_pivotal_docs_index()

    matches, terms = _best_matching_sections(_iter_doc_paths(), topic)
    if not matches:
        return {
            "ok": False,
            "topic": topic,
            "matched_terms": terms,
            "content": "",
            "matches": [],
            "truncated": False,
            "message": f"No Pivotal documentation section matched topic '{topic}'.",
        }

    best = matches[0]
    content, truncated = _trim_text(str(best["content"]), max_chars)
    return {
        "ok": True,
        "topic": topic,
        "matched_terms": terms,
        "path": best["path"],
        "source": best["source"],
        "url": best["url"],
        "heading": best["heading"],
        "start_line": best["start_line"],
        "content": content,
        "truncated": truncated,
        "matches": [
            {
                "path": match["path"],
                "url": match["url"],
                "heading": match["heading"],
                "start_line": match["start_line"],
                "score": match["score"],
            }
            for match in matches
        ],
    }


def search_pivotal_docs(query: str, max_results: int = 8, max_chars: int = 1200) -> dict[str, Any]:
    """Search Pivotal docs sections and return compact matching excerpts."""
    matches, terms = _best_matching_sections(_iter_doc_paths(), query, max_results=max_results)
    results: list[dict[str, Any]] = []
    for match in matches:
        content, truncated = _trim_text(str(match["content"]), max_chars)
        results.append({
            "path": match["path"],
            "source": match["source"],
            "url": match["url"],
            "heading": match["heading"],
            "start_line": match["start_line"],
            "score": match["score"],
            "excerpt": content,
            "truncated": truncated,
        })
    return {
        "ok": bool(results),
        "query": query,
        "matched_terms": terms,
        "results": results,
        "message": "" if results else f"No Pivotal docs matched query '{query}'.",
    }


def get_pivotal_syntax(topic: Optional[str] = None, max_chars: int = _FULL_SYNTAX_MAX_CHARS) -> dict[str, Any]:
    """Return all or part of PIVOTAL.md for MCP syntax grounding."""
    if not topic:
        text = _SYNTAX_PATH.read_text(encoding="utf-8")
        content, truncated = _trim_text(text, max_chars)
        return {
            "ok": True,
            "topic": None,
            "source": str(_SYNTAX_PATH),
            "path": _relative_doc_path(_SYNTAX_PATH),
            "url": _public_doc_url(_relative_doc_path(_SYNTAX_PATH)),
            "content": content,
            "truncated": truncated,
        }

    matches, terms = _best_matching_sections([_SYNTAX_PATH], topic)
    if matches:
        best = matches[0]
        content, truncated = _trim_text(str(best["content"]), max_chars)
        return {
            "ok": True,
            "topic": topic,
            "matched_terms": terms,
            "source": best["source"],
            "path": best["path"],
            "url": best["url"],
            "heading": best["heading"],
            "start_line": best["start_line"],
            "content": content,
            "truncated": truncated,
            "matches": [
                {
                    "path": match["path"],
                    "url": match["url"],
                    "heading": match["heading"],
                    "start_line": match["start_line"],
                    "score": match["score"],
                }
                for match in matches
            ],
        }

    return {
        "ok": False,
        "topic": topic,
        "matched_terms": terms,
        "source": str(_SYNTAX_PATH),
        "content": "",
        "truncated": False,
        "message": f"No Pivotal syntax section matched topic '{topic}'.",
    }


def compile_pivotal_source(source: str, backend: str = "pandas") -> dict[str, Any]:
    """Compile Pivotal source without executing the generated code."""
    parser = DSLParser()
    ast_list = parser.parse(source)
    if isinstance(ast_list, dict) and "error" in ast_list:
        err = ast_list["error"]
        return {
            "ok": False,
            "stage": "parse",
            "error_type": getattr(err, "error_type", "Error"),
            "message": getattr(err, "message", str(err)),
            "line": getattr(err, "line", None),
            "column": getattr(err, "column", None),
            "source_line": getattr(err, "source_line", None),
            "suggestion": getattr(err, "suggestion", None),
        }
    try:
        code_blocks = parser.generate_code(ast_list, backend=backend)
    except Exception as exc:  # noqa: BLE001 - structured tool boundary
        return {
            "ok": False,
            "stage": "codegen",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    generated_code = "\n\n".join(code_blocks)
    if backend != "sql":
        try:
            ast.parse(generated_code)
        except SyntaxError as exc:
            return {
                "ok": False,
                "stage": "codegen_syntax",
                "backend": backend,
                "error_type": type(exc).__name__,
                "message": exc.msg,
                "line": exc.lineno,
                "column": exc.offset,
                "generated_code": generated_code,
            }
    return {
        "ok": True,
        "stage": "codegen",
        "backend": backend,
        "generated_code": generated_code,
    }


def _highlight_type(ttype: Any) -> str:
    from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Text

    if ttype in Text:
        return "text"
    if ttype in Comment:
        return "comment"
    if ttype in Keyword.Constant:
        return "constant"
    if ttype in Keyword.Declaration:
        return "clause"
    if ttype in Keyword:
        return "keyword"
    if ttype in Name.Builtin:
        return "builtin"
    if ttype in Name.Variable:
        return "variable"
    if ttype in Name:
        return "name"
    if ttype in String:
        return "string"
    if ttype in Number:
        return "number"
    if ttype in Operator:
        return "operator"
    if ttype in Punctuation:
        return "punctuation"
    return "text"


def highlight_pivotal_source(
    source: str,
    *,
    include_html: bool = True,
    include_tokens: bool = True,
    include_css: bool = True,
    include_copy_button: bool = True,
) -> dict[str, Any]:
    """Return syntax-highlighted Pivotal as copyable HTML and/or a token stream."""
    from pygments import lex

    from .lexer import PivotalLexer

    parts: list[str] = []
    tokens: list[dict[str, Any]] = []
    offset = 0
    for ttype, text in lex(source, PivotalLexer()):
        kind = _highlight_type(ttype)
        end = offset + len(text)
        css_class = f"pvt-{kind}"
        if include_html:
            escaped = html.escape(text)
            if kind == "text":
                parts.append(escaped)
            else:
                parts.append(f'<span class="{css_class}">{escaped}</span>')
        if include_tokens:
            tokens.append({
                "text": text,
                "type": kind,
                "class": css_class,
                "start": offset,
                "end": end,
            })
        offset = end

    result: dict[str, Any] = {"ok": True}
    if include_html:
        highlighted_html = "".join(parts)
        result["html"] = (
            _render_highlighted_code_block(highlighted_html, include_css=include_css)
            if include_copy_button
            else highlighted_html
        )
    if include_tokens:
        result["tokens"] = tokens
    if include_css:
        result["css"] = _HIGHLIGHT_CSS
    return result


def get_pivotal_examples(kind: Optional[str] = None) -> dict[str, Any]:
    """Return MCP tool-call examples, especially input_files shape."""
    examples = {
        "run": {
            "description": "Run Pivotal against a CSV file loaded as table 'sales'.",
            "create_fixture_powershell": (
                "Set-Content -Path .\\sales.csv -Value @'\n"
                "region,product,amount,price,quantity,cost\n"
                "North,Widget,120,10,12,80\n"
                "South,Gadget,75,15,5,40\n"
                "West,Doohickey,240,20,12,150\n"
                "North,Gizmo,180,30,6,110\n"
                "'@"
            ),
            "tool": "pivotal_run",
            "arguments": {
                "source": (
                    "with sales as report\n"
                    "    filter amount > 100\n"
                    "    revenue = price * quantity\n"
                    "    margin = revenue - cost\n"
                    "    group by region\n"
                    "        agg sum revenue as total_revenue, sum margin as total_margin\n"
                    "    sort total_revenue desc\n"
                ),
                "input_files": {"sales": "C:\\path\\to\\sales.csv"},
                "return_tables": ["report"],
                "timeout_seconds": 30,
            },
            "notes": [
                "input_files is a mapping of Pivotal table name to local CSV/Parquet file path.",
                "The key must be a table name like 'sales', not a filename like 'sales.csv'.",
                "The value must be a path to an existing .csv or .parquet file, not inline CSV text.",
            ],
        },
        "compare": {
            "description": "Compare pandas code and Pivotal code on the same input table.",
            "tool": "pivotal_compare",
            "arguments": {
                "pandas_source": (
                    "report = sales.copy()\n"
                    "report = report[report['amount'] > 100].copy()\n"
                    "report['revenue'] = report['price'] * report['quantity']\n"
                    "report['margin'] = report['revenue'] - report['cost']\n"
                    "report = report.groupby('region', as_index=False).agg(\n"
                    "    total_revenue=('revenue', 'sum'),\n"
                    "    total_margin=('margin', 'sum'),\n"
                    ")\n"
                    "report = report.sort_values('total_revenue', ascending=False).reset_index(drop=True)\n"
                ),
                "pivotal_source": (
                    "with sales as report\n"
                    "    filter amount > 100\n"
                    "    revenue = price * quantity\n"
                    "    margin = revenue - cost\n"
                    "    group by region\n"
                    "        agg sum revenue as total_revenue, sum margin as total_margin\n"
                    "    sort total_revenue desc\n"
                ),
                "output_table": "report",
                "input_files": {"sales": "C:\\path\\to\\sales.csv"},
            },
        },
    }
    if kind:
        key = kind.lower()
        if key in examples:
            return {"ok": True, "kind": key, "examples": {key: examples[key]}}
        return {
            "ok": False,
            "kind": kind,
            "message": f"Unknown example kind '{kind}'. Use one of: {', '.join(examples)}.",
            "examples": {},
        }
    return {"ok": True, "kind": None, "examples": examples}


def _create_fastmcp(
    name: str,
    instructions: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    stateless_http: bool = False,
):
    """Create the FastMCP server.

    Importing the MCP SDK is intentionally lazy because Pivotal supports Python
    3.9, while the current MCP Python SDK requires Python 3.10+.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The Pivotal MCP server requires the optional MCP dependency. "
            "Install it with: pip install 'pivotal-lang[mcp]'"
        ) from exc

    return FastMCP(
        name,
        instructions=instructions,
        host=host,
        port=port,
        streamable_http_path="/mcp",
        stateless_http=stateless_http,
    )


def _register_readonly_tools(mcp) -> None:
    """Register tools that never execute user code or read user data files."""

    @mcp.tool()
    def pivotal_syntax(topic: Optional[str] = None, max_chars: int = _FULL_SYNTAX_MAX_CHARS) -> dict[str, Any]:
        """Return Pivotal language syntax guidance from PIVOTAL.md."""
        return get_pivotal_syntax(topic=topic, max_chars=max_chars)

    @mcp.tool()
    def pivotal_docs_index() -> dict[str, Any]:
        """Return available Pivotal docs pages and their headings."""
        return get_pivotal_docs_index()

    @mcp.tool()
    def pivotal_docs(
        topic: Optional[str] = None,
        path: Optional[str] = None,
        max_chars: int = 12000,
    ) -> dict[str, Any]:
        """Return Pivotal docs by topic or allowlisted docs path."""
        return get_pivotal_docs(topic=topic, path=path, max_chars=max_chars)

    @mcp.tool()
    def pivotal_docs_search(query: str, max_results: int = 8, max_chars: int = 1200) -> dict[str, Any]:
        """Search Pivotal docs sections and return compact excerpts."""
        return search_pivotal_docs(query=query, max_results=max_results, max_chars=max_chars)

    @mcp.tool()
    def pivotal_examples(kind: Optional[str] = None) -> dict[str, Any]:
        """Return working MCP call examples. kind can be 'run' or 'compare'."""
        return get_pivotal_examples(kind=kind)

    @mcp.tool()
    def pivotal_compile(
        source: str,
        backend: str = "pandas",
    ) -> dict[str, Any]:
        """Compile Pivotal source to backend code without executing data pipelines."""
        return compile_pivotal_source(source, backend=backend)

    @mcp.tool()
    def pivotal_highlight(
        source: str,
        include_html: bool = True,
        include_tokens: bool = True,
        include_css: bool = True,
        include_copy_button: bool = True,
    ) -> dict[str, Any]:
        """Return syntax-highlighted Pivotal source as copyable HTML and/or tokens."""
        try:
            return highlight_pivotal_source(
                source,
                include_html=include_html,
                include_tokens=include_tokens,
                include_css=include_css,
                include_copy_button=include_copy_button,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "highlight",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }


def _register_execution_tools(mcp) -> None:
    """Register local tools that may execute code or read local files."""

    @mcp.tool()
    def pivotal_run(
        source: str,
        backend: str = "pandas",
        input_files: Optional[dict[str, str]] = None,
        return_tables: Optional[list[str]] = None,
        max_rows: int = 20,
        timeout_seconds: float = 10,
        include_generated_code: bool = True,
    ) -> dict[str, Any]:
        """Run Pivotal source and return structured results.

        input_files must map Pivotal table names to existing local CSV/Parquet
        file paths. Example: {"sales": "C:\\data\\sales.csv"}. The key is the
        table name used in Pivotal (`with sales`), not the filename, and the
        value is a file path, not inline CSV content.
        """
        try:
            from .runner import run_pivotal_isolated

            inputs = _load_input_files(input_files)
            return run_pivotal_isolated(
                source,
                backend=backend,
                inputs=inputs,
                return_tables=return_tables,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                include_generated_code=include_generated_code,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "mcp_input",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    @mcp.tool()
    def pivotal_compare(
        pandas_source: str,
        pivotal_source: str,
        output_table: str,
        input_files: Optional[dict[str, str]] = None,
        backend: str = "pandas",
        max_rows: int = 20,
        timeout_seconds: float = 10,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        check_dtype: bool = False,
        max_differences: int = 20,
    ) -> dict[str, Any]:
        """Compare a pandas script and Pivotal source on the same inputs.

        input_files must map table/variable names to existing local CSV/Parquet
        file paths. Example: {"sales": "C:\\data\\sales.csv"} makes a DataFrame
        named `sales` available to both pandas_source and pivotal_source.
        """
        try:
            from .runner import compare_pandas_to_pivotal_isolated

            inputs = _load_input_files(input_files)
            return compare_pandas_to_pivotal_isolated(
                pandas_source,
                pivotal_source,
                output_table=output_table,
                backend=backend,
                inputs=inputs,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                atol=atol,
                rtol=rtol,
                check_dtype=check_dtype,
                max_differences=max_differences,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "mcp_input",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    @mcp.tool()
    def pivotal_compare_files(
        pandas_path: str,
        pivotal_path: str,
        output_table: str,
        input_files: Optional[dict[str, str]] = None,
        backend: str = "pandas",
        max_rows: int = 20,
        timeout_seconds: float = 10,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        check_dtype: bool = False,
        max_differences: int = 20,
    ) -> dict[str, Any]:
        """Compare pandas and Pivotal files on the same inputs.

        input_files has the same format as pivotal_compare: table name keys,
        local CSV/Parquet path values.
        """
        try:
            return pivotal_compare(
                _read_text_file(pandas_path),
                _read_text_file(pivotal_path),
                output_table,
                input_files=input_files,
                backend=backend,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                atol=atol,
                rtol=rtol,
                check_dtype=check_dtype,
                max_differences=max_differences,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "mcp_input",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

def create_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    stateless_http: bool = False,
):
    """Create the full local MCP server with compile, run, and compare tools."""
    mcp = _create_fastmcp(
        "Pivotal",
        instructions=(
            "Tools for generating, running, and comparing Pivotal DSL code. "
            f"{_CORE_GUIDANCE}"
            f"{_VERIFICATION_GUIDANCE}"
            "When syntax is uncertain, call pivotal_docs_search, pivotal_docs, "
            "or pivotal_syntax before writing code, then verify generated code "
            "with pivotal_run or pivotal_compare before giving a final answer. "
            "Prefer native Pivotal syntax and use python blocks only for "
            "operations that Pivotal cannot express. For tools with input_files, "
            "pass a mapping of Pivotal table name to local CSV/Parquet file path, "
            "for example {'sales': 'C:\\\\data\\\\sales.csv'}. Do not use the "
            "filename as the key, and do not pass inline CSV text as the value."
        ),
        host=host,
        port=port,
        stateless_http=stateless_http,
    )
    _register_readonly_tools(mcp)
    _register_execution_tools(mcp)
    return mcp


def create_readonly_mcp_server(
    *,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    stateless_http: bool = True,
):
    """Create a hosted-safe MCP server with docs, syntax, examples, and compile tools."""
    if port is None:
        port = int(os.environ.get("PORT", "8000"))
    mcp = _create_fastmcp(
        "Pivotal Read Only",
        instructions=(
            "Read-only tools for learning and compiling Pivotal DSL code. "
            f"{_CORE_GUIDANCE}"
            f"{_VERIFICATION_GUIDANCE}"
            "When syntax is uncertain, call pivotal_docs_search, pivotal_docs, "
            "or pivotal_syntax before writing code, then use pivotal_compile "
            "to check that the source parses and compiles. "
            "This server parses and compiles Pivotal source but does not run "
            "data pipelines, execute user Python, read user files, or compare "
            "against pandas scripts."
        ),
        host=host,
        port=port,
        stateless_http=stateless_http,
    )
    _register_readonly_tools(mcp)
    return mcp


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pivotal MCP server.")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Expose only docs, syntax, examples, and compile tools for hosted use.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport to use. Default preserves local stdio behavior.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host for HTTP transports. Defaults to 0.0.0.0 in read-only mode, otherwise 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for HTTP transports. Defaults to $PORT or 8000.",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Optional mount path passed through to FastMCP.run().",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "8000"))
    host = args.host or ("0.0.0.0" if args.read_only else "127.0.0.1")
    stateless_http = args.read_only and args.transport == "streamable-http"

    if args.read_only:
        server = create_readonly_mcp_server(
            host=host,
            port=port,
            stateless_http=stateless_http,
        )
    else:
        server = create_mcp_server(
            host=host,
            port=port,
            stateless_http=stateless_http,
        )
    server.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
