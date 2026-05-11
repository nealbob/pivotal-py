"""Generate editor highlighter assets from pivotal/syntax_tokens.json."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOKENS_PATH = ROOT / "pivotal" / "syntax_tokens.json"
VSCODE_GRAMMAR = ROOT / "editors" / "vscode" / "syntaxes" / "pivotal.tmLanguage.json"
JUPYTER_LANGUAGE = ROOT / "editors" / "jupyterlab" / "src" / "language.ts"


def _load_tokens() -> dict[str, list[str]]:
    return json.loads(TOKENS_PATH.read_text(encoding="utf-8"))


def _regex_words(words: list[str]) -> str:
    return "|".join(re.escape(word) for word in words)


def _unique(words: list[str]) -> list[str]:
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return result


def _ts_record(name: str, words: list[str]) -> str:
    lines = [f"const {name}: Record<string, true> = {{"]
    for word in words:
        lines.append(f"  {word}: true,")
    lines.append("};")
    return "\n".join(lines)


def _replace_ts_record(text: str, name: str, words: list[str]) -> str:
    pattern = rf"const {name}: Record<string, true> = \{{.*?\n\}};"
    replacement = _ts_record(name, words)
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Could not find TypeScript record '{name}' in {JUPYTER_LANGUAGE}")
    return new_text


def generate_vscode(tokens: dict[str, list[str]]) -> str:
    grammar = json.loads(VSCODE_GRAMMAR.read_text(encoding="utf-8"))
    repo = grammar["repository"]
    repo["keywords-primary"]["match"] = rf"\b({_regex_words(tokens['statement_keywords'])})\b"
    repo["keywords-clause"]["match"] = rf"\b({_regex_words(tokens['clause_keywords'])})\b"
    repo["keywords-merge-type"]["match"] = rf"\b({_regex_words(tokens['merge_modifiers'])})\b"
    repo["sort-modifiers"]["match"] = rf"\b({_regex_words(tokens['sort_modifiers'])})\b"

    word_ops = tokens["word_operators"]
    not_ops = tokens["negatable_word_operators"]
    repo["keywords-comparator"]["patterns"][0]["match"] = rf"\bnot\s+({_regex_words(not_ops)})\b"
    repo["keywords-comparator"]["patterns"][1]["match"] = rf"\b({_regex_words([op for op in word_ops if op not in ('and', 'or')])})\b"
    repo["keywords-logical"]["match"] = rf"(?i)\b({_regex_words([op for op in word_ops if op in ('and', 'or')])})\b"
    repo["agg-functions"]["match"] = rf"\b({_regex_words(tokens['builtin_functions'])})\b"

    return json.dumps(grammar, indent=2, ensure_ascii=False) + "\n"


def generate_jupyter(tokens: dict[str, list[str]]) -> str:
    text = JUPYTER_LANGUAGE.read_text(encoding="utf-8")
    keywords = _unique(tokens["statement_keywords"] + tokens["clause_keywords"])
    builtins = _unique(
        tokens["builtin_functions"]
        + tokens["merge_modifiers"]
        + tokens["sort_modifiers"]
        + tokens["word_operators"]
        + tokens["format_types"]
    )
    text = _replace_ts_record(text, "keywords", keywords)
    text = _replace_ts_record(text, "builtins", builtins)
    text = _replace_ts_record(text, "atoms", tokens["constants"])
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated assets are stale.")
    args = parser.parse_args(argv)

    tokens = _load_tokens()
    outputs = {
        VSCODE_GRAMMAR: generate_vscode(tokens),
        JUPYTER_LANGUAGE: generate_jupyter(tokens),
    }

    stale = []
    for path, new_text in outputs.items():
        old_bytes = path.read_bytes()
        new_bytes = new_text.encode("utf-8")
        if old_bytes != new_bytes:
            stale.append(path)
            if not args.check:
                path.write_bytes(new_bytes)

    if stale and args.check:
        for path in stale:
            print(f"stale generated syntax asset: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
