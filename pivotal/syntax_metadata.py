"""Shared syntax-token metadata for Pivotal highlighters and editor tooling."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def load_syntax_tokens() -> dict[str, list[str]]:
    """Load the canonical syntax-token metadata bundled with Pivotal."""
    text = resources.files("pivotal").joinpath("syntax_tokens.json").read_text(encoding="utf-8")
    data = json.loads(text)
    return {key: list(value) for key, value in data.items()}


def token_category(name: str) -> list[str]:
    """Return one token category by name."""
    return load_syntax_tokens().get(name, [])


def all_token_metadata() -> dict[str, Any]:
    """Return a JSON-serialisable copy of the token metadata."""
    return {key: list(value) for key, value in load_syntax_tokens().items()}
