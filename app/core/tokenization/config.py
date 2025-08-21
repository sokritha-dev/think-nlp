from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenizationConfig:
    method: str = "wordpunct"  # "wordpunct" | "regex"
    regex_pattern: Optional[str] = None  # used if method == "regex"
    lowercase: bool = (
        False  # token-level lowercasing (usually False if already cleaned)
    )
    min_token_len: int = 1  # drop tokens shorter than this
    keep_alnum_only: bool = False  # drop tokens with non-alnum chars
    remove_numbers_only: bool = False  # drop tokens that are purely digits
    drop_empty_tokens: bool = True
