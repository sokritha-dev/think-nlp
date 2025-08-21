from __future__ import annotations
import re
from typing import List

try:
    from nltk.tokenize import wordpunct_tokenize  # type: ignore

    _HAS_NLTK = True
except Exception:  # pragma: no cover
    _HAS_NLTK = False

from app.core.tokenization.base import Tokenizer
from app.core.tokenization.config import TokenizationConfig


class DefaultTokenizer(Tokenizer):
    """Adapter: tokenizes using NLTK wordpunct with safe fallbacks & filters."""

    def __init__(self, config: TokenizationConfig | None = None):
        self.cfg = config or TokenizationConfig()
        self._regex = re.compile(self.cfg.regex_pattern or r"\b\w+\b")

    def _tokenize_raw(self, text: str) -> List[str]:
        s = text or ""
        if self.cfg.method == "regex" or not _HAS_NLTK:
            return self._regex.findall(s)
        return [t for t in wordpunct_tokenize(s)]

    def tokenize(self, text: str) -> List[str]:
        toks = self._tokenize_raw(text)
        out: List[str] = []
        for t in toks:
            if not t and self.cfg.drop_empty_tokens:
                continue
            if self.cfg.lowercase:
                t = t.lower()
            if self.cfg.keep_alnum_only and re.search(r"[^A-Za-z0-9]", t):
                continue
            if self.cfg.remove_numbers_only and t.isdigit():
                continue
            if len(t) < self.cfg.min_token_len:
                continue
            out.append(t)
        return out
