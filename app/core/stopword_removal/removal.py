from __future__ import annotations
from typing import List, Tuple, Set

from app.core.stopword_removal.base import StopwordRemover
from app.core.stopword_removal.config import StopwordConfig

try:
    from nltk.corpus import stopwords as nltk_stopwords  # type: ignore

    _HAS_NLTK = True
except Exception:  # pragma: no cover
    _HAS_NLTK = False


# small, safe default if NLTK isn't available
_FALLBACK = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "to",
    "from",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
}


class DefaultStopwordRemover(StopwordRemover):
    def __init__(self, config: StopwordConfig | None = None):
        self.cfg = config or StopwordConfig()
        self._stopset = self._build_stopset()

    def _build_stopset(self) -> Set[str]:
        base: Set[str] = set()
        if _HAS_NLTK:
            try:
                base |= set(nltk_stopwords.words(self.cfg.language))
            except Exception:
                base |= _FALLBACK
        else:
            base |= _FALLBACK

        base |= set(self.cfg.custom_stopwords)
        base -= set(self.cfg.exclude_stopwords)

        if self.cfg.preserve_negations:
            for w in ("no", "not", "never"):
                base.discard(w)

        if self.cfg.lowercase:
            base = {w.lower() for w in base}
        return base

    def remove(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        cleaned: List[str] = []
        removed: List[str] = []
        for t in tokens:
            norm = t.lower() if self.cfg.lowercase else t
            if norm in self._stopset:
                removed.append(t)
                continue
            cleaned.append(t)
        return cleaned, removed
