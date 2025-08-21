from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LemmatizationConfig:
    use_pos_tagging: bool = True  # tag tokens then lemmatize by POS
    lowercase: bool = False  # lowercase tokens before lemmatization (usually False)
    fallback_if_missing: bool = True  # if NLTK unavailable, return tokens unchanged
