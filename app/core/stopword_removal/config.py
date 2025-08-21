from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Iterable


def _to_set(x: Iterable[str] | None) -> Set[str]:
    return set(map(str, x or []))


@dataclass(frozen=True)
class StopwordConfig:
    language: str = "english"
    custom_stopwords: Set[str] = field(default_factory=set)  # extra words to remove
    exclude_stopwords: Set[str] = field(
        default_factory=set
    )  # words to keep even if in list
    lowercase: bool = True
    preserve_negations: bool = True  # keep {no, not, never}
