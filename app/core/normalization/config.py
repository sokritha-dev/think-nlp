from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


def default_broken_map() -> Dict[str, str]:
    # Same defaults you’ve been using
    return {
        "ca n't": "can't",
        "wo n't": "won't",
        "did n't": "didn't",
        "does n't": "doesn't",
        "is n't": "isn't",
        "was n't": "wasn't",
        "were n't": "weren't",
        "should n't": "shouldn't",
        "would n't": "wouldn't",
        "could n't": "couldn't",
        "do n't": "don't",
        "has n't": "hasn't",
        "have n't": "have n't".replace(" ", ""),  # keeps your behavior
        "had n't": "hadn't",
        "must n't": "mustn't",
        "might n't": "mightn't",
    }


@dataclass(frozen=True)
class NormalizationConfig:
    broken_map: Dict[str, str] = field(default_factory=default_broken_map)
    lowercase: bool = True
    collapse_whitespace: bool = True
    unicode_nfkc: bool = True
    expand_contractions: bool = True  # uses python 'contractions' package
