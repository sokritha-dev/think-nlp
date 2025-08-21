from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TopicLabelConfig:
    strategy: Literal["explicit", "keywords", "default"] = "default"
    model_name: str = "all-MiniLM-L6-v2"  # for SBERT
    num_keywords: int = 2  # for default heuristic
