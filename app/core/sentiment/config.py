from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SentimentConfig:
    method: Literal["vader", "textblob", "bert"] = "vader"
    # Column in the labeled DataFrame used to build the text
    text_column: str = "lemmatized_tokens"
    # Topic identifiers & labels in the labeled DataFrame
    topic_id_col: str = "topic_id"
    topic_label_col: str = "topic_label"
