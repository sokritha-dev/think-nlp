from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TopicModelConfig:
    backend: Literal["gensim", "sklearn"] = "gensim"
    passes: int = 10  # gensim only
    random_state: int = 42
    topn_words: int = 10  # words per topic (summary)
    max_features: Optional[int] = None  # sklearn CountVectorizer cap
    # safety / performance:
    sample_docs_for_estimation: Optional[int] = 5000  # None = use all


@dataclass(frozen=True)
class TopicEstimationConfig:
    method: Literal["coherence", "perplexity"] = "perplexity"
    min_k: int = 3
    max_k: int = 10
