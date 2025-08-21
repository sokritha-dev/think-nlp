from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class EDAConfig:
    text_column: str = "lemmatized_tokens"  # list[str] or whitespace-joined str
    top_words: int = 100  # for word cloud
    ngram_top_k: int = 20  # top-K bigrams/trigrams
    use_sklearn_stopwords: bool = True  # use sklearn 'english' list in n-grams
