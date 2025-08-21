from __future__ import annotations
from collections import Counter
from typing import Dict, Any, List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from app.core.eda.base import EDAAnalyzer
from app.core.eda.config import EDAConfig


class DefaultEDAAnalyzer(EDAAnalyzer):
    """Adapter: fast, dependency-light EDA on lemmatized tokens."""

    def __init__(self, config: EDAConfig | None = None):
        self.cfg = config or EDAConfig()

    def _docs_from_column(self, series: pd.Series) -> List[str]:
        # Accept list[str] or already-joined strings; coerce to whitespace-joined strings.
        docs: List[str] = []
        for x in series.fillna(""):
            if isinstance(x, list):
                docs.append(" ".join(map(str, x)))
            else:
                docs.append(str(x))
        return docs

    def _extract_ngrams(self, docs: List[str], n: int, top_k: int, label: str):
        vectorizer = CountVectorizer(
            stop_words=("english" if self.cfg.use_sklearn_stopwords else None),
            ngram_range=(n, n),
            token_pattern=r"\b\w+\b",
        )
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        freqs = X.sum(axis=0).A1
        s = pd.Series(freqs, index=vocab).sort_values(ascending=False).head(top_k)
        return [{label: k, "count": int(v)} for k, v in s.items()]

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.cfg.text_column not in df.columns:
            raise KeyError(f"Column '{self.cfg.text_column}' not found in DataFrame.")
        docs = self._docs_from_column(df[self.cfg.text_column])

        # Word cloud: top N tokens by raw frequency (no stopword removal here by design)
        all_words = " ".join(docs).split()
        word_freq = Counter(all_words)
        word_cloud = [
            {"text": w, "value": int(c)}
            for w, c in word_freq.most_common(self.cfg.top_words)
        ]

        # Length distribution: token count per doc
        lengths = [len(d.split()) for d in docs]
        length_series = pd.Series(lengths)
        length_distribution = (
            length_series.value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"index": "length", 0: "count"})
        )
        length_distribution_data = [
            {"length": int(r["length"]), "count": int(r["count"])}
            for _, r in length_distribution.iterrows()
        ]

        # N-grams
        bigrams = self._extract_ngrams(
            docs, n=2, top_k=self.cfg.ngram_top_k, label="bigram"
        )
        trigrams = self._extract_ngrams(
            docs, n=3, top_k=self.cfg.ngram_top_k, label="trigram"
        )

        return {
            "word_cloud": word_cloud,
            "length_distribution": length_distribution_data,
            "bigrams": bigrams,
            "trigrams": trigrams,
        }
