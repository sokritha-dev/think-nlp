from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from app.core.topic_modeling.base import TopicModeler, TopicEstimator
from app.core.topic_modeling.config import TopicModelConfig
from app.core.topic_modeling.utils import ensure_token_list


class SklearnLDAModeler(TopicModeler, TopicEstimator):
    def __init__(self, cfg: TopicModelConfig):
        self.cfg = cfg

    def _to_bow(self, token_series: pd.Series):
        docs = [" ".join(ensure_token_list(x)) for x in token_series]
        vect = CountVectorizer(max_features=self.cfg.max_features)
        X = vect.fit_transform(docs)
        return X, vect

    def estimate_k(self, token_series: pd.Series, min_k: int, max_k: int) -> int:
        sample = (
            token_series.sample(
                n=min(len(token_series), self.cfg.sample_docs_for_estimation)
            )
            if self.cfg.sample_docs_for_estimation
            else token_series
        )
        X, _ = self._to_bow(sample)
        best_k, best_score = min_k, float("inf")
        for k in range(min_k, max_k + 1):
            lda = LatentDirichletAllocation(
                n_components=k, random_state=self.cfg.random_state
            )
            lda.fit(X)
            score = lda.perplexity(X)
            if score < best_score:
                best_score, best_k = score, k
        return best_k

    def fit_predict(
        self, token_series: pd.Series, num_topics: int
    ) -> Tuple[List[int], List[Dict]]:
        X, vect = self._to_bow(token_series)
        lda = LatentDirichletAllocation(
            n_components=num_topics, random_state=self.cfg.random_state
        )
        doc_topic = lda.fit_transform(X)  # shape: (n_docs, num_topics)
        dom = doc_topic.argmax(axis=1).tolist()

        # topic summary
        feature_names = vect.get_feature_names_out()
        topics: List[Dict] = []
        comp = lda.components_  # shape: (n_topics, n_terms)
        for i, row in enumerate(comp):
            top_idx = row.argsort()[-self.cfg.topn_words :][::-1]
            words = [feature_names[j] for j in top_idx]
            topics.append(
                {
                    "topic_id": str(i),
                    "keywords": ", ".join(words),
                    "label": f"Topic {i}",
                }
            )
        return dom, topics
