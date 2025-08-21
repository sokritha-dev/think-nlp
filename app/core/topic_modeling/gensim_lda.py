from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel

from app.core.topic_modeling.base import TopicModeler, TopicEstimator
from app.core.topic_modeling.config import TopicModelConfig
from app.core.topic_modeling.utils import ensure_token_list


class GensimLDAModeler(TopicModeler, TopicEstimator):
    def __init__(self, cfg: TopicModelConfig):
        self.cfg = cfg

    def _prepare(self, token_series: pd.Series):
        texts: List[List[str]] = [ensure_token_list(x) for x in token_series]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(t) for t in texts]
        return texts, dictionary, corpus

    def estimate_k(self, token_series: pd.Series, min_k: int, max_k: int) -> int:
        texts, dictionary, corpus = self._prepare(
            token_series.sample(
                n=min(len(token_series), self.cfg.sample_docs_for_estimation)
            )
            if self.cfg.sample_docs_for_estimation
            else token_series
        )
        best_k, best_score = min_k, float("-inf")
        for k in range(min_k, max_k + 1):
            lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=self.cfg.passes,
                random_state=self.cfg.random_state,
            )
            coh = CoherenceModel(
                model=lda, texts=texts, dictionary=dictionary, coherence="c_v"
            ).get_coherence()
            if coh > best_score:
                best_score, best_k = coh, k
        return best_k

    def fit_predict(
        self, token_series: pd.Series, num_topics: int
    ) -> Tuple[List[int], List[Dict]]:
        texts, dictionary, corpus = self._prepare(token_series)
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=self.cfg.passes,
            random_state=self.cfg.random_state,
        )
        # dominant topic per doc
        dom: List[int] = []
        for bow in corpus:
            topic_probs = lda.get_document_topics(bow)
            dom.append(max(topic_probs, key=lambda x: x[1])[0] if topic_probs else -1)

        # topic summary
        topics: List[Dict] = []
        for i in range(num_topics):
            words = [w for w, _ in lda.show_topic(i, topn=self.cfg.topn_words)]
            topics.append(
                {
                    "topic_id": str(i),
                    "keywords": ", ".join(words),
                    "label": f"Topic {i}",
                }
            )
        return dom, topics
