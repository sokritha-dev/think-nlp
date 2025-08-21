from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import pandas as pd


class TopicEstimator(ABC):
    @abstractmethod
    def estimate_k(self, token_series: pd.Series, min_k: int, max_k: int) -> int: ...


class TopicModeler(ABC):
    @abstractmethod
    def fit_predict(
        self, token_series: pd.Series, num_topics: int
    ) -> Tuple[List[int], List[Dict]]:
        """
        Returns:
          - dominant topic id per document (len == len(token_series))
          - topic summary list: [{topic_id, keywords, label}, ...]
        """
        ...
