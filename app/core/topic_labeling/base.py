from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class TopicLabeler(ABC):
    """
    Given topic summaries and (optionally) user inputs, produce:
      - label_map: {topic_id -> label}
      - enriched_topics: topics with fields (label, confidence?, matched_with?)
    """

    @abstractmethod
    def label(
        self,
        topics: List[dict],
        *,
        explicit_map: Optional[Dict[int, str]] = None,
        user_keywords: Optional[List[str]] = None,
    ) -> Tuple[Dict[int, str], List[dict]]: ...
