from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple


class StopwordRemover(ABC):
    """Port: remove stopwords from a token list."""

    @abstractmethod
    def remove(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Returns (cleaned_tokens, removed_stopwords)
        """
        ...
