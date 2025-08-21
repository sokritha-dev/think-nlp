from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Lemmatizer(ABC):
    """Port: lemmatize a list of tokens (optionally using POS)."""

    @abstractmethod
    def lemmatize(self, tokens: List[str]) -> List[str]: ...
