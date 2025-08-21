from __future__ import annotations
from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str: ...
