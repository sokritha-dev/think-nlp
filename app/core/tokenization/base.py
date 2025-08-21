from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """Port: split a cleaned string into a list of tokens."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]: ...
