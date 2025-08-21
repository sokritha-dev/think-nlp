from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List


class SpecialCleaner(ABC):
    """Port: removes special chars/numbers/emoji according to config."""

    @abstractmethod
    def clean(self, text: str) -> Tuple[str, List[str]]:
        """
        Returns (cleaned_text, removed_characters_list)
        removed_characters_list can be deduplicated by the caller if needed.
        """
        ...
