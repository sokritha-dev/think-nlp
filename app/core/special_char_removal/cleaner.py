from __future__ import annotations
import re
from typing import Tuple, List

from app.core.special_char_removal.base import SpecialCleaner
from app.core.special_char_removal.config import SpecialCleanConfig

try:
    import emoji  # pip install emoji

    _HAS_EMOJI = True
except Exception:  # pragma: no cover
    _HAS_EMOJI = False


def _get_removed_chars(original: str, cleaned: str) -> List[str]:
    # unique chars removed; stable order not required
    return sorted(list(set(original) - set(cleaned)))


class DefaultSpecialCleaner(SpecialCleaner):
    """
    Adapter: matches your current behavior:
      - optionally remove emojis first
      - then apply regex based on remove_special/remove_numbers
    """

    def __init__(self, config: SpecialCleanConfig | None = None):
        self.cfg = config or SpecialCleanConfig()

    def clean(self, text: str) -> Tuple[str, List[str]]:
        original = text or ""

        s = original
        if self.cfg.remove_emoji and _HAS_EMOJI:
            s = emoji.replace_emoji(s, replace="")
        elif self.cfg.remove_emoji:
            # basic fallback: strip common emoji ranges
            s = re.sub(r"[\U00010000-\U0010ffff]", "", s)

        # Build the pattern
        if self.cfg.remove_special and self.cfg.remove_numbers:
            pattern = r"[^a-zA-Z\s]"
        elif self.cfg.remove_special and not self.cfg.remove_numbers:
            pattern = r"[^a-zA-Z0-9\s]"
        elif not self.cfg.remove_special and self.cfg.remove_numbers:
            pattern = r"[0-9]"
        else:
            cleaned = s
            return cleaned, _get_removed_chars(original, cleaned)

        cleaned = re.sub(pattern, "", s)
        removed = _get_removed_chars(original, cleaned)
        return cleaned, removed
