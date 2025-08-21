from __future__ import annotations
import re
import unicodedata
import contractions
from app.core.normalization.base import TextNormalizer
from app.core.normalization.config import NormalizationConfig


class DefaultTextNormalizer(TextNormalizer):
    _re_multi_ws = re.compile(r"\s{2,}")

    def __init__(self, config: NormalizationConfig | None = None):
        self.cfg = config or NormalizationConfig()

    def _fix_broken(self, s: str) -> str:
        for broken, fixed in self.cfg.broken_map.items():
            s = s.replace(broken, fixed)
        return s

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        s = str(text)

        if self.cfg.unicode_nfkc:
            s = unicodedata.normalize("NFKC", s)

        s = self._fix_broken(s)

        if self.cfg.expand_contractions:
            s = contractions.fix(s)

        if self.cfg.lowercase:
            s = s.lower()

        s = s.strip()
        if self.cfg.collapse_whitespace:
            s = self._re_multi_ws.sub(" ", s)
        return s
