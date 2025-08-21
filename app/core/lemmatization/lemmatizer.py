from __future__ import annotations
from typing import List

from app.core.lemmatization.base import Lemmatizer
from app.core.lemmatization.config import LemmatizationConfig

try:
    from nltk import pos_tag  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore

    _HAS_NLTK = True
except Exception:  # pragma: no cover
    _HAS_NLTK = False


def _to_wn_pos(tag: str):
    # Penn tag -> WordNet POS
    if tag.startswith("J"):
        return "a"  # ADJ
    if tag.startswith("V"):
        return "v"  # VERB
    if tag.startswith("N"):
        return "n"  # NOUN
    if tag.startswith("R"):
        return "r"  # ADV
    return "n"  # default


class DefaultLemmatizer(Lemmatizer):
    def __init__(self, config: LemmatizationConfig | None = None):
        self.cfg = config or LemmatizationConfig()
        self._wn = WordNetLemmatizer() if _HAS_NLTK else None

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if not _HAS_NLTK or self._wn is None:
            return tokens if self.cfg.fallback_if_missing else []

        toks = [t.lower() for t in tokens] if self.cfg.lowercase else list(tokens)

        if not self.cfg.use_pos_tagging:
            return [self._wn.lemmatize(t) for t in toks]

        # POS-aware lemmatization
        try:
            tags = pos_tag(toks)  # [('arrived','VBD'),('rooms','NNS'),...]
        except Exception:
            # fallback to no-POS lemmatization
            return [self._wn.lemmatize(t) for t in toks]

        out: List[str] = []
        for token, tag in tags:
            out.append(self._wn.lemmatize(token, _to_wn_pos(tag)))
        return out
