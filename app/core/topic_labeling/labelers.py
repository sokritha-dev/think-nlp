from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from app.core.topic_labeling.base import TopicLabeler
from app.core.topic_labeling.config import TopicLabelConfig


# --- helper used by default heuristic ---
def _generate_default_labels(topics: List[dict], k: int) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for t in topics:
        tid = int(t["topic_id"])
        kw = (t.get("keywords") or "").split(", ")
        label = " & ".join(kw[:k]) if kw else f"Topic {tid}"
        out[tid] = label
    return out


# --- Explicit mapping ---
@dataclass
class ExplicitLabeler(TopicLabeler):
    cfg: TopicLabelConfig

    def label(
        self,
        topics: List[dict],
        *,
        explicit_map: Optional[Dict[int, str]] = None,
        user_keywords: Optional[List[str]] = None,
    ) -> Tuple[Dict[int, str], List[dict]]:
        if not explicit_map:
            raise ValueError("ExplicitLabeler requires `explicit_map`.")
        # enrich topics
        enriched = []
        for t in topics:
            tid = int(t["topic_id"])
            lbl = explicit_map.get(tid, f"Topic {tid}")
            enriched.append(
                {
                    **t,
                    "label": lbl,
                    "confidence": None,
                    "matched_with": "explicit_map",
                }
            )
        return explicit_map, enriched


# --- SBERT keyword matcher (lazy load, safe fallback) ---
class SBertKeywordLabeler(TopicLabeler):
    def __init__(self, cfg: TopicLabelConfig):
        self.cfg = cfg
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model_name)
        except Exception:
            # mark as unavailable; we'll do a simple fallback
            self._model = False  # sentinel

    def _fallback_match(
        self, topics: List[dict], user_keywords: List[str]
    ) -> Tuple[Dict[int, str], List[dict]]:
        # naive similarity: longest common substring length ratio
        def score(a: str, b: str) -> float:
            a, b = a.lower(), b.lower()
            best = 0
            for i in range(len(a)):
                for j in range(i + 1, len(a) + 1):
                    s = a[i:j]
                    if s and s in b:
                        best = max(best, len(s))
            return best / max(1, len(b))

        label_map: Dict[int, str] = {}
        enriched: List[dict] = []
        for t in topics:
            tid = int(t["topic_id"])
            topic_kw = (t.get("keywords") or "").replace(",", " ")
            best_lbl, best_sc = None, -1.0
            for kw in user_keywords:
                sc = score(topic_kw, kw)
                if sc > best_sc:
                    best_sc, best_lbl = sc, kw
            lbl = best_lbl or f"Topic {tid}"
            label_map[tid] = lbl
            enriched.append(
                {
                    **t,
                    "label": lbl,
                    "confidence": round(best_sc, 4),
                    "matched_with": "fallback_keywords",
                }
            )
        return label_map, enriched

    def label(
        self,
        topics: List[dict],
        *,
        explicit_map: Optional[Dict[int, str]] = None,
        user_keywords: Optional[List[str]] = None,
    ) -> Tuple[Dict[int, str], List[dict]]:
        if not user_keywords:
            raise ValueError("SBertKeywordLabeler requires `user_keywords`.")
        self._ensure_model()
        if self._model is False:
            return self._fallback_match(topics, user_keywords)

        # real SBERT
        from sentence_transformers import util

        model = self._model
        user_embeds = model.encode(user_keywords, convert_to_tensor=True)

        label_map: Dict[int, str] = {}
        enriched: List[dict] = []
        for t in topics:
            tid = int(t["topic_id"])
            topic_keywords = (t.get("keywords") or "").replace(",", " ")
            topic_embed = model.encode(topic_keywords, convert_to_tensor=True)
            sims = util.cos_sim(topic_embed, user_embeds)[0]
            best_idx = int(sims.argmax().item())
            best_score = float(sims[best_idx].item())
            matched_label = user_keywords[best_idx]
            label_map[tid] = matched_label
            enriched.append(
                {
                    **t,
                    "label": matched_label,
                    "confidence": round(best_score, 4),
                    "matched_with": "auto_match_keywords",
                }
            )
        return label_map, enriched


# --- Default heuristic (top-k keywords) ---
@dataclass
class DefaultHeuristicLabeler(TopicLabeler):
    cfg: TopicLabelConfig

    def label(
        self,
        topics: List[dict],
        *,
        explicit_map: Optional[Dict[int, str]] = None,
        user_keywords: Optional[List[str]] = None,
    ) -> Tuple[Dict[int, str], List[dict]]:
        label_map = _generate_default_labels(topics, self.cfg.num_keywords)
        enriched = []
        for t in topics:
            tid = int(t["topic_id"])
            lbl = label_map.get(tid, f"Topic {tid}")
            enriched.append(
                {
                    **t,
                    "label": lbl,
                    "confidence": None,
                    "matched_with": "auto_generated",
                }
            )
        return label_map, enriched
