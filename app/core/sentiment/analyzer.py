# app/core/sentiment/analyzer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Union, Iterable, List
import pandas as pd


@dataclass(frozen=True)
class SentimentConfig:
    method: str = "vader"  # "vader" | "textblob" | "bert"
    # Thresholds:
    vader_pos: float = 0.05
    vader_neg: float = -0.05
    blob_pos: float = 0.10
    blob_neg: float = -0.10
    # For 2-class models (BERT): define a "neutral band" around 0.5
    bert_neutral_band: float = 0.05  # 0.45..0.55 → neutral
    # Inference controls:
    bert_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    bert_max_length: int = 512
    bert_batch_size: int = 32
    bert_device: Optional[int] = None  # None=auto CPU, or int CUDA device id


class SentimentAnalyzer(Protocol):
    def score_series(self, texts: pd.Series) -> pd.Series: ...


# ----------------------------
# Utilities
# ----------------------------


def _as_text_series(x: Union[pd.Series, Iterable[str], List[str]]) -> pd.Series:
    s = x if isinstance(x, pd.Series) else pd.Series(list(x))
    # normalize to strings, keep index
    return s.fillna("").astype(str)


# ----------------------------
# VADER
# ----------------------------


class VaderAnalyzer:
    def __init__(self, cfg: SentimentConfig):
        self.cfg = cfg
        self._sia = None  # lazy

    def _ensure_sia(self):
        if self._sia is not None:
            return
        import nltk

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer

        self._sia = SentimentIntensityAnalyzer()

    def _label(self, txt: str) -> str:
        sc = self._sia.polarity_scores(txt)["compound"]
        if sc > self.cfg.vader_pos:
            return "positive"
        if sc < self.cfg.vader_neg:
            return "negative"
        return "neutral"

    def score_series(self, texts: pd.Series) -> pd.Series:
        self._ensure_sia()
        s = _as_text_series(texts)
        # map preserves index
        return s.map(self._label)


# ----------------------------
# TextBlob
# ----------------------------


class TextBlobAnalyzer:
    def __init__(self, cfg: SentimentConfig):
        self.cfg = cfg
        # nothing to preload

    def _label(self, txt: str) -> str:
        from textblob import TextBlob

        pol = TextBlob(txt).sentiment.polarity
        if pol > self.cfg.blob_pos:
            return "positive"
        if pol < self.cfg.blob_neg:
            return "negative"
        return "neutral"

    def score_series(self, texts: pd.Series) -> pd.Series:
        s = _as_text_series(texts)
        return s.map(self._label)


# ----------------------------
# BERT (HF pipeline, batched)
# ----------------------------


class BertAnalyzer:
    def __init__(self, cfg: SentimentConfig):
        self.cfg = cfg
        self._pipe = None  # lazy

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        from transformers import pipeline

        # device=None => CPU; pass int for GPU
        self._pipe = pipeline(
            "sentiment-analysis",
            model=self.cfg.bert_model,
            device=self.cfg.bert_device if self.cfg.bert_device is not None else -1,
        )

    def _label_from_hf(self, label: str, score: float) -> str:
        # HF returns POSITIVE/NEGATIVE, add neutral band around 0.5 if desired
        # score is prob of predicted class; we only know argmax class here.
        # Simple band: if score within [0.5 - band, 0.5 + band] => neutral.
        band = self.cfg.bert_neutral_band
        if abs(score - 0.5) <= band:
            return "neutral"
        return label.lower()  # "positive" | "negative"

    def score_series(self, texts: pd.Series) -> pd.Series:
        self._ensure_pipe()
        s = _as_text_series(texts)

        # Batch the list for better throughput
        results: List[str] = []
        batch: List[str] = []
        idx_chunks: List[List[int]] = []
        cur_idx_chunk: List[int] = []

        for i, txt in enumerate(s.tolist()):
            # truncate per config
            batch.append(txt[: self.cfg.bert_max_length])
            cur_idx_chunk.append(i)
            if len(batch) >= self.cfg.bert_batch_size:
                outs = self._pipe(
                    batch,
                    truncation=True,
                    max_length=self.cfg.bert_max_length,
                )
                results.extend(
                    [
                        self._label_from_hf(o["label"], float(o.get("score", 1.0)))
                        for o in outs
                    ]
                )
                batch = []
                idx_chunks.append(cur_idx_chunk)
                cur_idx_chunk = []

        if batch:
            outs = self._pipe(
                batch,
                truncation=True,
                max_length=self.cfg.bert_max_length,
            )
            results.extend(
                [
                    self._label_from_hf(o["label"], float(o.get("score", 1.0)))
                    for o in outs
                ]
            )
            idx_chunks.append(cur_idx_chunk)

        # Rebuild a Series with original index order
        ser = pd.Series(results, index=pd.RangeIndex(len(results)))
        ser.index = s.index  # align to original
        return ser


# ----------------------------
# Factory with caching
# ----------------------------

_analyzer_cache: Dict[str, SentimentAnalyzer] = {}


def analyzer_for(method_or_cfg: Union[str, SentimentConfig]) -> SentimentAnalyzer:
    if isinstance(method_or_cfg, str):
        cfg = SentimentConfig(method=method_or_cfg.lower())
    else:
        cfg = method_or_cfg

    key = f"{cfg.method}|{cfg.bert_model}|{cfg.bert_device}|{cfg.bert_batch_size}|{cfg.bert_max_length}|{cfg.bert_neutral_band}|{cfg.vader_pos}|{cfg.vader_neg}|{cfg.blob_pos}|{cfg.blob_neg}"
    if key in _analyzer_cache:
        return _analyzer_cache[key]

    m = cfg.method.lower()
    if m == "vader":
        inst: SentimentAnalyzer = VaderAnalyzer(cfg)
    elif m == "textblob":
        inst = TextBlobAnalyzer(cfg)
    elif m == "bert":
        inst = BertAnalyzer(cfg)
    else:
        raise ValueError(f"Unsupported sentiment method: {cfg.method}")

    _analyzer_cache[key] = inst
    return inst
