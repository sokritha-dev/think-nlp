from __future__ import annotations
import ast
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.sentiment.analyzer import SentimentAnalyzer
from app.core.sentiment.config import SentimentConfig
from app.models.db.topic_model import TopicModel
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.schemas.sentiment import SentimentTopicBreakdown
from app.services.file_service import FileService


def _safe_pct(n: int, d: int) -> float:
    if not d:
        return 0.0
    v = (n / d) * 100.0
    return 0.0 if not math.isfinite(v) else round(v, 1)


def _tokens_to_text(series: pd.Series) -> pd.Series:
    """Join token lists to text; if value is a stringified list, parse first."""

    def conv(x):
        if isinstance(x, list):
            return " ".join(map(str, x))
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return " ".join(map(str, val))
            except Exception:
                pass
            return x
        return ""

    return series.apply(conv)


@dataclass
class SentimentResult:
    overall: Dict[str, float]
    per_topic: List[SentimentTopicBreakdown]
    reused: bool


class SentimentService:
    """
    Reuses existing SentimentAnalysis if it's fresher than TopicModel.label_updated_at.
    Otherwise computes from labeled DF (downloaded via FileService, or you can pass one).
    """

    def __init__(
        self, files: FileService, analyzer: SentimentAnalyzer, cfg: SentimentConfig
    ):
        self.files = files
        self.analyzer = analyzer
        self.cfg = cfg

    async def ensure_sentiment(
        self,
        db: AsyncSession,
        *,
        topic_model: TopicModel,
        df_labeled: Optional[pd.DataFrame] = None,
    ) -> SentimentResult:
        # 1) try reuse
        existing = (
            (
                await db.execute(
                    select(SentimentAnalysis)
                    .where(SentimentAnalysis.topic_model_id == topic_model.id)
                    .where(SentimentAnalysis.method == self.cfg.method)
                    .order_by(SentimentAnalysis.updated_at.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

        if (
            existing
            and existing.status == "done"
            and topic_model.label_updated_at
            and existing.updated_at
            and self._as_utc(existing.updated_at)
            >= self._as_utc(topic_model.label_updated_at)
        ):
            return SentimentResult(
                overall={
                    "positive": float(existing.overall_positive or 0.0),
                    "neutral": float(existing.overall_neutral or 0.0),
                    "negative": float(existing.overall_negative or 0.0),
                },
                per_topic=[
                    SentimentTopicBreakdown(**t)
                    for t in json.loads(existing.per_topic_json or "[]")
                ],
                reused=True,
            )

        # 2) load DF if needed
        if df_labeled is None:
            df_labeled = await self.files.download_df(topic_model.s3_key)

        # 3) build text & score
        tcol = self.cfg.text_column
        if tcol not in df_labeled.columns:
            raise ValueError(f"Column '{tcol}' not found in labeled DataFrame.")

        texts = _tokens_to_text(df_labeled[tcol])
        sentiments = self.analyzer.score_series(texts)
        df = df_labeled.copy()
        df["sentiment"] = sentiments

        # 4) overall
        counts = df["sentiment"].value_counts().to_dict()
        total = int(len(df))
        overall = {
            "positive": _safe_pct(int(counts.get("positive", 0)), total),
            "neutral": _safe_pct(int(counts.get("neutral", 0)), total),
            "negative": _safe_pct(int(counts.get("negative", 0)), total),
        }

        # 5) per-topic (group by topic_id; enrich with label/keywords from summary)
        id_col = self.cfg.topic_id_col
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in labeled DataFrame.")

        enriched = json.loads(topic_model.summary_json or "[]")
        meta: Dict[int, Tuple[str, List[str]]] = {}
        for t in enriched:
            tid = int(t["topic_id"])
            label = t.get("label") or f"Topic {tid}"
            kws = t.get("keywords", [])
            if isinstance(kws, str):
                kws = [k.strip() for k in kws.split(",") if k.strip()]
            meta[tid] = (label, kws)

        per_topic: List[SentimentTopicBreakdown] = []
        topic_ids = sorted({int(t["topic_id"]) for t in enriched})
        for tid in topic_ids:
            group = df[df[id_col] == tid]
            n = int(len(group))
            pos = int((group["sentiment"] == "positive").sum())
            neu = int((group["sentiment"] == "neutral").sum())
            neg = int((group["sentiment"] == "negative").sum())
            label, keywords = meta.get(tid, (f"Topic {tid}", []))
            per_topic.append(
                SentimentTopicBreakdown(
                    label=label,
                    keywords=keywords,
                    positive=_safe_pct(pos, n),
                    neutral=_safe_pct(neu, n),
                    negative=_safe_pct(neg, n),
                )
            )

        # 6) upsert SentimentAnalysis
        row = existing or SentimentAnalysis(
            id=str(uuid4()),
            topic_model_id=topic_model.id,
            method=self.cfg.method,
        )
        row.status = "done"
        row.updated_at = datetime.now(timezone.utc)
        row.overall_positive = overall["positive"]
        row.overall_neutral = overall["neutral"]
        row.overall_negative = overall["negative"]
        row.per_topic_json = json.dumps([t.model_dump() for t in per_topic])

        if existing is None:
            db.add(row)
        await db.commit()

        return SentimentResult(overall=overall, per_topic=per_topic, reused=False)

    @staticmethod
    def _as_utc(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
