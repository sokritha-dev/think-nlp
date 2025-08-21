from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List

import pandas as pd

from app.models.db.topic_model import TopicModel
from app.services.file_service import FileService
from app.core.topic_labeling.base import TopicLabeler


@dataclass(frozen=True)
class TopicLabelResult:
    key: str
    url: str
    df: pd.DataFrame
    topics: List[dict]  # enriched topics with label
    label_map: Dict[int, str]
    reused: bool


class TopicLabelingService:
    """
    - Reuse if label is fresh vs topic model and inputs unchanged
    - Load topic CSV via FileService unless df_topics provided
    - Write back *_labeled.csv.gz; update TopicModel row
    """

    def __init__(self, files: FileService, labeler: TopicLabeler):
        self.files = files
        self.labeler = labeler

    def _unchanged_inputs(self, tm: TopicModel, *, explicit_map, user_keywords) -> bool:
        prev_kw = json.loads(tm.label_keywords or "[]")
        prev_map = json.loads(tm.label_map_json or "{}")
        return (explicit_map or {}) == prev_map and (user_keywords or []) == (
            prev_kw or []
        )

    async def ensure_labeled(
        self,
        db,
        topic_model: TopicModel,
        *,
        df_topics: Optional[pd.DataFrame] = None,
        explicit_map: Optional[Dict[int, str]] = None,
        user_keywords: Optional[List[str]] = None,
    ) -> TopicLabelResult:
        # Reuse: unchanged inputs and fresher than topic_updated_at
        if (
            topic_model.label_updated_at
            and topic_model.topic_updated_at
            and topic_model.label_updated_at >= topic_model.topic_updated_at
            and self._unchanged_inputs(
                topic_model, explicit_map=explicit_map, user_keywords=user_keywords
            )
        ):
            # no file I/O needed
            return TopicLabelResult(
                key=topic_model.s3_key,
                url=topic_model.s3_url,
                df=(
                    df_topics
                    if df_topics is not None
                    else await self.files.download_df(topic_model.s3_key)
                ),
                topics=json.loads(topic_model.summary_json),
                label_map=json.loads(topic_model.label_map_json or "{}"),
                reused=True,
            )

        # Prepare topics (from DB summary_json)
        topics = json.loads(topic_model.summary_json or "[]")

        # Compute labeling via adapter
        label_map, enriched = self.labeler.label(
            topics, explicit_map=explicit_map, user_keywords=user_keywords
        )

        # Load CSV if not provided
        if df_topics is None:
            df_topics = await self.files.download_df(topic_model.s3_key)

        # Apply labels to DataFrame
        df_topics = df_topics.copy()
        df_topics["topic_label"] = df_topics["topic_id"].apply(
            lambda x: label_map.get(int(x), f"Topic {x}")
        )

        # Upload labeled CSV (suffix)
        base = (
            topic_model.s3_key.replace("_labeled", "")
            .replace(".csv.gz", "")
            .replace(".csv", "")
        )
        new_key = f"{base}_labeled.csv.gz"
        key, url = await self.files.upload_df(df_topics, new_key)

        # Persist TopicModel row
        topic_model.s3_key = key
        topic_model.s3_url = url
        topic_model.summary_json = json.dumps(enriched)
        topic_model.label_keywords = json.dumps(user_keywords or [])
        topic_model.label_map_json = json.dumps(label_map or {})
        topic_model.label_updated_at = datetime.now(timezone.utc)
        await db.commit()

        return TopicLabelResult(
            key=key,
            url=url,
            df=df_topics,
            topics=enriched,
            label_map=label_map,
            reused=False,
        )
