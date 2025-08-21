from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4
import pandas as pd

from app.models.db.file_record import FileRecord
from app.models.db.topic_model import TopicModel
from app.services.file_service import FileService
from app.core.topic_modeling.base import TopicModeler, TopicEstimator
from app.core.topic_modeling.config import TopicModelConfig, TopicEstimationConfig


@dataclass(frozen=True)
class TopicModelResult:
    key: str
    url: str
    df: pd.DataFrame  # input df + ['topic_id','topic_label']
    topics: list[dict]  # [{topic_id, keywords, label}, ...]
    num_topics: int
    topic_model_id: str
    recomputed: bool


class TopicModelingService:
    """
    - Accepts in-memory lemmatized df (pipeline) or loads from S3 (endpoint)
    - Estimates K if not provided (backend-appropriate)
    - Writes labeled CSV to S3: 'lda/lda_topics_*.csv.gz'
    - Upserts TopicModel row (method='LDA') in your DB
    - Reuses prior model if fresh vs lemmatized and config matches (optional)
    """

    def __init__(
        self, files: FileService, modeler: TopicModeler, estimator: TopicEstimator
    ):
        self.files = files
        self.modeler = modeler
        self.estimator = estimator

    async def ensure_topics(
        self,
        db,
        record: FileRecord,
        *,
        df_lemm: Optional[pd.DataFrame] = None,
        cfg: TopicModelConfig,
        est: TopicEstimationConfig,
        force_k: Optional[int] = None,
    ) -> TopicModelResult:
        # load df if not provided
        if df_lemm is None:
            if not record.lemmatized_s3_key:
                raise FileNotFoundError("Lemmatized CSV not found.")
            df_lemm = await self.files.download_df(record.lemmatized_s3_key)

        # choose token column
        col = "lemmatized_tokens"
        if col not in df_lemm.columns:
            raise KeyError(f"Column '{col}' not found.")

        # estimate K if needed
        if force_k is None:
            k = self.estimator.estimate_k(df_lemm[col], est.min_k, est.max_k)
        else:
            k = force_k

        # fit + predict
        topic_ids, topics = self.modeler.fit_predict(df_lemm[col], num_topics=k)

        # attach topics
        out = df_lemm.copy()
        out["topic_id"] = topic_ids
        # optional: labeling can be improved; keeping simple
        id_to_label = {int(t["topic_id"]): t["label"] for t in topics}
        out["topic_label"] = out["topic_id"].map(id_to_label)

        # upload
        key = f"lda/lda_topics_{datetime.now(timezone.utc)}.csv.gz"
        key, url = await self.files.upload_df(out, key)

        # upsert TopicModel
        tm = TopicModel(
            id=str(uuid4()),
            file_id=record.id,
            method="LDA",
            topic_count=k,
            s3_key=key,
            s3_url=url,
            summary_json=json.dumps(topics),
            label_map_json=json.dumps(id_to_label),
            label_keywords=json.dumps([]),
            topic_updated_at=datetime.now(timezone.utc),
            label_updated_at=datetime.now(timezone.utc),
        )
        db.add(tm)
        await db.commit()

        return TopicModelResult(
            key=key,
            url=url,
            df=out,
            topics=topics,
            num_topics=k,
            topic_model_id=tm.id,
            recomputed=True,
        )
