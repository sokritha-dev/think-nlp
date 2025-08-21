# app/api/topic.py

from datetime import datetime, timezone
import json
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression
from app.core.file_handler.storage import S3Storage
from app.core.topic_labeling.config import TopicLabelConfig
from app.core.topic_labeling.labelers import (
    DefaultHeuristicLabeler,
    ExplicitLabeler,
    SBertKeywordLabeler,
)
from app.core.topic_modeling.config import TopicEstimationConfig, TopicModelConfig
from app.core.topic_modeling.gensim_lda import GensimLDAModeler
from app.messages.clean_messages import FILE_NOT_FOUND
from app.models.db.file_record import FileRecord
from app.models.db.topic_model import TopicModel
from app.schemas.topic import (
    LDATopicRequest,
    LDATopicResponse,
    TopicLabelRequest,
    TopicLabelResponse,
    TopicLabelResponseData,
)
from app.services.file_service import FileService
from app.services.topic_labeling_service import TopicLabelingService
from app.services.topic_modeling_service import TopicModelingService
from app.utils.exceptions import NotFoundError, BadRequestError, ServerError
from app.utils.response_builder import success_response
from app.messages.topic_messages import (
    LDA_COMPLETED,
    LDA_UP_TO_DATE,
    LEMMATIZED_FILE_NOT_FOUND,
    TOPIC_LABELING_COMPLETED,
    TOPIC_MODEL_NOT_FOUND,
)

router = APIRouter(prefix="/api/topic", tags=["Topic Modeling"])
logger = logging.getLogger(__name__)


def _files() -> FileService:
    """Shared FileService (S3 + CSV + Gzip)."""
    return FileService(
        storage=S3Storage(),
        codec=CsvCodec(index=False),
        compression=GzipCompression(),
    )


@router.post("/lda", response_model=LDATopicResponse)
async def run_lda_topic_modeling(
    req: LDATopicRequest, db: AsyncSession = Depends(get_db)
):
    """
    Run (or reuse) LDA topic modeling.
    - If req.num_topics is provided, we force that K.
    - Otherwise we estimate K using the estimator (default: coherence).
    - Results are cached/recomputed when the underlying lemmatized CSV hasn't changed
      and K matches the existing TopicModel.
    """
    try:
        # 1) Load file record + guard
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)
        if not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND", message=LEMMATIZED_FILE_NOT_FOUND
            )

        # 2) Build modeler/estimator and service
        files = _files()
        tm_cfg = TopicModelConfig(
            backend="gensim",
            passes=10,
            topn_words=10,  # tweak as you wish
        )
        # If client didn’t force K, we allow estimation (narrow bounds optional)
        est_cfg = (
            TopicEstimationConfig(method="coherence", min_k=3, max_k=10)
            if not req.num_topics
            else None
        )

        modeler = GensimLDAModeler(tm_cfg)
        estimator = modeler  # GensimLDAModeler implements both interfaces

        tm_service = TopicModelingService(
            files=files, modeler=modeler, estimator=estimator
        )

        # 3) Service decides reuse vs recompute, persists/updates TopicModel row,
        #    uploads CSV, and returns a uniform result object.
        tm_res = await tm_service.ensure_topics(
            db=db,
            record=record,
            df_lemm=None,  # let the service load lemmatized CSV via FileService
            cfg=tm_cfg,
            est=est_cfg,
            force_k=req.num_topics,  # None => estimator picks
        )

        return success_response(
            message=LDA_UP_TO_DATE if tm_res.recomputed else LDA_COMPLETED,
            data={
                "file_id": record.id,
                "topic_model_id": tm_res.topic_model_id,
                "lda_topics_s3_url": tm_res.url,
                "topics": tm_res.topics,  # [{topic_id, keywords, label?...}]
                "topic_count": tm_res.num_topics,
                "recomputed": tm_res.recomputed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    except (NotFoundError, BadRequestError):
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected LDA error: {e}")
        raise ServerError(code="LDA_FAILED", message="LDA topic modeling failed.")


@router.get("/lda")
async def get_lda_info(
    file_id: str = Query(...),
    num_topics: Optional[int] = Query(None, ge=1),
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            (await db.execute(select(FileRecord).filter_by(id=file_id)))
            .scalars()
            .first()
        )
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        # Build query incrementally so we don't filter by topic_count=None by accident
        q = select(TopicModel).where(
            TopicModel.file_id == file_id,
            TopicModel.method == "LDA",
        )
        if num_topics is not None:
            q = q.where(TopicModel.topic_count == num_topics)

        # Deterministic “latest” ordering + limit 1
        q = q.order_by(
            TopicModel.label_updated_at.desc().nullslast(),
            TopicModel.topic_updated_at.desc().nullslast(),
            TopicModel.created_at.desc().nullslast(),
            TopicModel.id.desc(),  # tie-breaker
        ).limit(1)

        existing = (await db.execute(q)).scalars().first()
        if not existing:
            raise NotFoundError(
                code="LDA_NOT_FOUND",
                message="No LDA topic model found for this file.",
            )

        return success_response(
            message="LDA topic info loaded successfully.",
            data={
                "file_id": file_id,
                "topic_model_id": existing.id,
                "topic_count": existing.topic_count,
                "lda_topics_s3_url": existing.s3_url,
                "topics": json.loads(existing.summary_json or "[]"),
                "topic_updated_at": existing.topic_updated_at.isoformat()
                if existing.topic_updated_at
                else None,
                "label_updated_at": existing.label_updated_at.isoformat()
                if existing.label_updated_at
                else None,
            },
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to fetch LDA info: {e}")
        raise ServerError(code="LDA_INFO_FAILED", message="Failed to load LDA info.")


@router.post("/label", response_model=TopicLabelResponse)
async def label_topics(req: TopicLabelRequest, db: AsyncSession = Depends(get_db)):
    """
    Apply topic labels to an existing TopicModel CSV on S3.

    Inference rules (mutually exclusive inputs):
      - If `label_map` is provided => Explicit labeling
      - Else if `keywords` is provided => SBERT keyword matching
      - Else => Default heuristic from top-k topic keywords
    """
    try:
        # 1) Load TopicModel
        topic_model = (
            await db.execute(select(TopicModel).filter_by(id=req.topic_model_id))
        ).scalar_one_or_none()
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        # 2) Validate inputs (can't provide both)
        if req.label_map and req.keywords:
            raise BadRequestError(
                code="MUTUALLY_EXCLUSIVE",
                message="Provide either `label_map` or `keywords`, not both.",
            )

        files = _files()

        # 3) Pick labeler + config
        if req.label_map:
            cfg = TopicLabelConfig(strategy="explicit")
            labeler = ExplicitLabeler(cfg)
            explicit_map = {int(k): v for k, v in req.label_map.items()}
            user_keywords = None
        elif req.keywords:
            cfg = TopicLabelConfig(strategy="keywords", model_name="all-MiniLM-L6-v2")
            labeler = SBertKeywordLabeler(cfg)
            explicit_map = None
            user_keywords = req.keywords
        else:
            cfg = TopicLabelConfig(
                strategy="default",
                num_keywords=topic_model.topic_count,
            )
            labeler = DefaultHeuristicLabeler(cfg)
            explicit_map = None
            user_keywords = None

        label_svc = TopicLabelingService(files=files, labeler=labeler)

        # 4) Service handles:
        #    - reuse if fresh & inputs unchanged
        #    - otherwise label, upload *_labeled.csv.gz, update TopicModel row
        res = await label_svc.ensure_labeled(
            db=db,
            topic_model=topic_model,
            df_topics=None,  # let service load via FileService
            explicit_map=explicit_map,  # only for explicit mode
            user_keywords=user_keywords,  # only for keyword mode
        )

        # 5) Response
        return success_response(
            message=("Labels already exist and inputs haven't changed.")
            if res.reused
            else TOPIC_LABELING_COMPLETED,
            data=TopicLabelResponseData(
                topic_model_id=topic_model.id,
                labeled_s3_url=res.url,
                columns=res.df.columns.tolist(),
                record_count=len(res.df),
                topics=res.topics,  # enriched summary (label/confidence/matched_with)
            ),
        )

    except (NotFoundError, BadRequestError):
        raise
    except Exception as e:
        logger.exception(f"❌ Topic labeling failed: {e}")
        raise ServerError(
            code="TOPIC_LABELING_FAILED",
            message="Topic labeling failed.",
        )


@router.get("/label", response_model=TopicLabelResponse)
async def get_existing_topic_labels(
    topic_model_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the current labels & metadata for a TopicModel without recomputing.
    """
    try:
        topic_model = (
            await db.execute(select(TopicModel).filter_by(id=topic_model_id))
        ).scalar_one_or_none()
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        topic_summary = json.loads(topic_model.summary_json or "[]")

        return success_response(
            message="Topic labels retrieved successfully.",
            data=TopicLabelResponseData(
                topic_model_id=topic_model.id,
                labeled_s3_url=topic_model.s3_url,
                columns=[
                    "stopword_removed",
                    "lemmatized_tokens",
                    "topic_id",
                    "topic_label",
                ],
                record_count=None,
                topics=topic_summary,
            ),
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to fetch topic labels: {e}")
        raise ServerError(
            code="TOPIC_LABEL_FETCH_FAILED",
            message="Failed to retrieve topic labeling result.",
        )
