from datetime import datetime
import json
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks, Depends, Query
import logging

from app.core.database import get_db
from app.messages.analysis_messages import (
    SENTIMENT_ANALYSIS_ALREADY_EXISTS,
)
from app.messages.topic_messages import TOPIC_MODEL_NOT_FOUND
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.models.db.topic_model import TopicModel
from app.services.background_run_sentiment import run_sentiment_background
from app.schemas.sentiment import (
    SentimentRequest,
    SentimentResponse,
    SentimentChartData,
)

from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/sentiment", tags=["Sentiment"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=SentimentResponse)
async def analyze_sentiment(
    req: SentimentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    try:
        # 1. Validate topic model
        result = await db.execute(select(TopicModel).filter_by(id=req.topic_model_id))
        topic_model = result.scalars().first()
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        # 2. Check for existing sentiment record
        result = await db.execute(
            select(SentimentAnalysis)
            .filter_by(topic_model_id=req.topic_model_id, method=req.method)
            .order_by(SentimentAnalysis.updated_at.desc())
        )
        existing = result.scalars().first()

        if (
            existing
            and existing.status == "done"
            and existing.updated_at >= topic_model.label_updated_at
        ):
            chart = SentimentChartData(
                overall={
                    "positive": existing.overall_positive,
                    "neutral": existing.overall_neutral,
                    "negative": existing.overall_negative,
                },
                per_topic=json.loads(existing.per_topic_json),
                should_recompute=False,
            ).model_dump()

            return success_response(
                message=SENTIMENT_ANALYSIS_ALREADY_EXISTS,
                data={"status": existing.status, **chart},
            )

        # 3. Create new pending entry if none exists or previous failed/outdated
        new_entry = SentimentAnalysis(
            id=str(uuid4()),
            topic_model_id=req.topic_model_id,
            method=req.method,
            status="pending",
            updated_at=datetime.now(),
        )
        db.add(new_entry)
        await db.commit()

        # 4. Launch background task
        background_tasks.add_task(
            run_sentiment_background, req.topic_model_id, req.method.lower()
        )

        return success_response(
            message="Sentiment analysis started in background.",
            data={"status": "pending"},
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Sentiment analysis POST failed: {e}")
        raise ServerError(
            code="SENTIMENT_ANALYSIS_FAILED",
            message="An unexpected error occurred during sentiment initialization.",
        )


@router.get("/", response_model=SentimentResponse)
async def get_sentiment_result(
    topic_model_id: str = Query(...),
    method: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await db.execute(select(TopicModel).filter_by(id=topic_model_id))
        topic_model = result.scalars().first()

        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        stmt = (
            select(SentimentAnalysis)
            .where(SentimentAnalysis.topic_model_id == topic_model_id)
            .where(SentimentAnalysis.method == method)
            .order_by(SentimentAnalysis.updated_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        existing = result.scalars().first()

        if not existing:
            raise NotFoundError(
                code="SENTIMENT_ANALYSIS_NOT_FOUND",
                message="Sentiment Analysis not found.",
            )

        should_recompute = topic_model.label_updated_at > existing.updated_at
        per_topic = []
        if existing.per_topic_json:
            per_topic = json.loads(existing.per_topic_json)

        summary_result = SentimentChartData(
            overall={
                "positive": existing.overall_positive or 0.0,
                "neutral": existing.overall_neutral or 0.0,
                "negative": existing.overall_negative or 0.0,
            },
            per_topic=per_topic,
            should_recompute=should_recompute,
        ).model_dump()

        return success_response(
            message=SENTIMENT_ANALYSIS_ALREADY_EXISTS,
            data={"status": existing.status, **summary_result},
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Sentiment lookup failed: {e}")
        raise ServerError(
            code="SENTIMENT_LOOKUP_FAILED", message="Failed to get sentiment analysis."
        )
