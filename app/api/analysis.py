from datetime import datetime
import json
from uuid import uuid4
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
import logging

from app.core.database import get_db
from app.messages.analysis_messages import (
    SENTIMENT_ANALYSIS_ALREADY_EXISTS,
    SENTIMENT_ANALYSIS_SUCCESS,
    TOPIC_FILE_INVALID,
)
from app.messages.topic_messages import TOPIC_MODEL_NOT_FOUND
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.models.db.topic_model import TopicModel
from app.services.s3_uploader import download_file_from_s3
from app.schemas.sentiment import (
    SentimentRequest,
    SentimentResponse,
    SentimentChartData,
    SentimentTopicBreakdown,
)
from app.services.sentiment_analysis import (
    analyze_sentiment_bert,
    analyze_sentiment_textblob,
    analyze_sentiment_vader,
)
from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError

router = APIRouter(prefix="/api/sentiment", tags=["Sentiment"])
logger = logging.getLogger(__name__)


def classify_sentiment(text: str, method: str) -> str:
    if method == "vader":
        return analyze_sentiment_vader(text)
    elif method == "textblob":
        return analyze_sentiment_textblob(text)
    elif method == "bert":
        return analyze_sentiment_bert(text)
    else:
        raise ValueError("Unsupported method")


@router.post("/", response_model=SentimentResponse)
async def analyze_sentiment(req: SentimentRequest, db: Session = Depends(get_db)):
    try:
        topic_model = db.query(TopicModel).filter_by(id=req.topic_model_id).first()
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        existing = (
            db.query(SentimentAnalysis)
            .filter_by(topic_model_id=req.topic_model_id, method=req.method)
            .order_by(SentimentAnalysis.updated_at.desc())
            .first()
        )
        if (
            existing
            and topic_model.updated_at
            and existing.updated_at >= topic_model.updated_at
        ):
            return success_response(
                message=SENTIMENT_ANALYSIS_ALREADY_EXISTS,
                data=json.loads(existing.per_topic_json),
            )

        file_bytes = download_file_from_s3(topic_model.s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        if not all(
            col in df.columns
            for col in ["lemmatized_tokens", "topic_id", "topic_label"]
        ):
            raise ServerError(code="TOPIC_FILE_INVALID", message=TOPIC_FILE_INVALID)

        df["text"] = df["lemmatized_tokens"].apply(
            lambda x: " ".join(eval(x)) if isinstance(x, str) else str(x)
        )
        df["sentiment"] = df["text"].apply(
            lambda x: classify_sentiment(x, req.method.lower())
        )

        sentiment_counts = df["sentiment"].value_counts().to_dict()
        total = len(df)
        overall = {
            "positive": round((sentiment_counts.get("positive", 0) / total) * 100, 1),
            "neutral": round((sentiment_counts.get("neutral", 0) / total) * 100, 1),
            "negative": round((sentiment_counts.get("negative", 0) / total) * 100, 1),
        }

        topic_summary = json.loads(topic_model.summary_json)
        topic_map = {int(t["topic_id"]): t for t in topic_summary}

        topic_stats = []
        for topic_id, group in df.groupby("topic_label"):
            count = len(group)
            tid = int(group["topic_id"].iloc[0])
            keywords = topic_map.get(tid, {}).get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]

            topic_stats.append(
                SentimentTopicBreakdown(
                    label=topic_id,
                    positive=round(
                        (group["sentiment"] == "positive").sum() / count * 100, 1
                    ),
                    neutral=round(
                        (group["sentiment"] == "neutral").sum() / count * 100, 1
                    ),
                    negative=round(
                        (group["sentiment"] == "negative").sum() / count * 100, 1
                    ),
                    keywords=keywords,
                )
            )

        summary_result = SentimentChartData(
            overall=overall, per_topic=topic_stats
        ).model_dump()

        entry = SentimentAnalysis(
            id=str(uuid4()),
            topic_model_id=req.topic_model_id,
            method=req.method,
            overall_positive=overall["positive"],
            overall_neutral=overall["neutral"],
            overall_negative=overall["negative"],
            per_topic_json=json.dumps([t.model_dump() for t in topic_stats]),
            updated_at=datetime.now(),
        )
        db.add(entry)
        db.commit()

        return success_response(message=SENTIMENT_ANALYSIS_SUCCESS, data=summary_result)

    except NotFoundError as e:
        raise e
    except ServerError as e:
        raise e
    except Exception as e:
        logger.exception(f"Sentiment analysis failed: {e}")
        raise ServerError(
            code="SENTIMENT_ANALYSIS_FAILED", message="Sentiment analysis failed."
        )
