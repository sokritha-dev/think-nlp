import json
from uuid import uuid4
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
import logging

from app.core.database import get_db
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
            raise HTTPException(status_code=404, detail="Topic model not found")

        # ✅ Check for cached result
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
            logger.info("⏩ Skipping sentiment computation - result still valid.")
            return SentimentResponse(
                status="success",
                message=f"Sentiment already computed using {req.method}.",
                data=json.loads(existing.result_json),
            )

        # 🧠 Recompute sentiment
        file_bytes = download_file_from_s3(topic_model.s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        if (
            "lemmatized_tokens" not in df.columns
            or "topic_id" not in df.columns
            or "topic_label" not in df.columns
        ):
            raise HTTPException(
                status_code=400,
                detail="Required columns missing in topic-labeled file",
            )

        df["text"] = df["lemmatized_tokens"].apply(
            lambda x: " ".join(eval(x)) if isinstance(x, str) else str(x)
        )
        df["sentiment"] = df["text"].apply(
            lambda x: classify_sentiment(x, req.method.lower())
        )

        # 📊 Calculate sentiment
        sentiment_counts = df["sentiment"].value_counts().to_dict()
        total = len(df)
        overall = {
            "positive": round((sentiment_counts.get("positive", 0) / total) * 100, 1),
            "neutral": round((sentiment_counts.get("neutral", 0) / total) * 100, 1),
            "negative": round((sentiment_counts.get("negative", 0) / total) * 100, 1),
        }

        topic_stats = []
        for topic_id, group in df.groupby("topic_label"):
            count = len(group)
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
                )
            )

        # 💾 Save to DB
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

        return SentimentResponse(
            status="success",
            message=f"Sentiment analysis completed using {req.method}.",
            data=summary_result,
        )

    except Exception as e:
        logger.exception(f"❌ Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")
