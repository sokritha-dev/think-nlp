from datetime import datetime
import gzip
import json
import logging
import pandas as pd
from io import BytesIO

from sqlalchemy import select

from app.models.db.sentiment_analysis import SentimentAnalysis
from app.models.db.topic_model import TopicModel
from app.services.s3_uploader import download_file_from_s3
from app.services.sentiment_analysis import (
    analyze_sentiment_bert,
    analyze_sentiment_textblob,
    analyze_sentiment_vader,
)

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


async def run_sentiment_background(db, topic_model_id: str, method: str):
    sentiment_entry = None

    try:
        # Get latest pending/processing SentimentAnalysis record
        result = await db.execute(
            select(SentimentAnalysis)
            .filter_by(topic_model_id=topic_model_id, method=method)
            .order_by(SentimentAnalysis.updated_at.desc())
        )
        sentiment_entry = result.scalars().first()
        if not sentiment_entry:
            logger.warning(
                f"SentimentAnalysis row not found for: {topic_model_id}, {method}"
            )
            return

        # Mark as processing
        sentiment_entry.status = "processing"
        sentiment_entry.updated_at = datetime.now()
        await db.commit()

        # Download topic file
        topic_model_result = await db.execute(
            select(TopicModel).filter_by(id=topic_model_id)
        )
        topic_model = topic_model_result.scalars().first()
        if not topic_model:
            logger.error(
                f"TopicModel {topic_model_id} unexpectedly missing at task time."
            )
            sentiment_entry.status = "failed"
            await db.commit()
            return

        file_bytes = await download_file_from_s3(topic_model.s3_key)
        if topic_model.s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        if not all(
            col in df.columns
            for col in ["lemmatized_tokens", "topic_id", "topic_label"]
        ):
            sentiment_entry.status = "failed"
            await db.commit()
            logger.error("Required columns missing in topic file.")
            return

        # Prepare sentiment
        df["text"] = df["lemmatized_tokens"].apply(
            lambda x: " ".join(eval(x)) if isinstance(x, str) else str(x)
        )
        df["sentiment"] = df["text"].apply(lambda x: classify_sentiment(x, method))

        # Aggregate
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
                {
                    "label": topic_id,
                    "positive": round(
                        (group["sentiment"] == "positive").sum() / count * 100, 1
                    ),
                    "neutral": round(
                        (group["sentiment"] == "neutral").sum() / count * 100, 1
                    ),
                    "negative": round(
                        (group["sentiment"] == "negative").sum() / count * 100, 1
                    ),
                    "keywords": keywords,
                }
            )

        # ✅ Final update
        sentiment_entry.status = "done"
        sentiment_entry.updated_at = datetime.now()
        sentiment_entry.per_topic_json = json.dumps(topic_stats)
        sentiment_entry.overall_positive = overall["positive"]
        sentiment_entry.overall_neutral = overall["neutral"]
        sentiment_entry.overall_negative = overall["negative"]
        await db.commit()

    except Exception as e:
        logger.exception(f"Sentiment background task failed: {e}")
        if sentiment_entry:
            sentiment_entry.status = "failed"
            await db.commit()
