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
from app.core.database import async_session  # ✅ Adjust import as needed

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


async def run_sentiment_background(_, topic_model_id: str, method: str):
    try:
        # === Step 1: Retrieve data from DB ===
        async with async_session() as db:
            result = await db.execute(
                select(SentimentAnalysis)
                .filter_by(topic_model_id=topic_model_id, method=method)
                .order_by(SentimentAnalysis.updated_at.desc())
            )
            sentiment_entry = result.scalars().first()
            if not sentiment_entry:
                logger.warning(
                    f"No SentimentAnalysis row for {topic_model_id}, {method}"
                )
                return

            sentiment_entry.status = "processing"
            sentiment_entry.updated_at = datetime.now()
            await db.commit()

            topic_model_result = await db.execute(
                select(TopicModel).filter_by(id=topic_model_id)
            )
            topic_model = topic_model_result.scalars().first()
            if not topic_model:
                sentiment_entry.status = "failed"
                await db.commit()
                logger.error(f"TopicModel {topic_model_id} not found")
                return

        # === Step 2: Heavy sentiment processing without DB ===
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
            logger.error("Missing required columns in topic file")
            async with async_session() as db:
                result = await db.execute(
                    select(SentimentAnalysis)
                    .filter_by(topic_model_id=topic_model_id, method=method)
                    .order_by(SentimentAnalysis.updated_at.desc())
                )
                sentiment_entry = result.scalars().first()
                if sentiment_entry:
                    sentiment_entry.status = "failed"
                    await db.commit()
            return

        df["text"] = df["lemmatized_tokens"].apply(
            lambda x: " ".join(eval(x)) if isinstance(x, str) else str(x)
        )
        df["sentiment"] = df["text"].apply(lambda x: classify_sentiment(x, method))

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

        # === Step 3: Save results back to DB ===
        async with async_session() as db:
            result = await db.execute(
                select(SentimentAnalysis)
                .filter_by(topic_model_id=topic_model_id, method=method)
                .order_by(SentimentAnalysis.updated_at.desc())
            )
            sentiment_entry = result.scalars().first()
            if sentiment_entry:
                sentiment_entry.status = "done"
                sentiment_entry.updated_at = datetime.now()
                sentiment_entry.per_topic_json = json.dumps(topic_stats)
                sentiment_entry.overall_positive = overall["positive"]
                sentiment_entry.overall_neutral = overall["neutral"]
                sentiment_entry.overall_negative = overall["negative"]
                await db.commit()

    except Exception as e:
        logger.exception(f"Sentiment background task failed: {e}")
        async with async_session() as db:
            result = await db.execute(
                select(SentimentAnalysis)
                .filter_by(topic_model_id=topic_model_id, method=method)
                .order_by(SentimentAnalysis.updated_at.desc())
            )
            sentiment_entry = result.scalars().first()
            if sentiment_entry:
                sentiment_entry.status = "failed"
                await db.commit()
