from datetime import datetime
import ast
import json
import logging
from sqlalchemy import select

from app.core.database import async_session
from app.core.sentiment.analyzer import analyzer_for, SentimentConfig
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.models.db.topic_model import TopicModel

from app.services.file_service import FileService
from app.core.file_handler.storage import S3Storage
from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression

logger = logging.getLogger(__name__)


def _files() -> FileService:
    return FileService(
        storage=S3Storage(),
        codec=CsvCodec(index=False),
        compression=GzipCompression(),
    )


def _safe_pct(num: int, den: int) -> float:
    return 0.0 if not den else round((num / den) * 100.0, 1)


def _to_tokens(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else x.split()
        except Exception:
            return x.split()
    return []


async def run_sentiment_background(topic_model_id: str, method: str):
    method = (method or "vader").lower()

    try:
        # === Step 1: fetch the pending row + topic model ===
        async with async_session() as db:
            sa = (
                (
                    await db.execute(
                        select(SentimentAnalysis)
                        .filter_by(topic_model_id=topic_model_id, method=method)
                        .order_by(SentimentAnalysis.updated_at.desc())
                    )
                )
                .scalars()
                .first()
            )
            if not sa:
                logger.warning(
                    f"No SentimentAnalysis row for {topic_model_id} ({method})"
                )
                return

            sa.status = "processing"
            sa.updated_at = datetime.now()
            await db.commit()

            tm = (
                (await db.execute(select(TopicModel).filter_by(id=topic_model_id)))
                .scalars()
                .first()
            )

            if not tm:
                sa.status = "failed"
                await db.commit()
                logger.error(f"TopicModel {topic_model_id} not found")
                return

        # === Step 2: load DF via FileService and score sentiments (no DB I/O here) ===
        files = _files()
        df = await files.download_df(tm.s3_key)  # transparently handles .csv vs .csv.gz

        required = {"lemmatized_tokens", "topic_id", "topic_label"}
        if not required.issubset(df.columns):
            logger.error(f"Missing required columns in topic file: need {required}")
            async with async_session() as db:
                sa = (
                    (
                        await db.execute(
                            select(SentimentAnalysis)
                            .filter_by(topic_model_id=topic_model_id, method=method)
                            .order_by(SentimentAnalysis.updated_at.desc())
                        )
                    )
                    .scalars()
                    .first()
                )
                if sa:
                    sa.status = "failed"
                    await db.commit()
            return

        # Build plain text per row from tokens (robust to stringified lists)
        toks = df["lemmatized_tokens"].apply(_to_tokens)
        df["text"] = toks.apply(lambda ts: " ".join(map(str, ts)))

        # Use your analyzer factory (VADER/TextBlob/BERT)
        analyzer = analyzer_for(SentimentConfig(method=method))
        df["sentiment"] = analyzer.score_series(df["text"])

        # === Step 3: aggregate ===
        counts = df["sentiment"].value_counts().to_dict()
        total = len(df)
        overall = {
            "positive": _safe_pct(counts.get("positive", 0), total),
            "neutral": _safe_pct(counts.get("neutral", 0), total),
            "negative": _safe_pct(counts.get("negative", 0), total),
        }

        topic_summary = json.loads(tm.summary_json or "[]")
        topic_map = {int(t["topic_id"]): t for t in topic_summary}

        topic_stats = []
        for label, group in df.groupby("topic_label"):
            n = len(group)
            # pick the most frequent topic_id inside this label (safer than iloc[0])
            tid = int(group["topic_id"].mode().iloc[0]) if n else 0
            kws = topic_map.get(tid, {}).get("keywords", [])
            if isinstance(kws, str):
                kws = [k.strip() for k in kws.split(",") if k.strip()]

            topic_stats.append(
                {
                    "label": label,
                    "positive": _safe_pct((group["sentiment"] == "positive").sum(), n),
                    "neutral": _safe_pct((group["sentiment"] == "neutral").sum(), n),
                    "negative": _safe_pct((group["sentiment"] == "negative").sum(), n),
                    "keywords": kws,
                }
            )

        # === Step 4: persist ===
        async with async_session() as db:
            sa = (
                (
                    await db.execute(
                        select(SentimentAnalysis)
                        .filter_by(topic_model_id=topic_model_id, method=method)
                        .order_by(SentimentAnalysis.updated_at.desc())
                    )
                )
                .scalars()
                .first()
            )
            if sa:
                sa.status = "done"
                sa.updated_at = datetime.now()
                sa.per_topic_json = json.dumps(topic_stats)
                sa.overall_positive = overall["positive"]
                sa.overall_neutral = overall["neutral"]
                sa.overall_negative = overall["negative"]
                await db.commit()

    except Exception as e:
        logger.exception(f"Sentiment background task failed: {e}")
        async with async_session() as db:
            sa = (
                (
                    await db.execute(
                        select(SentimentAnalysis)
                        .filter_by(topic_model_id=topic_model_id, method=method)
                        .order_by(SentimentAnalysis.updated_at.desc())
                    )
                )
                .scalars()
                .first()
            )
            if sa:
                sa.status = "failed"
                await db.commit()
