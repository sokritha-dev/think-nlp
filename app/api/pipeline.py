# app/api/pipeline.py

from datetime import datetime
import gzip
import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
import pandas as pd
from uuid import uuid4
from io import BytesIO
import logging

from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.models.db.topic_model import TopicModel
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.schemas.pipeline import FullPipelineRequest
from app.schemas.sentiment import SentimentTopicBreakdown
from app.services.s3_uploader import (
    generate_presigned_url,
    download_file_from_s3,
    save_csv_to_s3,
)
from app.services.preprocess import (
    normalize_text,
    remove_special_characters,
    tokenize_text,
    remove_stopwords_from_tokens,
    lemmatize_tokens,
)
from app.services.topic_modeling import estimate_best_num_topics, apply_lda_model
from app.services.topic_labeling import generate_default_labels
from app.services.sentiment_analysis import analyze_sentiment_vader
from app.eda.eda_analysis import EDA

from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError
from app.messages.pipeline_messages import (
    FILE_NOT_FOUND,
    SAMPLE_FILE_NOT_FOUND,
    TOPIC_MODEL_NOT_FOUND,
    SENTIMENT_RESULT_NOT_FOUND,
    FULL_PIPELINE_SUCCESS,
    SAMPLE_DATA_URL_SUCCESS,
)

router = APIRouter(prefix="/api/pipeline", tags=["Full Pipeline"])
logger = logging.getLogger(__name__)


@router.post("/sentiment-analysis")
def run_full_pipeline(req: FullPipelineRequest, db: Session = Depends(get_db)):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        existing_topic_model = (
            db.query(TopicModel)
            .filter_by(file_id=file_id, method="LDA")
            .order_by(TopicModel.updated_at.desc())
            .first()
        )
        if existing_topic_model:
            existing_sentiment = (
                db.query(SentimentAnalysis)
                .filter_by(topic_model_id=existing_topic_model.id, method="vader")
                .order_by(SentimentAnalysis.updated_at.desc())
                .first()
            )
            if (
                existing_sentiment
                and record.lemmatized_updated_at
                and existing_sentiment.updated_at >= record.lemmatized_updated_at
            ):
                logger.info("⏩ Skipping pipeline — sentiment already up-to-date.")
                per_topic = [
                    SentimentTopicBreakdown(**t)
                    for t in json.loads(existing_sentiment.per_topic_json or "[]")
                ]
                return success_response(
                    message="Sentiment already computed. Reusing existing result.",
                    data={
                        "overall": {
                            "positive": existing_sentiment.overall_positive,
                            "neutral": existing_sentiment.overall_neutral,
                            "negative": existing_sentiment.overall_negative,
                        },
                        "per_topic": [t.model_dump() for t in per_topic],
                    },
                )

        original_bytes = download_file_from_s3(record.s3_key)
        if record.s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(original_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(original_bytes))

        df["normalized_review"] = df["review"].dropna().apply(normalize_text)

        norm_key, norm_url = save_csv_to_s3(df, "normalization", suffix="normalized")
        record.normalized_s3_key = norm_key
        record.normalized_s3_url = norm_url
        record.normalized_updated_at = datetime.now()

        cleaned_rows, removed_chars = [], []
        for row in df["normalized_review"].dropna():
            cleaned, removed = remove_special_characters(row, True, True, True)
            cleaned_rows.append(cleaned)
            removed_chars.extend(removed)
        df["special_cleaned"] = cleaned_rows

        clean_key, clean_url = save_csv_to_s3(df, "cleaned", suffix="special_cleaned")
        record.special_cleaned_s3_key = clean_key
        record.special_cleaned_s3_url = clean_url
        record.special_cleaned_flags = json.dumps(
            {"remove_special": True, "remove_numbers": True, "remove_emoji": True}
        )
        record.special_cleaned_removed = json.dumps(list(set(removed_chars)))
        record.special_cleaned_updated_at = datetime.now()

        df["tokens"] = df["special_cleaned"].dropna().apply(tokenize_text)

        token_key, token_url = save_csv_to_s3(df, "tokenization", suffix="tokens")
        record.tokenized_s3_key = token_key
        record.tokenized_s3_url = token_url
        record.tokenized_updated_at = datetime.now()

        df["stopword_removed"] = df["tokens"].apply(
            lambda toks: remove_stopwords_from_tokens(toks)["cleaned_tokens"]
        )

        stopword_key, stopword_url = save_csv_to_s3(
            df, "stopwords", suffix="stopword_removed"
        )
        record.stopword_s3_key = stopword_key
        record.stopword_s3_url = stopword_url
        record.stopword_updated_at = datetime.now()

        df["lemmatized_tokens"] = df["stopword_removed"].apply(
            lambda toks: lemmatize_tokens(toks)["lemmatized_tokens"]
        )

        lemma_key, lemma_url = save_csv_to_s3(df, "lemmatization", suffix="lemmatized")
        record.lemmatized_s3_key = lemma_key
        record.lemmatized_s3_url = lemma_url
        record.lemmatized_updated_at = datetime.now()

        db.commit()

        eda = EDA(df=df, file_id=file_id)
        image_urls = eda.run_eda()
        record.eda_wordcloud_url = image_urls.get("word_cloud")
        record.eda_text_length_url = image_urls.get("length_distribution")
        record.eda_word_freq_url = image_urls.get("common_words")
        record.eda_bigram_url = image_urls.get("2gram")
        record.eda_trigram_url = image_urls.get("3gram")
        record.eda_updated_at = datetime.now()
        db.commit()

        tokens = df["lemmatized_tokens"].astype(str)
        num_topics = estimate_best_num_topics(tokens)
        df["topic_id"], topic_summary = apply_lda_model(tokens, num_topics=num_topics)

        lda_key, lda_url = save_csv_to_s3(df, "lda", suffix="lda_topics")
        lda_entry = TopicModel(
            id=str(uuid4()),
            file_id=file_id,
            method="LDA",
            topic_count=num_topics,
            s3_key=lda_key,
            s3_url=lda_url,
            summary_json=json.dumps(topic_summary),
            updated_at=datetime.now(),
        )
        db.add(lda_entry)
        db.commit()

        label_map = generate_default_labels(topic_summary)
        df["topic_label"] = df["topic_id"].apply(
            lambda x: label_map.get(int(x), f"Topic {x}")
        )
        for topic in topic_summary:
            tid = int(topic["topic_id"])
            topic["label"] = label_map.get(tid)
            topic.setdefault("keywords", [])

        df["text"] = df["lemmatized_tokens"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else " ".join(eval(x))
        )
        df["sentiment"] = df["text"].apply(analyze_sentiment_vader)

        sentiment_counts = df["sentiment"].value_counts().to_dict()
        total = len(df)
        overall = {
            "positive": round((sentiment_counts.get("positive", 0) / total) * 100, 1),
            "neutral": round((sentiment_counts.get("neutral", 0) / total) * 100, 1),
            "negative": round((sentiment_counts.get("negative", 0) / total) * 100, 1),
        }

        per_topic = []
        for topic in topic_summary:
            label = topic["label"]
            keywords = (
                topic["keywords"].split(", ")
                if isinstance(topic["keywords"], str)
                else topic["keywords"]
            )

            group = df[df["topic_label"] == label]
            count = len(group)
            per_topic.append(
                SentimentTopicBreakdown(
                    label=label,
                    keywords=keywords,
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

        sentiment_entry = SentimentAnalysis(
            id=str(uuid4()),
            topic_model_id=lda_entry.id,
            method="vader",
            overall_positive=overall["positive"],
            overall_neutral=overall["neutral"],
            overall_negative=overall["negative"],
            per_topic_json=json.dumps([t.model_dump() for t in per_topic]),
            updated_at=datetime.now(),
        )
        db.add(sentiment_entry)
        db.commit()

        return success_response(
            message=FULL_PIPELINE_SUCCESS,
            data={
                "overall": overall,
                "per_topic": [t.model_dump() for t in per_topic],
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Full pipeline failed: {e}")
        raise ServerError(
            code="FULL_PIPELINE_FAILED",
            message="Full pipeline sentiment analysis failed.",
        )


@router.get("/result")
def get_result(file_id: str = Query(...), db: Session = Depends(get_db)):
    try:
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=SAMPLE_FILE_NOT_FOUND)

        topic_model = (
            db.query(TopicModel)
            .filter_by(file_id=file_id, method="LDA")
            .order_by(TopicModel.updated_at.desc())
            .first()
        )
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND",
                message=TOPIC_MODEL_NOT_FOUND,
            )

        sentiment = (
            db.query(SentimentAnalysis)
            .filter_by(topic_model_id=topic_model.id, method="vader")
            .order_by(SentimentAnalysis.updated_at.desc())
            .first()
        )
        if not sentiment:
            raise NotFoundError(
                code="SENTIMENT_RESULT_NOT_FOUND", message=SENTIMENT_RESULT_NOT_FOUND
            )

        per_topic = [
            SentimentTopicBreakdown(**t)
            for t in json.loads(sentiment.per_topic_json or "[]")
        ]

        return success_response(
            message="Sentiment analysis result loaded successfully.",
            data={
                "file_id": file_id,
                "overall": {
                    "positive": sentiment.overall_positive,
                    "neutral": sentiment.overall_neutral,
                    "negative": sentiment.overall_negative,
                },
                "per_topic": [t.model_dump() for t in per_topic],
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Failed to load result: {e}")
        raise ServerError(
            code="RESULT_LOAD_FAILED", message="Failed to load sentiment result."
        )


@router.get("/sample-data-url")
def get_sample_data_url(file_id: str = Query(...)):
    try:
        s3_key = f"user-data/{file_id}.csv.gz"
        s3_url = generate_presigned_url(
            bucket="nlp-learner",
            key=s3_key,
        )
        return success_response(
            message=SAMPLE_DATA_URL_SUCCESS, data={"s3_url": s3_url}
        )
    except Exception as e:
        logger.exception(f"❌ Failed to generate sample data URL: {e}")
        raise ServerError(
            code="SAMPLE_DATA_URL_FAILED", message="Failed to generate sample data URL."
        )
