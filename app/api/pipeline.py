from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
from uuid import uuid4
from io import BytesIO
import json
import logging

from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.models.db.topic_model import TopicModel
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.schemas.pipeline import FullPipelineRequest
from app.schemas.sentiment import SentimentChartData, SentimentTopicBreakdown
from app.services.s3_uploader import (
    generate_presigned_url,
    upload_file_to_s3,
    download_file_from_s3,
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

router = APIRouter(prefix="/api/pipeline", tags=["Full Pipeline"])
logger = logging.getLogger(__name__)


@router.post("/sentiment-analysis", response_model=SentimentChartData)
def run_full_pipeline(req: FullPipelineRequest, db: Session = Depends(get_db)):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="File not found")

        # Step 1: Normalize
        original_bytes = download_file_from_s3(record.s3_key)
        df = pd.read_csv(BytesIO(original_bytes))
        df["normalized_review"] = df["review"].dropna().apply(normalize_text)

        norm_buffer = BytesIO()
        df.to_csv(norm_buffer, index=False)
        norm_buffer.seek(0)
        norm_key = f"normalization/normalized_{uuid4()}.csv"
        norm_url = upload_file_to_s3(norm_buffer, norm_key)
        record.normalized_s3_key = norm_key
        record.normalized_s3_url = norm_url

        # Step 2: Special cleaning
        cleaned_rows = []
        removed_chars = []
        for row in df["normalized_review"].dropna():
            cleaned, removed = remove_special_characters(row, True, True, True)
            cleaned_rows.append(cleaned)
            removed_chars.extend(removed)
        df["special_cleaned"] = cleaned_rows

        clean_buffer = BytesIO()
        df.to_csv(clean_buffer, index=False)
        clean_buffer.seek(0)
        clean_key = f"cleaned/special_cleaned_{uuid4()}.csv"
        clean_url = upload_file_to_s3(clean_buffer, clean_key)
        record.special_cleaned_s3_key = clean_key
        record.special_cleaned_s3_url = clean_url
        record.special_cleaned_flags = json.dumps(
            {"remove_special": True, "remove_numbers": True, "remove_emoji": True}
        )
        record.special_cleaned_updated_at = datetime.now()

        # Step 3: Tokenization
        df["tokens"] = df["special_cleaned"].dropna().apply(tokenize_text)

        token_buffer = BytesIO()
        df.to_csv(token_buffer, index=False)
        token_buffer.seek(0)
        token_key = f"tokenization/tokens_{uuid4()}.csv"
        token_url = upload_file_to_s3(token_buffer, token_key)
        record.tokenized_s3_key = token_key
        record.tokenized_s3_url = token_url
        record.tokenized_updated_at = datetime.now()

        # Step 4: Stopword Removal
        df["stopword_removed"] = df["tokens"].apply(
            lambda toks: remove_stopwords_from_tokens(toks)["cleaned_tokens"]
        )

        stopword_buffer = BytesIO()
        df.to_csv(stopword_buffer, index=False)
        stopword_buffer.seek(0)
        stopword_key = f"stopwords/stopword_removed_{uuid4()}.csv"
        stopword_url = upload_file_to_s3(stopword_buffer, stopword_key)
        record.stopword_s3_key = stopword_key
        record.stopword_s3_url = stopword_url
        record.stopword_updated_at = datetime.now()

        # Step 5: Lemmatization
        df["lemmatized_tokens"] = df["stopword_removed"].apply(
            lambda toks: lemmatize_tokens(toks)["lemmatized_tokens"]
        )

        lemma_buffer = BytesIO()
        df.to_csv(lemma_buffer, index=False)
        lemma_buffer.seek(0)
        lemma_key = f"lemmatization/lemmatized_{uuid4()}.csv"
        lemma_url = upload_file_to_s3(lemma_buffer, lemma_key)
        record.lemmatized_s3_key = lemma_key
        record.lemmatized_s3_url = lemma_url
        record.lemmatized_updated_at = datetime.now()

        db.commit()

        # Step 6: EDA
        eda = EDA(df=df, file_id=file_id)
        image_urls = eda.run_eda()
        record.eda_wordcloud_url = image_urls.get("word_cloud")
        record.eda_text_length_url = image_urls.get("length_distribution")
        record.eda_word_freq_url = image_urls.get("common_words")
        record.eda_bigram_url = image_urls.get("2gram")
        record.eda_trigram_url = image_urls.get("3gram")
        record.eda_updated_at = datetime.now()
        db.commit()

        # Step 7: Topic Modeling
        tokens = df["lemmatized_tokens"].astype(str)
        num_topics = estimate_best_num_topics(tokens)
        df["topic_id"], topic_summary = apply_lda_model(tokens, num_topics=num_topics)

        lda_buffer = BytesIO()
        df.to_csv(lda_buffer, index=False)
        lda_buffer.seek(0)
        lda_key = f"lda/lda_topics_{uuid4()}.csv"
        lda_url = upload_file_to_s3(lda_buffer, lda_key)

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

        # Step 8: Labeling
        label_map = generate_default_labels(topic_summary)
        df["topic_label"] = df["topic_id"].apply(
            lambda x: label_map.get(int(x), f"Topic {x}")
        )
        for topic in topic_summary:
            tid = int(topic["topic_id"])
            topic["label"] = label_map.get(tid)

        label_buffer = BytesIO()
        df.to_csv(label_buffer, index=False)
        label_buffer.seek(0)
        labeled_key = lda_key.replace(".csv", "_labeled.csv")
        labeled_url = upload_file_to_s3(label_buffer, labeled_key)

        lda_entry.s3_key = labeled_key
        lda_entry.s3_url = labeled_url
        lda_entry.summary_json = json.dumps(topic_summary)
        lda_entry.updated_at = datetime.now()
        db.commit()

        # Step 9: Sentiment
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
        for label, group in df.groupby("topic_label"):
            count = len(group)
            per_topic.append(
                SentimentTopicBreakdown(
                    label=label,
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
            per_topic_json=json.dumps([t.dict() for t in per_topic]),
            updated_at=datetime.now(),
        )
        db.add(sentiment_entry)
        db.commit()

        return SentimentChartData(overall=overall, per_topic=per_topic)

    except Exception as e:
        logger.exception(f"❌ Full pipeline failed: {e}")
        raise HTTPException(
            status_code=500, detail="Full pipeline sentiment analysis failed"
        )


# Replace this with your actual file ID for the sample
SAMPLE_FILE_ID = "1dac7777-0b64-40e5-8eda-5370324268cc"


@router.get("/sample-result", response_model=SentimentChartData)
def get_sample_result(db: Session = Depends(get_db)):
    try:
        # Validate sample file exists
        record = db.query(FileRecord).filter_by(id=SAMPLE_FILE_ID).first()
        if not record:
            raise HTTPException(status_code=404, detail="Sample file not found")

        # Get latest TopicModel
        topic_model = (
            db.query(TopicModel)
            .filter_by(file_id=SAMPLE_FILE_ID, method="LDA")
            .order_by(TopicModel.updated_at.desc())
            .first()
        )
        if not topic_model:
            raise HTTPException(status_code=404, detail="Sample topic model not found")

        # Get latest SentimentAnalysis
        sentiment = (
            db.query(SentimentAnalysis)
            .filter_by(topic_model_id=topic_model.id, method="bert")
            .order_by(SentimentAnalysis.updated_at.desc())
            .first()
        )
        if not sentiment:
            raise HTTPException(
                status_code=404, detail="Sentiment result for sample not found"
            )

        per_topic = [
            SentimentTopicBreakdown(**t)
            for t in json.loads(sentiment.per_topic_json or "[]")
        ]

        return SentimentChartData(
            overall={
                "positive": sentiment.overall_positive,
                "neutral": sentiment.overall_neutral,
                "negative": sentiment.overall_negative,
            },
            per_topic=per_topic,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load sample result: {e}"
        )


@router.get("/sample-data-url")
def get_sample_data_url():
    s3_url = generate_presigned_url(
        bucket="nlp-learner",
        key="user-data/2fb68a00-ba11-47fa-8d42-cb6c51d4e703.csv",
    )

    return {"status": "success", "data": {"s3_url": s3_url}}
