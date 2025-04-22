import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4
import pandas as pd
from io import BytesIO
from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.models.db.topic_model import TopicModel
from app.schemas.topic import (
    LDATopicRequest,
    LDATopicResponse,
    TopicLabelRequest,
    TopicLabelResponse,
    TopicLabelResponseData,
)
from app.services.s3_uploader import download_file_from_s3, upload_file_to_s3
from app.services.topic_labeling import auto_match_labels, generate_default_labels
from app.services.topic_modeling import apply_lda_model, estimate_best_num_topics
from app.services.s3_uploader import delete_file_from_s3

import logging

router = APIRouter(prefix="/api/topic", tags=["Topic Modeling"])
logger = logging.getLogger(__name__)


@router.post("/lda", response_model=LDATopicResponse)
async def run_lda_topic_modeling(req: LDATopicRequest, db: Session = Depends(get_db)):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.lemmatized_s3_key:
            raise HTTPException(status_code=404, detail="Lemmatized file not found")

        # Check for recent result
        latest_lda = (
            db.query(TopicModel)
            .filter_by(file_id=file_id, method="LDA")
            .order_by(TopicModel.created_at.desc())
            .first()
        )

        if (
            latest_lda
            and record.lemmatized_updated_at
            and latest_lda.created_at >= record.lemmatized_updated_at
        ):
            logger.info("⏩ Skipping LDA - result already up-to-date.")
            return LDATopicResponse(
                status="success",
                message="Existing LDA result is still valid.",
                data={
                    "file_id": file_id,
                    "lda_topics_s3_url": latest_lda.s3_url,
                    "topics": json.loads(latest_lda.summary_json),
                },
            )

        # Download and read
        file_bytes = download_file_from_s3(record.lemmatized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        if "lemmatized_tokens" not in df.columns:
            raise HTTPException(
                status_code=400, detail="Missing 'lemmatized_tokens' column"
            )

        tokens = df["lemmatized_tokens"].astype(str)

        # Topic count
        num_topics = req.num_topics or estimate_best_num_topics(tokens)
        logger.info(f"🔢 Using topic count: {num_topics}")

        # Apply LDA
        df["topic_id"], topic_summary = apply_lda_model(tokens, num_topics=num_topics)

        # Upload result
        output_buffer = BytesIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"lda/lda_topics_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key, content_type="text/csv")

        # Save or update DB record
        existing = db.query(TopicModel).filter_by(file_id=file_id, method="LDA").first()
        if existing:
            # ✅ Delete previous file from S3 before updating
            if existing.s3_key and existing.s3_key != new_s3_key:
                try:
                    delete_file_from_s3(existing.s3_key)
                    logger.info(f"🗑️ Deleted old LDA file from S3: {existing.s3_key}")
                except Exception as del_err:
                    logger.warning(f"⚠️ Failed to delete old LDA file: {del_err}")

            existing.s3_key = new_s3_key
            existing.s3_url = s3_url
            existing.topic_count = num_topics
            existing.summary_json = json.dumps(topic_summary)
            db.commit()
        else:
            new_entry = TopicModel(
                id=str(uuid4()),
                file_id=file_id,
                method="LDA",
                topic_count=num_topics,
                s3_key=new_s3_key,
                s3_url=s3_url,
                summary_json=json.dumps(topic_summary),
            )
            db.add(new_entry)
            db.commit()

        return LDATopicResponse(
            status="success",
            message="LDA topic modeling completed and saved to S3.",
            data={
                "file_id": file_id,
                "lda_topics_s3_url": s3_url,
                "topics": topic_summary,
            },
        )

    except HTTPException as he:
        logger.warning(f"⚠️ LDA modeling failed: {he.detail}")
        raise he
    except Exception as e:
        logger.exception(f"❌ Unexpected LDA error: {e}")
        raise HTTPException(status_code=500, detail="LDA topic modeling failed.")


@router.post("/label", response_model=TopicLabelResponse)
async def label_topics(req: TopicLabelRequest, db: Session = Depends(get_db)):
    try:
        topic_model = db.query(TopicModel).filter_by(id=req.topic_model_id).first()
        if not topic_model:
            raise HTTPException(status_code=404, detail="Topic model not found")

        topic_summary = json.loads(topic_model.summary_json)
        label_map = {}
        enriched_topics = []

        # === Skip labeling if already labeled ===
        if (
            all("label" in topic for topic in topic_summary)
            and not req.label_map
            and not req.keywords
        ):
            logger.info(
                "⏩ Labeling skipped — already labeled and no user input provided."
            )
            return TopicLabelResponse(
                status="success",
                message="Topic labels already exist. No update performed.",
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

        # === Labeling strategy ===
        if req.label_map:
            label_map = {int(k): v for k, v in req.label_map.items()}
            for topic in topic_summary:
                tid = int(topic["topic_id"])
                topic["label"] = label_map.get(tid, f"Topic {tid}")
                topic["confidence"] = None
                topic["matched_with"] = "explicit_map"
        elif req.keywords:
            label_map, enriched_topics = auto_match_labels(topic_summary, req.keywords)
            topic_summary = enriched_topics
        else:
            label_map = generate_default_labels(topic_summary)
            for topic in topic_summary:
                tid = int(topic["topic_id"])
                topic["label"] = label_map.get(tid, f"Topic {tid}")
                topic["confidence"] = None
                topic["matched_with"] = "auto_generated"

        # === Read and label file ===
        file_bytes = download_file_from_s3(topic_model.s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["topic_label"] = df["topic_id"].apply(
            lambda x: label_map.get(int(x), f"Topic {x}")
        )

        # === Save labeled file to S3 ===
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        # Always strip older suffix before appending new
        new_s3_key = topic_model.s3_key.replace("_labeled", "").replace(
            ".csv", "_labeled.csv"
        )

        # Delete old S3 file if key has changed
        if topic_model.s3_key != new_s3_key:
            try:
                delete_file_from_s3(topic_model.s3_key)
                logger.info(f"🗑️ Deleted old S3 file: {topic_model.s3_key}")
            except Exception as del_err:
                logger.warning(f"⚠️ Failed to delete old S3 file: {del_err}")

        s3_url = upload_file_to_s3(buffer, new_s3_key)

        # === Update DB ===
        topic_model.s3_key = new_s3_key
        topic_model.s3_url = s3_url
        topic_model.summary_json = json.dumps(topic_summary)
        db.commit()

        return TopicLabelResponse(
            status="success",
            message="Topics labeled and saved to S3.",
            data=TopicLabelResponseData(
                topic_model_id=topic_model.id,
                labeled_s3_url=s3_url,
                columns=df.columns.tolist(),
                record_count=len(df),
                topics=topic_summary,
            ),
        )

    except Exception as e:
        logger.exception(f"❌ Topic labeling failed: {e}")
        raise HTTPException(status_code=500, detail="Topic labeling failed.")
