# app/api/topic.py

from datetime import datetime
import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from uuid import uuid4
import pandas as pd
from io import BytesIO
import logging
import gzip

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
from app.services.s3_uploader import (
    download_file_from_s3,
    upload_file_to_s3,
    delete_file_from_s3,
)
from app.services.topic_labeling import auto_match_labels, generate_default_labels
from app.services.topic_modeling import apply_lda_model, estimate_best_num_topics

from app.utils.exceptions import NotFoundError, BadRequestError, ServerError
from app.utils.response_builder import success_response
from app.messages.topic_messages import (
    LDA_COMPLETED,
    LDA_UP_TO_DATE,
    LEMMATIZED_FILE_NOT_FOUND,
    LEMMATIZED_COLUMN_MISSING,
    TOPIC_LABELING_COMPLETED,
    TOPIC_MODEL_NOT_FOUND,
)

router = APIRouter(prefix="/api/topic", tags=["Topic Modeling"])
logger = logging.getLogger(__name__)


@router.post("/lda", response_model=LDATopicResponse)
async def run_lda_topic_modeling(req: LDATopicRequest, db: Session = Depends(get_db)):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND", message=LEMMATIZED_FILE_NOT_FOUND
            )

        existing = db.query(TopicModel).filter_by(file_id=file_id, method="LDA").first()

        if (
            existing
            and record.lemmatized_updated_at
            and existing.updated_at >= record.lemmatized_updated_at
            and req.num_topics == existing.topic_count
        ):
            logger.info("⏩ Skipping LDA - result already up-to-date.")
            return success_response(
                message=LDA_UP_TO_DATE,
                data={
                    "file_id": file_id,
                    "topic_model_id": existing.id,
                    "lda_topics_s3_url": existing.s3_url,
                    "topics": json.loads(existing.summary_json),
                },
            )

        file_bytes = download_file_from_s3(record.lemmatized_s3_key)
        if record.lemmatized_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        if "lemmatized_tokens" not in df.columns:
            raise BadRequestError(
                code="LEMMATIZED_COLUMN_MISSING", message=LEMMATIZED_COLUMN_MISSING
            )

        tokens = df["lemmatized_tokens"].astype(str)
        num_topics = req.num_topics or estimate_best_num_topics(tokens)
        logger.info(f"🔢 Using topic count: {num_topics}")

        df["topic_id"], topic_summary = apply_lda_model(tokens, num_topics=num_topics)

        output_buffer = BytesIO()
        with gzip.GzipFile(fileobj=output_buffer, mode="wb") as gz:
            df.to_csv(gz, index=False)
        output_buffer.seek(0)

        new_s3_key = f"lda/lda_topics_{uuid4()}.csv.gz"
        s3_url = upload_file_to_s3(
            output_buffer, new_s3_key, content_type="application/gzip"
        )

        if existing:
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
            existing.updated_at = datetime.now()
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
                updated_at=datetime.now(),
            )
            db.add(new_entry)
            db.commit()

        return success_response(
            message=LDA_COMPLETED,
            data={
                "file_id": file_id,
                "topic_model_id": existing.id if existing else new_entry.id,
                "lda_topics_s3_url": s3_url,
                "topics": topic_summary,
            },
        )

    except (NotFoundError, BadRequestError) as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Unexpected LDA error: {e}")
        raise ServerError(code="LDA_FAILED", message="LDA topic modeling failed.")


@router.post("/label", response_model=TopicLabelResponse)
async def label_topics(req: TopicLabelRequest, db: Session = Depends(get_db)):
    try:
        topic_model = db.query(TopicModel).filter_by(id=req.topic_model_id).first()
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        topic_summary = json.loads(topic_model.summary_json)
        label_map = {}

        # Check if recomputation needed
        prev_keywords = json.loads(topic_model.label_keywords or "[]")
        prev_label_map = json.loads(topic_model.label_map_json or "{}")

        if (
            all("label" in t for t in topic_summary)
            and req.keywords == (prev_keywords or None)
            and (req.label_map or {}) == prev_label_map
        ):
            logger.info("⏩ Skipping labeling — inputs unchanged.")
            return success_response(
                message="Labels already exist and inputs haven't changed.",
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

        # Labeling logic
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

        # Save labeled file
        file_bytes = download_file_from_s3(topic_model.s3_key)
        if topic_model.s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        df["topic_label"] = df["topic_id"].apply(
            lambda x: label_map.get(int(x), f"Topic {x}")
        )

        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            df.to_csv(gz, index=False)
        buffer.seek(0)

        base_key = (
            topic_model.s3_key.replace("_labeled", "")
            .replace(".gz", "")
            .replace(".csv", "")
        )
        new_s3_key = f"{base_key}_labeled.csv.gz"

        if topic_model.s3_key != new_s3_key:
            try:
                delete_file_from_s3(topic_model.s3_key)
            except Exception as del_err:
                logger.warning(f"⚠️ Failed to delete old S3 file: {del_err}")

        s3_url = upload_file_to_s3(buffer, new_s3_key)

        topic_model.s3_key = new_s3_key
        topic_model.s3_url = s3_url
        topic_model.summary_json = json.dumps(topic_summary)
        topic_model.label_keywords = json.dumps(req.keywords or [])
        topic_model.label_map_json = json.dumps(req.label_map or {})
        db.commit()

        return success_response(
            message=TOPIC_LABELING_COMPLETED,
            data=TopicLabelResponseData(
                topic_model_id=topic_model.id,
                labeled_s3_url=s3_url,
                columns=df.columns.tolist(),
                record_count=len(df),
                topics=topic_summary,
            ),
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Topic labeling failed: {e}")
        raise ServerError(
            code="TOPIC_LABELING_FAILED", message="Topic labeling failed."
        )


@router.get("/label", response_model=TopicLabelResponse)
async def get_existing_topic_labels(
    topic_model_id: str = Query(...),  # Required query param
    db: Session = Depends(get_db),
):
    try:
        topic_model = db.query(TopicModel).filter_by(id=topic_model_id).first()
        if not topic_model:
            raise NotFoundError(
                code="TOPIC_MODEL_NOT_FOUND", message=TOPIC_MODEL_NOT_FOUND
            )

        topic_summary = json.loads(topic_model.summary_json)

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

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Failed to fetch topic labels: {e}")
        raise ServerError(
            code="TOPIC_LABEL_FETCH_FAILED",
            message="Failed to retrieve topic labeling result.",
        )
