# app/api/eda.py

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
import logging

from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.schemas.eda import EDARequest
from app.services.s3_uploader import delete_file_from_s3, download_file_from_s3
from app.eda.eda_analysis import EDA

from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError
from app.messages.eda_messages import (
    FILE_NOT_FOUND_FOR_EDA,
    LEMMATIZED_FILE_NOT_FOUND,
    EDA_ALREADY_UP_TO_DATE,
    EDA_GENERATION_SUCCESS,
)

router = APIRouter(prefix="/api/eda", tags=["EDA"])
logger = logging.getLogger(__name__)


@router.post("/summary")
async def generate_eda(req: EDARequest, db: Session = Depends(get_db)):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND_FOR_EDA)

        if not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND", message=LEMMATIZED_FILE_NOT_FOUND
            )

        # ✅ Skip if EDA is already up-to-date
        if (
            record.eda_wordcloud_url
            and record.eda_updated_at
            and record.lemmatized_updated_at
            and record.eda_updated_at >= record.lemmatized_updated_at
        ):
            logger.info("⏩ Skipping EDA — already up to date.")
            return success_response(
                message=EDA_ALREADY_UP_TO_DATE,
                data={
                    "file_id": file_id,
                    "word_cloud": record.eda_wordcloud_url,
                    "length_distribution": record.eda_text_length_url,
                    "common_words": record.eda_word_freq_url,
                    "2gram": record.eda_bigram_url,
                    "3gram": record.eda_trigram_url,
                },
            )

        # Step 1: Download lemmatized file
        file_bytes = download_file_from_s3(record.lemmatized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # Step 2: Delete old EDA images from S3
        for key_attr in [
            "eda_wordcloud_url",
            "eda_text_length_url",
            "eda_word_freq_url",
            "eda_bigram_url",
            "eda_trigram_url",
        ]:
            old_url = getattr(record, key_attr)
            if old_url:
                try:
                    s3_key = old_url.split("amazonaws.com/")[-1]
                    delete_file_from_s3(s3_key)
                    logger.info(f"🗑️ Deleted old EDA image: {s3_key}")
                except Exception as del_err:
                    logger.warning(f"⚠️ Failed to delete {key_attr} from S3: {del_err}")

        # Step 3: Run EDA and upload new images
        eda = EDA(df=df, file_id=file_id)
        image_urls = eda.run_eda()

        # Step 4: Save new image URLs and timestamp to DB
        record.eda_wordcloud_url = image_urls.get("word_cloud")
        record.eda_text_length_url = image_urls.get("length_distribution")
        record.eda_word_freq_url = image_urls.get("common_words")
        record.eda_bigram_url = image_urls.get("2gram")
        record.eda_trigram_url = image_urls.get("3gram")
        record.eda_updated_at = datetime.now()
        db.commit()

        logger.info(f"✅ EDA completed for file {file_id}")

        return success_response(
            message=EDA_GENERATION_SUCCESS,
            data={
                "file_id": file_id,
                **image_urls,
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Failed to generate EDA for file {req.file_id}: {e}")
        raise ServerError(code="EDA_FAILED", message="Failed to generate EDA.")
