from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
import logging

from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.schemas.eda import EDARequest
from app.services.s3_uploader import delete_file_from_s3, download_file_from_s3
from app.eda.eda_analysis import EDA

router = APIRouter(prefix="/api/eda", tags=["EDA"])
logger = logging.getLogger(__name__)


@router.post("/summary")
async def generate_eda(req: EDARequest, db: Session = Depends(get_db)):
    try:
        # Step 1: Get file record
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="File not found")

        if not record.lemmatized_s3_key:
            raise HTTPException(status_code=404, detail="Lemmatized file not found")

        # Step 2: Download lemmatized file from S3
        file_bytes = download_file_from_s3(record.lemmatized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # Step 3: Delete old EDA files from S3
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
                    # Extract s3 key from full URL
                    s3_key = old_url.split("amazonaws.com/")[-1]
                    delete_file_from_s3(s3_key)
                    logger.info(f"🗑️ Deleted old EDA image: {s3_key}")
                except Exception as del_err:
                    logger.warning(f"⚠️ Failed to delete {key_attr} from S3: {del_err}")

        # Step 4: Run EDA
        eda = EDA(df=df, file_id=file_id)
        image_urls = eda.run_eda()

        # Step 5: Save new image URLs to DB
        record.eda_wordcloud_url = image_urls.get("word_cloud")
        record.eda_text_length_url = image_urls.get("length_distribution")
        record.eda_word_freq_url = image_urls.get("common_words")
        record.eda_bigram_url = image_urls.get("2gram")
        record.eda_trigram_url = image_urls.get("3gram")
        db.commit()

        logger.info(f"✅ EDA completed for file {file_id}")

        return {
            "status": "success",
            "message": "EDA visualizations generated and uploaded.",
            "data": {
                "file_id": file_id,
                **image_urls,
            },
        }

    except HTTPException as he:
        logger.warning(f"⚠️ EDA error for file {req.file_id}: {he.detail}")
        raise he

    except Exception as e:
        logger.exception(f"❌ Failed to generate EDA for file {req.file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate EDA.")