# app/api/upload.py

from fastapi import APIRouter, Request, UploadFile, File, Depends
from uuid import uuid4
import pandas as pd
from sqlalchemy.orm import Session
import logging

from app.middlewares.file_validators import validate_csv
from app.schemas.upload import UploadData, UploadResponse
from app.services.hashing import compute_sha256
from app.services.s3_uploader import (
    upload_compressed_csv_to_s3,
    delete_file_from_s3,
)
from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.utils.exceptions import BadRequestError, ServerError
from app.utils.response_builder import success_response
from app.messages.upload_messages import (
    UPLOAD_SUCCESS,
    DUPLICATE_FILE_FOUND,
    UPLOAD_FAILED,
)
from app.middlewares.security import limiter


router = APIRouter(prefix="/api/upload", tags=["Upload"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=UploadResponse)
@limiter.limit("5/hour")
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    validated: tuple[bytes, pd.DataFrame] = Depends(validate_csv(["review"])),
    db: Session = Depends(get_db),
):
    s3_uploaded = False
    s3_key = None

    try:
        contents, df = validated

        # Step 1: Hash and check deduplication
        file_hash = compute_sha256(contents)
        existing = db.query(FileRecord).filter_by(file_hash=file_hash).first()
        if existing:
            logger.info(
                "⚠️ Duplicate upload detected. Returning existing file metadata."
            )
            return success_response(
                message=DUPLICATE_FILE_FOUND,
                data=UploadData(
                    file_id=existing.id,
                    file_url=existing.s3_url,
                    s3_key=existing.s3_key,
                    columns=existing.columns.split(","),
                    record_count=existing.record_count,
                ),
            )

        s3_key = f"user-data/{uuid4()}.csv.gz"
        s3_url = upload_compressed_csv_to_s3(contents, s3_key)

        s3_uploaded = True

        # Step 3: Save to DB
        file_record = FileRecord(
            id=str(uuid4()),
            file_name=file.filename,
            s3_key=s3_key,
            s3_url=s3_url,
            columns=",".join(df.columns),
            record_count=len(df),
            file_hash=file_hash,
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        logger.info(f"✅ Uploaded and saved {s3_key} ({len(df)} rows)")

        return success_response(
            message=UPLOAD_SUCCESS,
            data=UploadData(
                file_id=file_record.id,
                file_url=s3_url,
                s3_key=s3_key,
                columns=df.columns.tolist(),
                record_count=len(df),
            ),
        )

    except BadRequestError as he:
        logger.warning(f"⚠️ Upload bad request: {he.detail}")
        raise he

    except Exception as e:
        logger.exception(f"❌ Unexpected upload error: {e}")
        if s3_uploaded and s3_key:
            try:
                delete_file_from_s3(s3_key)
                logger.info(f"🗑️ Rolled back S3 upload: {s3_key}")
            except Exception as s3e:
                logger.error(f"❗ Failed to delete file from S3 during rollback: {s3e}")

        raise ServerError(code="UPLOAD_FAILED", message=UPLOAD_FAILED)
