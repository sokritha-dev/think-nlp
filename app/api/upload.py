# app/api/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from uuid import uuid4
from io import StringIO, BytesIO
import pandas as pd
from sqlalchemy.orm import Session
import logging

from app.middlewares.file_validators import validate_csv
from app.schemas.upload import UploadData, UploadResponse
from app.services.hashing import compute_sha256
from app.services.s3_uploader import upload_file_to_s3, delete_file_from_s3
from app.core.database import get_db
from app.models.db.file_record import FileRecord

router = APIRouter(prefix="/api/upload", tags=["Upload"])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@router.post("/", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    contents: bytes = Depends(validate_csv),
    db: Session = Depends(get_db),
):
    s3_uploaded = False
    s3_key = None

    try:
        # Decode CSV
        decoded = contents.decode("utf-8")
        try:
            df = pd.read_csv(StringIO(decoded))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        if "review" not in df.columns:
            raise HTTPException(
                status_code=400, detail="Missing required 'review' column"
            )

        # Step 1: Hash and check deduplication
        file_hash = compute_sha256(contents)
        existing = db.query(FileRecord).filter_by(file_hash=file_hash).first()
        if existing:
            logger.info(
                "⚠️ Duplicate upload detected. Returning existing file metadata."
            )
            return UploadResponse(
                status="success",
                message="This file was previously uploaded. Reusing the existing file.",
                data=UploadData(
                    file_id=existing.id,
                    file_url=existing.s3_url,
                    s3_key=existing.s3_key,
                    columns=existing.columns.split(","),
                    record_count=existing.record_count,
                ),
            )

        # Step 2: Upload to S3
        s3_key = f"user-data/{uuid4()}.csv"
        s3_url = upload_file_to_s3(BytesIO(contents), s3_key, content_type="text/csv")
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

        return UploadResponse(
            status="success",
            message="File uploaded successfully",
            data=UploadData(
                file_id=file_record.id,  # ✅ include this
                file_url=s3_url,
                s3_key=s3_key,
                columns=df.columns.tolist(),
                record_count=len(df),
            ),
        )

    except HTTPException as he:
        logger.warning(f"⚠️ Upload failed: {he.detail}")
        raise he

    except Exception as e:
        logger.exception(f"❌ Unexpected error occurred::: {e}")
        if s3_uploaded and s3_key:
            try:
                delete_file_from_s3(s3_key)
                logger.info(f"🗑️ Rolled back S3 upload: {s3_key}")
            except Exception as s3e:
                logger.error(f"❗ Failed to delete file from S3: {s3e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
