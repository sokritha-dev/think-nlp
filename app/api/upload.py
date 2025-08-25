from fastapi import APIRouter, Request, UploadFile, File, Depends
from uuid import uuid4
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.middlewares.file_validators import validate_csv
from app.schemas.upload import UploadData, UploadResponse
from app.utils.hashing import compute_sha256
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

from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression
from app.core.file_handler.storage import S3Storage
from app.services.file_service import FileService

from app.utils.telemetry import astep, mark_reuse
from opentelemetry import trace


router = APIRouter(prefix="/api/upload", tags=["Upload"])
logger = logging.getLogger(__name__)


def _files() -> FileService:
    return FileService(
        storage=S3Storage(),
        codec=CsvCodec(index=False),
        compression=GzipCompression(),
    )


@router.post("/", response_model=UploadResponse)
@limiter.limit("5/hour")
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    validated: tuple[bytes, pd.DataFrame] = Depends(validate_csv(["review"])),
    db: AsyncSession = Depends(get_db),
):
    s3_uploaded = False
    s3_key = None
    files = _files()

    try:
        contents, df = validated

        # Enrich the current request span (from FastAPI instrumentation)
        span = trace.get_current_span()
        if span is not None:
            span.set_attribute("app.upload.filename", file.filename or "")
            span.set_attribute("app.upload.size_bytes", len(contents))

        # 1) Hash & dedupe
        async with astep("upload.dedupe", size_bytes=len(contents)):
            file_hash = compute_sha256(contents)
            existing = (
                await db.execute(select(FileRecord).filter_by(file_hash=file_hash))
            ).scalar_one_or_none()

        if existing:
            mark_reuse("upload.dedupe")
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

        # 2) S3 upload via FileService (auto-compress due to .gz key)
        s3_key = f"user-data/{uuid4()}.csv.gz"
        async with astep(
            "upload.s3",
            key=s3_key,
            rows=len(df),
            cols=len(df.columns),
            compressed=True,
        ):
            s3_uploaded = True
            _, s3_url = await files.upload_raw(contents, s3_key, force_compress=True)

        # 3) Save to DB
        async with astep("upload.db.save_file_record"):
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
            await db.commit()
            await db.refresh(file_record)

        logger.info(f"✅ Uploaded and saved {s3_key} ({len(df)} rows)")

        # 4) Build response
        async with astep("upload.response.build"):
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

    except BadRequestError:
        # Already shaped properly; FastAPI instrumentation will mark request span as error.
        raise

    except Exception as e:
        logger.exception(f"❌ Unexpected upload error: {e}")

        # Best-effort S3 rollback
        if s3_uploaded and s3_key:
            async with astep("upload.s3.rollback", key=s3_key):
                ok = await files.safe_delete(s3_key)
                if ok:
                    logger.info(f"🗑️ Rolled back S3 upload: {s3_key}")
                else:
                    logger.error("❗ Failed to delete file from S3 during rollback")

        raise ServerError(code="UPLOAD_FAILED", message=UPLOAD_FAILED)
