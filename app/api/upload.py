from fastapi import APIRouter, Request, UploadFile, File, Depends
from uuid import uuid4
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import time

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
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression
from app.core.file_handler.storage import S3Storage
from app.services.file_service import FileService

router = APIRouter(prefix="/api/upload", tags=["Upload"])
logger = logging.getLogger(__name__)

# --- Metrics ---
_meter = metrics.get_meter(__name__)
_upload_logic_ms = _meter.create_histogram(
    "upload.logic_ms", unit="ms", description="Time spent in upload business steps"
)
_s3_bytes = _meter.create_counter(
    "upload.s3_bytes_total", description="Total gzipped bytes uploaded to S3"
)
_dedupe_hits = _meter.create_counter(
    "upload.dedupe_hits_total", description="Count of duplicate uploads avoided"
)

_tracer = trace.get_tracer(__name__)


def _files() -> FileService:
    # index=False to avoid implicit pandas index
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

    # High-level span to group the whole endpoint
    with _tracer.start_as_current_span("upload.csv") as root_span:
        try:
            contents, df = validated

            # Annotate request (avoid PII: don't send raw names/columns)
            root_span.set_attribute("app.upload.filename", file.filename or "")
            root_span.set_attribute("app.upload.size_bytes", len(contents))

            # Step 1: Hash & dedupe
            t0 = time.perf_counter()
            with _tracer.start_as_current_span("dedupe.check"):
                file_hash = compute_sha256(contents)

                existing = (
                    await db.execute(select(FileRecord).filter_by(file_hash=file_hash))
                ).scalar_one_or_none()

                if existing:
                    _dedupe_hits.add(1)
                    root_span.add_event("dedupe_hit", {"file_id": existing.id})
                    _upload_logic_ms.record(
                        (time.perf_counter() - t0) * 1000, {"step": "dedupe"}
                    )
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
            _upload_logic_ms.record(
                (time.perf_counter() - t0) * 1000, {"step": "dedupe"}
            )

            # Step 2: S3 upload via FileService
            t1 = time.perf_counter()
            with _tracer.start_as_current_span("s3.upload") as s3_span:
                s3_key = f"user-data/{uuid4()}.csv.gz"
                s3_span.set_attribute("app.s3.key", s3_key)

                # (Optional) precise gzipped size metric
                raw_bytes = files.codec.to_bytes(df)
                gz_bytes = (
                    files.compression.compress(raw_bytes)
                    if files.compression
                    else raw_bytes
                )

                # Upload encoded DataFrame (will auto-compress because key endswith .gz)
                _, s3_url = await files.upload_df(df, s3_key)
                s3_span.set_attribute("app.s3.url_set", True)
                s3_uploaded = True
                _s3_bytes.add(len(gz_bytes), {"content": "csv_gz"})
            _upload_logic_ms.record(
                (time.perf_counter() - t1) * 1000, {"step": "s3_upload"}
            )

            # Step 3: Save to DB
            t2 = time.perf_counter()
            with _tracer.start_as_current_span("db.save_file_record"):
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
            _upload_logic_ms.record(
                (time.perf_counter() - t2) * 1000, {"step": "db_save"}
            )

            logger.info(f"✅ Uploaded and saved {s3_key} ({len(df)} rows)")
            root_span.set_status(Status(StatusCode.OK))

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
            root_span.record_exception(he)
            root_span.set_status(Status(StatusCode.ERROR, he.detail))
            logger.warning(f"⚠️ Upload bad request: {he.detail}")
            raise

        except Exception as e:
            root_span.record_exception(e)
            root_span.set_status(Status(StatusCode.ERROR, "unexpected upload error"))
            logger.exception(f"❌ Unexpected upload error: {e}")

            if s3_uploaded and s3_key:
                with _tracer.start_as_current_span("s3.rollback_delete") as rb_span:
                    ok = await files.safe_delete(s3_key)
                    rb_span.set_attribute("app.s3.rollback_deleted", bool(ok))
                    if ok:
                        logger.info(f"🗑️ Rolled back S3 upload: {s3_key}")
                    else:
                        rb_span.set_status(
                            Status(StatusCode.ERROR, "s3 rollback failed")
                        )
                        logger.error("❗ Failed to delete file from S3 during rollback")

            raise ServerError(code="UPLOAD_FAILED", message=UPLOAD_FAILED)
