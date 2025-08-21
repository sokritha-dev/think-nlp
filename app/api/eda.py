from __future__ import annotations
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression
from app.core.file_handler.storage import S3Storage
from app.core.eda.config import EDAConfig
from app.core.eda.eda_analyzer import DefaultEDAAnalyzer
from app.messages.eda_messages import (
    FILE_NOT_FOUND_FOR_EDA,
    LEMMATIZED_FILE_NOT_FOUND,
    EDA_GENERATION_SUCCESS,
)
from app.models.db.file_record import FileRecord
from app.schemas.eda import EDARequest
from app.services.file_service import FileService
from app.services.eda_service import EDAService
from app.utils.exceptions import NotFoundError, ServerError
from app.utils.response_builder import success_response

router = APIRouter(prefix="/api/eda", tags=["EDA"])
logger = logging.getLogger(__name__)


def _files() -> FileService:
    return FileService(
        storage=S3Storage(),
        codec=CsvCodec(index=False),
        compression=GzipCompression(),
    )


@router.post("/summary")
async def generate_eda(req: EDARequest, db: AsyncSession = Depends(get_db)):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND_FOR_EDA)
        if not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND", message=LEMMATIZED_FILE_NOT_FOUND
            )

        files = _files()
        analyzer = DefaultEDAAnalyzer(
            EDAConfig(text_column="lemmatized_tokens", top_words=100, ngram_top_k=20)
        )

        eda_service = EDAService(analyzer=analyzer, files=files)

        # df_lemm=None means the service will load via FileService
        res = await eda_service.ensure_eda(
            db=db,
            record=record,
            df_lemm=None,
        )

        return success_response(
            message=EDA_GENERATION_SUCCESS,
            data={
                "file_id": record.id,
                **res.eda,  # word_cloud, length_distribution, bigrams, trigrams
                "reused": res.reused,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to generate EDA for file {req.file_id}: {e}")
        raise ServerError(code="EDA_FAILED", message="Failed to generate EDA.")
