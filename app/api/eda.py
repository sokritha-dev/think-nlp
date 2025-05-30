from datetime import datetime
import gzip
import json
from fastapi import APIRouter, Depends
import pandas as pd
from io import BytesIO
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from app.core.database import get_db
from app.models.db.file_record import FileRecord
from app.schemas.eda import EDARequest
from app.services.s3_uploader import download_file_from_s3
from app.eda.eda_analysis import EDA

from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError
from app.messages.eda_messages import (
    FILE_NOT_FOUND_FOR_EDA,
    LEMMATIZED_FILE_NOT_FOUND,
    EDA_GENERATION_SUCCESS,
)

router = APIRouter(prefix="/api/eda", tags=["EDA"])
logger = logging.getLogger(__name__)


@router.post("/summary")
async def generate_eda(req: EDARequest, db: AsyncSession = Depends(get_db)):
    try:
        file_id = req.file_id
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND_FOR_EDA)

        if not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND", message=LEMMATIZED_FILE_NOT_FOUND
            )

        if record.lemmatized_updated_at < record.eda_updated_at:
            eda_result = json.loads(record.eda_analysis)

            return success_response(
                message=EDA_GENERATION_SUCCESS,
                data={
                    "file_id": file_id,
                    **eda_result,
                },
            )

        # Step 1: Download and decompress
        file_bytes = await download_file_from_s3(record.lemmatized_s3_key)
        if record.lemmatized_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        # Step 2: Run EDA (returns structured data now)
        eda = EDA(df=df, file_id=file_id)
        eda_result = eda.run_eda()  # returns dict of lists

        # Step 3: Optional metadata (for audit/update tracking)
        record.eda_analysis = json.dumps(eda_result)
        record.eda_updated_at = datetime.now()
        await db.commit()

        logger.info(f"✅ EDA (data) completed for file {file_id}")

        return success_response(
            message=EDA_GENERATION_SUCCESS,
            data={
                "file_id": file_id,
                **eda_result,
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Failed to generate EDA for file {req.file_id}: {e}")
        raise ServerError(code="EDA_FAILED", message="Failed to generate EDA.")
