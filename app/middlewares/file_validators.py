import time
import asyncio
import pandas as pd
from io import StringIO
from typing import List, Tuple
from fastapi import UploadFile, File, HTTPException, Depends
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = settings.MAX_SIZE_FILE_UPLOAD or 5
ALLOWED_CONTENT_TYPES = ["text/csv"]


def validate_extension(file: UploadFile = File(...)) -> UploadFile:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files are allowed.",
        )
    return file


async def validate_file_size(file: UploadFile = Depends(validate_extension)) -> bytes:
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )
    return contents


async def validate_required_columns(
    contents: bytes,
    required_columns: List[str] = ["review"],
) -> pd.DataFrame:
    try:
        decoded = contents.decode("utf-8")

        def parse_csv():
            return pd.read_csv(StringIO(decoded))

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, parse_csv)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}",
        )
    return df


async def validate_required_columns_not_empty(
    df: pd.DataFrame,
    contents: bytes,
    required_columns: List[str] = ["review"],
) -> Tuple[bytes, pd.DataFrame]:
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file contains no rows.",
        )

    for col in required_columns:
        if df[col].dropna().astype(str).str.strip().eq("").all():
            raise HTTPException(
                status_code=400,
                detail=f"The column '{col}' exists but contains no usable data.",
            )

    return contents, df


def validate_csv(required_columns: List[str] = ["review"]):
    async def dependency(
        file: UploadFile = File(...),
    ) -> Tuple[bytes, pd.DataFrame]:
        start = time.time()

        file_checked = validate_extension(file)
        contents = await validate_file_size(file_checked)
        df = await validate_required_columns(contents, required_columns)
        contents, df_checked = await validate_required_columns_not_empty(
            df, contents, required_columns
        )

        logger.info(f"✅ Validate Middleware success in {time.time() - start:.2f}s")
        return contents, df_checked

    return dependency
