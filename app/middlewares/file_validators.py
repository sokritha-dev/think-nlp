# app/middlewares/file_validators.py

from typing import List
from fastapi import UploadFile, File, HTTPException, Depends
import pandas as pd
from io import StringIO
from app.core.config import settings


MAX_FILE_SIZE_MB = settings.MAX_SIZE_FILE_UPLOAD | 5
ALLOWED_CONTENT_TYPES = ["text/csv"]


def validate_extension(file: UploadFile = File(...)) -> UploadFile:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files are allowed.",
        )
    return file


def validate_file_size(file: UploadFile = Depends(validate_extension)) -> bytes:
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )
    return contents


def validate_required_columns(
    contents: bytes = Depends(validate_file_size),
    required_columns: List[str] = ["review"],
) -> pd.DataFrame:
    try:
        decoded = contents.decode("utf-8")
        df = pd.read_csv(StringIO(decoded))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}",
        )
    return df


def validate_required_columns_not_empty(
    df: pd.DataFrame,
    contents: bytes,
    required_columns: List[str] = ["review"],
) -> tuple[bytes, pd.DataFrame]:
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
    def dependency(
        file: UploadFile = File(...),
    ) -> tuple[bytes, pd.DataFrame]:
        file_checked = validate_extension(file)
        contents = validate_file_size(file_checked)
        df = validate_required_columns(contents, required_columns)
        contents, df_checked = validate_required_columns_not_empty(
            df, contents, required_columns
        )
        return contents, df_checked

    return dependency
