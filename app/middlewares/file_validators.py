# app/middlewares/file_validators.py

from fastapi import UploadFile, File, HTTPException

MAX_FILE_SIZE_MB = 5  # You can adjust this limit
ALLOWED_CONTENT_TYPES = ["text/csv"]


def validate_csv(file: UploadFile = File(...)) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files are allowed.",
        )

    contents = file.file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    return contents
