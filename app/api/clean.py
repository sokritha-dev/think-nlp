from io import BytesIO
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
import pandas as pd
from app.models.db.file_record import FileRecord
from app.schemas.clean import (
    LemmatizeRequest,
    LemmatizedData,
    LemmatizedResponse,
    NormalizeRequest,
    NormalizeResponse,
    SpecialCharCleanedData,
    SpecialCharCleanedResponse,
    SpecialCleanRequest,
    StopwordTokenRequest,
    StopwordTokenResponse,
    StopwordTokenResponseData,
    TokenizeRequest,
    TokenizedResponse,
)
from app.services.preprocess import (
    lemmatize_tokens,
    normalize_text,
    remove_special_characters,
    remove_stopwords_from_tokens,
    tokenize_text,
)
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.s3_uploader import download_file_from_s3, upload_file_to_s3

import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clean", tags=["Cleaning"])


@router.post("/normalize", response_model=NormalizeResponse)
async def normalize_reviews_by_file_id(
    req: NormalizeRequest, db: Session = Depends(get_db)
):
    try:
        file_id = req.file_id
        broken_map = req.broken_map or {}

        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="File ID not found")

        # Download original CSV
        file_bytes = download_file_from_s3(record.s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # Normalize using custom map
        normalized_df = df[["review"]].dropna().copy()
        normalized_df["normalized_review"] = normalized_df["review"].apply(
            lambda r: normalize_text(r, broken_map)
        )

        # Save normalized CSV to S3
        output_buffer = BytesIO()
        normalized_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"normalization/normalized_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        # Update DB
        record.normalized_s3_key = new_s3_key
        record.normalized_s3_url = s3_url
        db.commit()

        return NormalizeResponse(
            status="success",
            message="Reviews normalized and saved to S3.",
            data={
                "file_id": file_id,
                "normalized_s3_url": s3_url,
                "record_count": len(normalized_df),
                "columns": normalized_df.columns.tolist(),
            },
        )
    except Exception as e:
        logger.exception(f"❌ Normalization failed::: {e}")
        raise HTTPException(status_code=500, detail="Normalization failed.")


@router.post("/remove-special", response_model=SpecialCharCleanedResponse)
async def remove_special_characters_by_file(
    req: SpecialCleanRequest, db: Session = Depends(get_db)
):
    try:
        record = db.query(FileRecord).filter_by(id=req.file_id).first()
        if not record or not record.normalized_s3_key:
            raise HTTPException(
                status_code=404,
                detail="File not found or normalization must be done first.",
            )

        # ✅ Download the normalized file from S3
        file_bytes = download_file_from_s3(record.normalized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # ✅ Clean the normalized review column
        cleaned_rows = []
        removed_chars = []
        for row in df["normalized_review"].dropna():
            cleaned, removed = remove_special_characters(
                row,
                remove_special=req.remove_special,
                remove_numbers=req.remove_numbers,
                remove_emoji=req.remove_emoji,
            )
            cleaned_rows.append({"normalized_review": row, "special_cleaned": cleaned})
            removed_chars.extend(removed)

        cleaned_df = pd.DataFrame(cleaned_rows)

        # ✅ Upload new cleaned file to S3
        output_buffer = BytesIO()
        cleaned_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"cleaned/special_cleaned_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        # ✅ Save to DB
        record.special_cleaned_s3_key = new_s3_key
        record.special_cleaned_s3_url = s3_url
        db.commit()

        return SpecialCharCleanedResponse(
            status="success",
            message="Special characters removed from normalized text and saved to S3.",
            data=SpecialCharCleanedData(
                file_id=record.id,
                cleaned_s3_url=s3_url,
                record_count=len(cleaned_df),
                columns=cleaned_df.columns.tolist(),
                removed_characters=list(set(removed_chars)),
            ),
        )

    except Exception as e:
        logger.exception(f"❌ Failed to clean normalized text::: {e}")
        raise HTTPException(status_code=500, detail="Special cleaning failed.")


@router.post("/tokenize", response_model=TokenizedResponse)
async def tokenize_reviews_by_file_id(
    req: TokenizeRequest, db: Session = Depends(get_db)
):
    try:
        # 1. Fetch record
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.special_cleaned_s3_key:
            raise HTTPException(status_code=404, detail="Cleaned file not found")

        # 2. Download the cleaned file
        file_bytes = download_file_from_s3(record.special_cleaned_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # 3. Tokenize the cleaned text
        df["tokens"] = df["special_cleaned"].dropna().apply(tokenize_text)

        # ✅ 4. Only keep relevant columns
        df_filtered = df[["special_cleaned", "tokens"]]

        # 5. Save new result to S3
        output_buffer = BytesIO()
        df_filtered.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"tokenization/tokens_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        # 6. Update database
        record.tokenized_s3_key = new_s3_key
        record.tokenized_s3_url = s3_url
        db.commit()

        return TokenizedResponse(
            status="success",
            message="Cleaned reviews tokenized and saved to S3.",
            data={
                "file_id": file_id,
                "tokenized_s3_url": s3_url,
                "record_count": len(df_filtered),
                "columns": df_filtered.columns.tolist(),
            },
        )

    except Exception as e:
        logger.exception(f"❌ Tokenization failed::: {e}")
        raise HTTPException(status_code=500, detail="Tokenization failed.")


@router.post("/remove-stopwords", response_model=StopwordTokenResponse)
async def remove_stopwords_by_file_id(
    req: StopwordTokenRequest, db: Session = Depends(get_db)
):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.tokenized_s3_key:
            raise HTTPException(status_code=404, detail="Tokenized file not found")

        # Download and read tokenized data
        file_bytes = download_file_from_s3(record.tokenized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["tokens"] = df["tokens"].apply(eval)

        # Remove stopwords using your helper
        stopword_results = df["tokens"].apply(
            lambda toks: remove_stopwords_from_tokens(
                tokens=toks,
                custom_stopwords=req.custom_stopwords or [],
                exclude_stopwords=req.exclude_stopwords or [],
            )
        )

        # Extract cleaned tokens only
        df["stopword_removed"] = stopword_results.map(lambda r: r["cleaned_tokens"])

        # Save to new CSV
        output_buffer = BytesIO()
        df[["tokens", "stopword_removed"]].to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"stopwords/stopword_removed_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        # Update DB
        record.stopword_s3_key = new_s3_key
        record.stopword_s3_url = s3_url
        db.commit()

        return StopwordTokenResponse(
            status="success",
            message="Stopwords removed and saved to S3.",
            data=StopwordTokenResponseData(
                file_id=req.file_id,
                tokenized_s3_url=record.tokenized_s3_url,
                stopword_s3_url=s3_url,
                record_count=len(df),
                columns=["tokens", "stopword_removed"],
            ),
        )

    except Exception as e:
        logger.exception(f"❌ Stopword removal failed::: {e}")
        raise HTTPException(status_code=500, detail="Stopword removal failed.")


@router.post("/lemmatize", response_model=LemmatizedResponse)
async def lemmatize_by_file_id(req: LemmatizeRequest, db: Session = Depends(get_db)):
    try:
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.stopword_s3_key:
            raise HTTPException(
                status_code=404, detail="Stopword-removed file not found"
            )

        # 1. Download from S3 and load CSV
        file_bytes = download_file_from_s3(record.stopword_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["stopword_removed"] = df["stopword_removed"].apply(eval)

        # 2. Apply lemmatization
        df["lemmatized_tokens"] = df["stopword_removed"].apply(
            lambda tokens: lemmatize_tokens(tokens)["lemmatized_tokens"]
        )

        # 3. Upload new file to S3
        output_buffer = BytesIO()
        df[["stopword_removed", "lemmatized_tokens"]].to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"lemmatization/lemmatized_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        # 4. Save to DB
        record.lemmatized_s3_key = new_s3_key
        record.lemmatized_s3_url = s3_url
        db.commit()

        return LemmatizedResponse(
            status="success",
            message="Lemmatized tokens saved to S3.",
            data=LemmatizedData(
                file_id=file_id,
                tokenized_s3_url=record.stopword_s3_url,
                lemmatized_s3_url=s3_url,
                record_count=len(df),
                columns=["stopword_removed", "lemmatized_tokens"],
            ),
        )

    except Exception as e:
        logger.exception(f"❌ Lemmatization failed::: {e}")
        raise HTTPException(status_code=500, detail="Lemmatization failed.")
