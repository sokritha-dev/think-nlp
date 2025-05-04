# app/api/clean.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from uuid import uuid4
from io import BytesIO
import json
import logging
import pandas as pd

from app.core.database import get_db
from app.messages.clean_messages import (
    CLEANED_FILE_NOT_FOUND,
    FILE_NOT_FOUND,
    LEMMATIZATION_ALREADY_EXISTS,
    LEMMATIZATION_SUCCESS,
    NORMALIZATION_ALREADY_EXISTS,
    NORMALIZATION_SUCCESS,
    NORMALIZED_FILE_NOT_FOUND,
    SPECIAL_CLEAN_ALREADY_EXISTS,
    SPECIAL_CLEAN_SUCCESS,
    STOPWORD_ALREADY_EXISTS,
    STOPWORD_FILE_NOT_FOUND,
    STOPWORD_SUCCESS,
    TOKENIZATION_ALREADY_EXISTS,
    TOKENIZATION_SUCCESS,
    TOKENIZED_FILE_NOT_FOUND,
)
from app.models.db.file_record import FileRecord
from app.schemas.clean import (
    LemmatizeRequest,
    NormalizeRequest,
    SpecialCleanRequest,
    StopwordTokenRequest,
    TokenizeRequest,
)
from app.services.preprocess import (
    lemmatize_tokens,
    normalize_text,
    remove_special_characters,
    remove_stopwords_from_tokens,
    tokenize_text,
)
from app.services.s3_uploader import (
    download_file_from_s3,
    generate_presigned_url,
    upload_file_to_s3,
    delete_file_from_s3,
)
from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError

router = APIRouter(prefix="/api/clean", tags=["Cleaning"])
logger = logging.getLogger(__name__)


@router.post("/normalize")
async def normalize_reviews_by_file_id(
    req: NormalizeRequest, db: Session = Depends(get_db)
):
    try:
        file_id = req.file_id
        broken_map = req.broken_map or {}
        broken_map_json = json.dumps(broken_map, sort_keys=True)

        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        # Reuse existing normalized data if broken_map is unchanged
        if record.normalized_s3_key and record.normalized_broken_map == broken_map_json:
            presigned_url = generate_presigned_url(
                bucket="nlp-learner", key=record.normalized_s3_key
            )
            preview_bytes = download_file_from_s3(record.normalized_s3_key)
            df = pd.read_csv(BytesIO(preview_bytes))
            return success_response(
                message=NORMALIZATION_ALREADY_EXISTS,
                data={
                    "file_id": file_id,
                    "normalized_s3_url": presigned_url,
                    "columns": ["review", "normalized_review"],
                    "broken_words": record.normalized_broken_map,
                    "before": df["review"].dropna().tolist()[:20],
                    "after": df["normalized_review"].dropna().tolist()[:20],
                    "total_records": record.record_count,
                },
            )

        # Delete previous if exists
        if record.normalized_s3_key:
            try:
                delete_file_from_s3(record.normalized_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old normalized file: {e}")

        file_bytes = download_file_from_s3(record.s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        normalized_df = df[["review"]].dropna().copy()
        normalized_df["normalized_review"] = normalized_df["review"].apply(
            lambda r: normalize_text(r, broken_map)
        )

        output_buffer = BytesIO()
        normalized_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"normalization/normalized_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)
        presigned_url = generate_presigned_url(bucket="nlp-learner", key=new_s3_key)

        record.normalized_s3_key = new_s3_key
        record.normalized_s3_url = s3_url
        record.normalized_broken_map = broken_map_json
        db.commit()

        return success_response(
            message=NORMALIZATION_SUCCESS,
            data={
                "file_id": file_id,
                "normalized_s3_url": presigned_url,
                "total_records": len(normalized_df),
                "columns": normalized_df.columns.tolist(),
                "broken_words": broken_map,
                "before": normalized_df["review"].tolist()[:20],
                "after": normalized_df["normalized_review"].tolist()[:20],
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Normalization failed: {e}")
        raise ServerError(code="NORMALIZATION_FAILED", message="Normalization failed.")


@router.get("/normalize")
async def get_normalized_reviews(
    file_id: str,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    try:
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        if not record.normalized_s3_key:
            raise NotFoundError(
                code="NORMALIZATION_NOT_FOUND",
                message="Normalization not found for this file.",
            )

        file_bytes = download_file_from_s3(record.normalized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        presigned_url = generate_presigned_url(
            bucket="nlp-learner", key=record.normalized_s3_key, expires_in=6000
        )

        start = (page - 1) * limit
        end = start + limit

        return success_response(
            message=NORMALIZATION_ALREADY_EXISTS,
            data={
                "file_id": file_id,
                "normalized_s3_url": presigned_url,
                "columns": ["review", "normalized_review"],
                "broken_words": json.loads(record.normalized_broken_map or "{}"),
                "before": df["review"].dropna().tolist()[start:end],
                "after": df["normalized_review"].dropna().tolist()[start:end],
                "total_records": len(df),
                "page": page,
                "limit": limit,
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Normalization preview failed: {e}")
        raise ServerError(code="NORMALIZATION_FAILED", message="Normalization failed.")


@router.post("/remove-special")
async def remove_special_characters_by_file(
    req: SpecialCleanRequest, db: Session = Depends(get_db)
):
    try:
        record = db.query(FileRecord).filter_by(id=req.file_id).first()
        if not record or not record.normalized_s3_key:
            raise NotFoundError(
                code="NORMALIZED_FILE_NOT_FOUND", message=NORMALIZED_FILE_NOT_FOUND
            )

        flags = {
            "remove_special": req.remove_special,
            "remove_numbers": req.remove_numbers,
            "remove_emoji": req.remove_emoji,
        }
        flags_json = json.dumps(flags, sort_keys=True)

        # Use cached result if flags are the same
        if record.special_cleaned_s3_key and record.special_cleaned_flags == flags_json:
            presigned_url = generate_presigned_url(
                "nlp-learner", record.special_cleaned_s3_key
            )
            preview_bytes = download_file_from_s3(record.special_cleaned_s3_key)
            df = pd.read_csv(BytesIO(preview_bytes))
            return success_response(
                message=SPECIAL_CLEAN_ALREADY_EXISTS,
                data={
                    "file_id": record.id,
                    "cleaned_s3_url": presigned_url,
                    "total_records": len(df),
                    "removed_characters": list(
                        set(json.loads(record.special_cleaned_removed or "[]"))
                    ),
                    "before": df["normalized_review"].dropna().tolist()[:20],
                    "after": df["special_cleaned"].dropna().tolist()[:20],
                },
            )

        # Delete previous cleaned file if flags changed
        if record.special_cleaned_s3_key:
            try:
                delete_file_from_s3(record.special_cleaned_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old special cleaned file: {e}")

        # Read normalized file
        file_bytes = download_file_from_s3(record.normalized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # Apply cleaning
        cleaned_rows, removed_chars = [], []
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

        # Upload new cleaned file
        output_buffer = BytesIO()
        cleaned_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"cleaned/special_cleaned_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)
        presigned_url = generate_presigned_url("nlp-learner", new_s3_key)

        # Update DB record
        record.special_cleaned_s3_key = new_s3_key
        record.special_cleaned_s3_url = s3_url
        record.special_cleaned_flags = flags_json
        record.special_cleaned_removed = json.dumps(list(set(removed_chars)))
        record.special_cleaned_updated_at = datetime.now()
        db.commit()

        return success_response(
            message=SPECIAL_CLEAN_SUCCESS,
            data={
                "file_id": record.id,
                "cleaned_s3_url": presigned_url,
                "total_records": len(cleaned_df),
                "removed_characters": list(set(removed_chars)),
                "before": cleaned_df["normalized_review"].tolist()[:20],
                "after": cleaned_df["special_cleaned"].tolist()[:20],
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Special character cleaning failed: {e}")
        raise ServerError(
            code="SPECIAL_CLEAN_FAILED", message="Special character cleaning failed."
        )


@router.get("/remove-special")
async def get_special_cleaned_reviews(
    file_id: str,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    try:
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="SPECIAL_CLEAN_NOT_FOUND",
                message="Special cleaned file not found for this file ID.",
            )

        presigned_url = generate_presigned_url(
            "nlp-learner", record.special_cleaned_s3_key
        )

        # Load preview from S3
        file_bytes = download_file_from_s3(record.special_cleaned_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        start = (page - 1) * page_size
        end = start + page_size
        before = df["normalized_review"].dropna().tolist()[start:end]
        after = df["special_cleaned"].dropna().tolist()[start:end]

        return success_response(
            message=SPECIAL_CLEAN_ALREADY_EXISTS,
            data={
                "file_id": record.id,
                "cleaned_s3_url": presigned_url,
                "removed_characters": list(
                    set(json.loads(record.special_cleaned_removed or "[]"))
                ),
                "before": before,
                "after": after,
                "total_records": len(df),
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Failed to get special cleaned preview: {e}")
        raise ServerError(
            code="GET_SPECIAL_CLEAN_FAILED",
            message="Failed to get special cleaned results.",
        )


@router.post("/tokenize")
async def tokenize_reviews_by_file_id(
    req: TokenizeRequest, db: Session = Depends(get_db)
):
    try:
        record = db.query(FileRecord).filter_by(id=req.file_id).first()
        if not record or not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="CLEANED_FILE_NOT_FOUND", message=CLEANED_FILE_NOT_FOUND
            )

        if record.tokenized_updated_at and record.special_cleaned_updated_at:
            if record.tokenized_updated_at >= record.special_cleaned_updated_at:
                return success_response(
                    message=TOKENIZATION_ALREADY_EXISTS,
                    data={
                        "file_id": record.id,
                        "tokenized_s3_url": record.tokenized_s3_url,
                        "columns": ["special_cleaned", "tokens"],
                    },
                )

        if record.tokenized_s3_key:
            try:
                delete_file_from_s3(record.tokenized_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old tokenized file: {e}")

        file_bytes = download_file_from_s3(record.special_cleaned_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        df["tokens"] = df["special_cleaned"].dropna().apply(tokenize_text)
        df_filtered = df[["special_cleaned", "tokens"]]

        output_buffer = BytesIO()
        df_filtered.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"tokenization/tokens_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        record.tokenized_s3_key = new_s3_key
        record.tokenized_s3_url = s3_url
        record.tokenized_updated_at = datetime.now()
        db.commit()

        return success_response(
            message=TOKENIZATION_SUCCESS,
            data={
                "file_id": record.id,
                "tokenized_s3_url": s3_url,
                "record_count": len(df_filtered),
                "columns": df_filtered.columns.tolist(),
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Tokenization failed: {e}")
        raise ServerError(code="TOKENIZATION_FAILED", message="Tokenization failed.")


@router.post("/remove-stopwords")
async def remove_stopwords_by_file_id(
    req: StopwordTokenRequest, db: Session = Depends(get_db)
):
    try:
        record = db.query(FileRecord).filter_by(id=req.file_id).first()
        if not record or not record.tokenized_s3_key:
            raise NotFoundError(
                code="TOKENIZED_FILE_NOT_FOUND", message=TOKENIZED_FILE_NOT_FOUND
            )

        current_config = json.dumps(
            {
                "custom_stopwords": req.custom_stopwords or [],
                "exclude_stopwords": req.exclude_stopwords or [],
            }
        )

        if (
            record.stopword_s3_key
            and record.tokenized_updated_at
            and record.stopword_updated_at
        ):
            if (
                record.stopword_updated_at >= record.tokenized_updated_at
                and record.stopword_config == current_config
            ):
                return success_response(
                    message=STOPWORD_ALREADY_EXISTS,
                    data={
                        "file_id": record.id,
                        "stopword_s3_url": record.stopword_s3_url,
                        "columns": ["tokens", "stopword_removed"],
                    },
                )

        if record.stopword_s3_key:
            try:
                delete_file_from_s3(record.stopword_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old stopword file: {e}")

        file_bytes = download_file_from_s3(record.tokenized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["tokens"] = df["tokens"].apply(eval)

        stopword_results = df["tokens"].apply(
            lambda toks: remove_stopwords_from_tokens(
                tokens=toks,
                custom_stopwords=req.custom_stopwords or [],
                exclude_stopwords=req.exclude_stopwords or [],
            )
        )
        df["stopword_removed"] = stopword_results.map(lambda r: r["cleaned_tokens"])

        output_buffer = BytesIO()
        df[["tokens", "stopword_removed"]].to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"stopwords/stopword_removed_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        record.stopword_s3_key = new_s3_key
        record.stopword_s3_url = s3_url
        record.stopword_config = current_config
        record.stopword_updated_at = datetime.now()
        db.commit()

        return success_response(
            message=STOPWORD_SUCCESS,
            data={
                "file_id": record.id,
                "stopword_s3_url": s3_url,
                "record_count": len(df),
                "columns": ["tokens", "stopword_removed"],
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Stopword removal failed: {e}")
        raise ServerError(
            code="STOPWORD_REMOVAL_FAILED", message="Stopword removal failed."
        )


@router.post("/lemmatize")
async def lemmatize_by_file_id(req: LemmatizeRequest, db: Session = Depends(get_db)):
    try:
        record = db.query(FileRecord).filter_by(id=req.file_id).first()
        if not record or not record.stopword_s3_key:
            raise NotFoundError(
                code="STOPWORD_FILE_NOT_FOUND", message=STOPWORD_FILE_NOT_FOUND
            )

        if (
            record.lemmatized_s3_key
            and record.stopword_updated_at
            and record.lemmatized_updated_at
        ):
            if record.lemmatized_updated_at >= record.stopword_updated_at:
                return success_response(
                    message=LEMMATIZATION_ALREADY_EXISTS,
                    data={
                        "file_id": record.id,
                        "lemmatized_s3_url": record.lemmatized_s3_url,
                        "columns": ["stopword_removed", "lemmatized_tokens"],
                    },
                )

        if record.lemmatized_s3_key:
            try:
                delete_file_from_s3(record.lemmatized_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old lemmatized file: {e}")

        file_bytes = download_file_from_s3(record.stopword_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["stopword_removed"] = df["stopword_removed"].apply(eval)

        df["lemmatized_tokens"] = df["stopword_removed"].apply(
            lambda tokens: lemmatize_tokens(tokens)["lemmatized_tokens"]
        )

        output_buffer = BytesIO()
        df[["stopword_removed", "lemmatized_tokens"]].to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"lemmatization/lemmatized_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        record.lemmatized_s3_key = new_s3_key
        record.lemmatized_s3_url = s3_url
        record.lemmatized_updated_at = datetime.now()
        db.commit()

        return success_response(
            message=LEMMATIZATION_SUCCESS,
            data={
                "file_id": record.id,
                "lemmatized_s3_url": s3_url,
                "record_count": len(df),
                "columns": ["stopword_removed", "lemmatized_tokens"],
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Lemmatization failed: {e}")
        raise ServerError(code="LEMMATIZATION_FAILED", message="Lemmatization failed.")
