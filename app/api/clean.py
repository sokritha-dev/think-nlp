import ast
import gzip
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from uuid import uuid4
from io import BytesIO
import json
import logging
import pandas as pd

from app.core.config import settings
from app.core.database import get_db
from app.messages.clean_messages import (
    CLEANED_FILE_NOT_FOUND,
    DATA_CLEANING_STATUS,
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
import time

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


@router.get("/status")
async def get_cleaning_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await db.execute(select(FileRecord).filter_by(id=file_id))
        record = result.scalars().first()

        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        steps_status = [
            {"step": "normalized", "should_recompute": False},
            {
                "step": "special_chars",
                "should_recompute": record.normalized_updated_at
                > record.special_cleaned_updated_at,
            },
            {
                "step": "tokenized",
                "should_recompute": record.special_cleaned_updated_at
                > record.tokenized_updated_at,
            },
            {
                "step": "stopwords",
                "should_recompute": record.tokenized_updated_at
                > record.stopword_updated_at,
            },
            {
                "step": "lemmatized",
                "should_recompute": record.stopword_updated_at
                > record.lemmatized_updated_at,
            },
        ]

        return success_response(
            message=DATA_CLEANING_STATUS,
            data={"steps": steps_status},
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Fetching data cleaning status failed: {e}")
        raise ServerError(
            code="CLEANING_STATUS_FAILED",
            message="Could not fetch data cleaning step status.",
        )


@router.post("/normalize")
async def normalize_reviews_by_file_id(
    req: NormalizeRequest, db: AsyncSession = Depends(get_db)
):
    try:
        file_id = req.file_id
        broken_map = req.broken_map or {}
        broken_map_json = json.dumps(broken_map, sort_keys=True)

        result = await db.execute(select(FileRecord).filter_by(id=file_id))
        record = result.scalars().first()

        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        # ✅ Use existing normalized version if config hasn't changed
        if record.normalized_s3_key and record.normalized_broken_map == broken_map_json:
            presigned_url = await generate_presigned_url(
                bucket=settings.AWS_S3_BUCKET_NAME, key=record.normalized_s3_key
            )
            # Decompress and load existing normalized file
            with gzip.GzipFile(
                fileobj=BytesIO(await download_file_from_s3(record.normalized_s3_key))
            ) as gz:
                df = pd.read_csv(gz)

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
                    "should_recompute": False,
                },
            )

        # 🔁 Delete previous version if exists
        if record.normalized_s3_key:
            try:
                await delete_file_from_s3(record.normalized_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old normalized file: {e}")

        # 📥 Decompress the uploaded source file
        with gzip.GzipFile(
            fileobj=BytesIO(await download_file_from_s3(record.s3_key))
        ) as gz:
            df = pd.read_csv(gz)

        normalized_df = df[["review"]].dropna().copy()
        normalized_df["normalized_review"] = normalized_df["review"].apply(
            lambda r: normalize_text(r, broken_map)
        )

        # 📤 Compress the new output before upload
        output_buffer = BytesIO()
        with gzip.GzipFile(fileobj=output_buffer, mode="wb") as gz_out:
            normalized_df.to_csv(gz_out, index=False)
        output_buffer.seek(0)

        new_s3_key = f"normalization/normalized_{uuid4()}.csv.gz"
        s3_url = await upload_file_to_s3(
            output_buffer, new_s3_key, content_type="application/gzip"
        )
        presigned_url = await generate_presigned_url(
            bucket=settings.AWS_S3_BUCKET_NAME, key=new_s3_key
        )

        # 🧠 Update DB record
        record.normalized_s3_key = new_s3_key
        record.normalized_s3_url = s3_url
        record.normalized_broken_map = broken_map_json
        record.normalized_updated_at = datetime.now()
        await db.commit()

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
                "should_recompute": False,
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
    db: AsyncSession = Depends(get_db),
):
    try:
        start = time.time()
        result = await db.execute(select(FileRecord).filter_by(id=file_id))
        record = result.scalars().first()
        logger.info(f"Database request time: {time.time() - start:.2f}s")
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        if not record.normalized_s3_key:
            raise NotFoundError(
                code="NORMALIZATION_NOT_FOUND",
                message="Normalization not found for this file.",
            )

        # ✅ Download and decompress gzip file
        start = time.time()
        file_bytes = await download_file_from_s3(record.normalized_s3_key)
        logger.info(f"S3 Download request time: {time.time() - start:.2f}s")

        start = time.time()
        with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
            df = pd.read_csv(gz)
        logger.info(
            f"Decompress & Access File request time: {time.time() - start:.2f}s"
        )

        presigned_url = await generate_presigned_url(
            bucket=settings.AWS_S3_BUCKET_NAME,
            key=record.normalized_s3_key,
            expires_in=6000,
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
                "should_recompute": False,
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Normalization preview failed: {e}")
        raise ServerError(code="NORMALIZATION_FAILED", message="Normalization failed.")


@router.post("/remove-special")
async def remove_special_characters_by_file(
    req: SpecialCleanRequest, db: AsyncSession = Depends(get_db)
):
    try:
        result = await db.execute(select(FileRecord).filter_by(id=req.file_id))
        record = result.scalars().first()

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
        if (
            record.special_cleaned_s3_key
            and record.special_cleaned_flags == flags_json
            and record.normalized_updated_at < record.special_cleaned_updated_at
        ):
            presigned_url = await generate_presigned_url(
                settings.AWS_S3_BUCKET_NAME, record.special_cleaned_s3_key
            )
            preview_bytes = await download_file_from_s3(record.special_cleaned_s3_key)

            with gzip.GzipFile(fileobj=BytesIO(preview_bytes)) as gz:
                df = pd.read_csv(gz)

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
                    "flags": flags,
                    "should_recompute": False,
                },
            )

        # Delete old cleaned file
        if record.special_cleaned_s3_key:
            try:
                await delete_file_from_s3(record.special_cleaned_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old special cleaned file: {e}")

        # Read normalized file (support .gz)
        file_bytes = await download_file_from_s3(record.normalized_s3_key)
        if record.normalized_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
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

        # Save as compressed gzip
        output_buffer = BytesIO()

        with gzip.GzipFile(fileobj=output_buffer, mode="wb") as gz:
            cleaned_df.to_csv(gz, index=False)
        output_buffer.seek(0)

        new_s3_key = f"cleaned/special_cleaned_{uuid4()}.csv.gz"
        s3_url = await upload_file_to_s3(
            output_buffer, new_s3_key, content_type="application/gzip"
        )
        presigned_url = await generate_presigned_url(
            settings.AWS_S3_BUCKET_NAME, new_s3_key
        )

        # Save metadata
        record.special_cleaned_s3_key = new_s3_key
        record.special_cleaned_s3_url = s3_url
        record.special_cleaned_flags = flags_json
        record.special_cleaned_removed = json.dumps(list(set(removed_chars)))
        record.special_cleaned_updated_at = datetime.now()

        await db.commit()

        return success_response(
            message=SPECIAL_CLEAN_SUCCESS,
            data={
                "file_id": record.id,
                "cleaned_s3_url": presigned_url,
                "total_records": len(cleaned_df),
                "removed_characters": list(set(removed_chars)),
                "before": cleaned_df["normalized_review"].tolist()[:20],
                "after": cleaned_df["special_cleaned"].tolist()[:20],
                "flags": flags,
                "should_recompute": False,
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Special character cleaning failed: {e}")
        raise ServerError(
            code="SPECIAL_CLEAN_FAILED",
            message="Special character cleaning failed.",
        )


@router.get("/remove-special")
async def get_special_cleaned_reviews(
    file_id: str,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record or not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="SPECIAL_CLEAN_NOT_FOUND",
                message="Special cleaned file not found for this file ID.",
            )

        should_recompute = False
        if record.normalized_updated_at and record.special_cleaned_updated_at:
            should_recompute = (
                record.normalized_updated_at > record.special_cleaned_updated_at
            )

        presigned_url = await generate_presigned_url(
            settings.AWS_S3_BUCKET_NAME, record.special_cleaned_s3_key
        )

        # ✅ Load and decompress if file is .gz
        file_bytes = await download_file_from_s3(record.special_cleaned_s3_key)
        if record.special_cleaned_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        before = df["normalized_review"].dropna().tolist()[start:end]
        after = df["special_cleaned"].dropna().tolist()[start:end]

        flags = json.loads(record.special_cleaned_flags or "{}")

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
                "flags": {
                    "remove_special": flags.get("remove_special", True),
                    "remove_numbers": flags.get("remove_numbers", True),
                    "remove_emoji": flags.get("remove_emoji", True),
                },
                "should_recompute": should_recompute,
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
    req: TokenizeRequest,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()

        if not record or not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="CLEANED_FILE_NOT_FOUND", message=CLEANED_FILE_NOT_FOUND
            )

        # Reuse tokenized if not outdated
        if record.tokenized_updated_at and record.special_cleaned_updated_at:
            if record.tokenized_updated_at >= record.special_cleaned_updated_at:
                file_bytes = await download_file_from_s3(record.tokenized_s3_key)
                with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                    df = pd.read_csv(gz)
                start, end = (page - 1) * page_size, page * page_size
                return success_response(
                    message=TOKENIZATION_ALREADY_EXISTS,
                    data={
                        "file_id": record.id,
                        "tokenized_s3_url": await generate_presigned_url(
                            settings.AWS_S3_BUCKET_NAME, record.tokenized_s3_key
                        ),
                        "total_records": len(df),
                        "columns": df.columns.tolist(),
                        "before": df["special_cleaned"].fillna("").tolist()[start:end],
                        "after": df["tokens"]
                        .fillna("")
                        .apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                        )
                        .tolist()[start:end],
                        "should_recompute": False,
                    },
                )

        # Delete old tokenized file
        if record.tokenized_s3_key:
            try:
                await delete_file_from_s3(record.tokenized_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old tokenized file: {e}")

        # Read cleaned data
        file_bytes = await download_file_from_s3(record.special_cleaned_s3_key)
        if record.special_cleaned_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        df["tokens"] = df["special_cleaned"].dropna().apply(tokenize_text)
        df_filtered = df[["special_cleaned", "tokens"]]

        # Save gzip
        output_buffer = BytesIO()
        with gzip.GzipFile(fileobj=output_buffer, mode="wb") as gz:
            df_filtered.to_csv(gz, index=False)
        output_buffer.seek(0)

        new_s3_key = f"tokenization/tokens_{uuid4()}.csv.gz"
        s3_url = await upload_file_to_s3(
            output_buffer, new_s3_key, content_type="application/gzip"
        )

        # Save DB
        record.tokenized_s3_key = new_s3_key
        record.tokenized_s3_url = s3_url
        record.tokenized_updated_at = datetime.now()
        await db.commit()

        # Paginated return
        start, end = (page - 1) * page_size, page * page_size
        before = df_filtered["special_cleaned"].fillna("").tolist()[start:end]
        after = df_filtered["tokens"].fillna("").tolist()[start:end]

        return success_response(
            message=TOKENIZATION_SUCCESS,
            data={
                "file_id": record.id,
                "tokenized_s3_url": await generate_presigned_url(
                    settings.AWS_S3_BUCKET_NAME, new_s3_key
                ),
                "total_records": len(df_filtered),
                "columns": df_filtered.columns.tolist(),
                "before": before,
                "after": after,
                "should_recompute": False,
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Tokenization failed: {e}")
        raise ServerError(code="TOKENIZATION_FAILED", message="Tokenization failed.")


@router.get("/tokenize")
async def get_tokenized_preview(
    file_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1),
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record or not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="CLEANED_FILE_NOT_FOUND", message=CLEANED_FILE_NOT_FOUND
            )

        if not record.tokenized_s3_key:
            raise NotFoundError(
                code="TOKENIZED_FILE_NOT_FOUND",
                message="Tokenized data has not been generated yet.",
            )

        should_recompute = False
        if record.tokenized_updated_at and record.special_cleaned_updated_at:
            should_recompute = (
                record.special_cleaned_updated_at > record.tokenized_updated_at
            )

        file_bytes = await download_file_from_s3(record.tokenized_s3_key)
        if record.tokenized_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        start = (page - 1) * page_size
        end = start + page_size
        paginated = df.iloc[start:end]

        before = paginated["special_cleaned"].fillna("").tolist()
        after = (
            paginated["tokens"]
            .fillna("")
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
            .tolist()
        )

        return success_response(
            message="Tokenized data preview retrieved successfully.",
            data={
                "file_id": file_id,
                "tokenized_s3_url": await generate_presigned_url(
                    settings.AWS_S3_BUCKET_NAME, record.tokenized_s3_key
                ),
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "should_recompute": should_recompute,
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Failed to load tokenized data: {e}")
        raise ServerError(
            code="GET_TOKENIZED_FAILED", message="Failed to load tokenized data."
        )


@router.post("/remove-stopwords")
async def remove_stopwords_by_file_id(
    req: StopwordTokenRequest,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()
        if not record or not record.tokenized_s3_key:
            raise NotFoundError(
                code="TOKENIZED_FILE_NOT_FOUND", message=TOKENIZED_FILE_NOT_FOUND
            )

        config = {
            "custom_stopwords": req.custom_stopwords or [],
            "exclude_stopwords": req.exclude_stopwords or [],
        }
        config_json = json.dumps(config, sort_keys=True)

        # Use cached result if config hasn't changed
        if (
            record.stopword_s3_key
            and record.stopword_updated_at
            and record.tokenized_updated_at
            and record.stopword_updated_at >= record.tokenized_updated_at
            and record.stopword_config == config_json
        ):
            file_bytes = await download_file_from_s3(record.stopword_s3_key)
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
            start, end = (page - 1) * page_size, page * page_size
            return success_response(
                message=STOPWORD_ALREADY_EXISTS,
                data={
                    "file_id": record.id,
                    "stopword_s3_url": await generate_presigned_url(
                        settings.AWS_S3_BUCKET_NAME, record.stopword_s3_key
                    ),
                    "total_records": len(df),
                    "columns": df.columns.tolist(),
                    "before": df["tokens"].fillna("").apply(eval).tolist()[start:end],
                    "after": df["stopword_removed"]
                    .fillna("")
                    .apply(eval)
                    .tolist()[start:end],
                    "should_recompute": False,
                    "config": config,
                },
            )

        # Delete old if exists
        if record.stopword_s3_key:
            try:
                await delete_file_from_s3(record.stopword_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old stopword file: {e}")

        # Read tokenized file (compressed or not)
        file_bytes = await download_file_from_s3(record.tokenized_s3_key)
        if record.tokenized_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
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

        # Save compressed
        output_buffer = BytesIO()
        with gzip.GzipFile(fileobj=output_buffer, mode="wb") as gz:
            df[["tokens", "stopword_removed"]].to_csv(gz, index=False)
        output_buffer.seek(0)

        new_s3_key = f"stopwords/stopword_removed_{uuid4()}.csv.gz"
        s3_url = await upload_file_to_s3(
            output_buffer, new_s3_key, content_type="application/gzip"
        )

        # Update DB
        record.stopword_s3_key = new_s3_key
        record.stopword_s3_url = s3_url
        record.stopword_config = config_json
        record.stopword_updated_at = datetime.now()

        await db.commit()

        start, end = (page - 1) * page_size, page * page_size
        before = df["tokens"].tolist()[start:end]
        after = df["stopword_removed"].tolist()[start:end]

        return success_response(
            message=STOPWORD_SUCCESS,
            data={
                "file_id": record.id,
                "stopword_s3_url": await generate_presigned_url(
                    settings.AWS_S3_BUCKET_NAME, new_s3_key
                ),
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "should_recompute": False,
                "config": config,
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Stopword removal failed: {e}")
        raise ServerError(
            code="STOPWORD_REMOVAL_FAILED", message="Stopword removal failed."
        )


@router.get("/remove-stopwords")
async def get_stopword_removal_preview(
    file_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1),
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record or not record.stopword_s3_key:
            raise NotFoundError(
                code="STOPWORD_REMOVED_NOT_FOUND",
                message="Stopword-removed file not found.",
            )

        should_recompute = False
        if record.tokenized_updated_at and record.stopword_updated_at:
            should_recompute = record.tokenized_updated_at > record.stopword_updated_at

        file_bytes = await download_file_from_s3(record.stopword_s3_key)
        if record.stopword_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        start, end = (page - 1) * page_size, page * page_size

        return success_response(
            message="Stopword-removed data preview retrieved successfully.",
            data={
                "file_id": file_id,
                "stopword_s3_url": await generate_presigned_url(
                    settings.AWS_S3_BUCKET_NAME, record.stopword_s3_key
                ),
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": df["tokens"].fillna("").apply(eval).tolist()[start:end],
                "after": df["stopword_removed"]
                .fillna("")
                .apply(eval)
                .tolist()[start:end],
                "should_recompute": should_recompute,
                "config": json.loads(record.stopword_config or "{}"),
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Failed to load stopword-removed data: {e}")
        raise ServerError(
            code="GET_STOPWORD_REMOVAL_FAILED",
            message="Failed to load stopword-removed data.",
        )


@router.post("/lemmatize")
async def lemmatize_by_file_id(
    req: LemmatizeRequest,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()

        if not record or not record.stopword_s3_key:
            raise NotFoundError(
                code="STOPWORD_FILE_NOT_FOUND", message=STOPWORD_FILE_NOT_FOUND
            )

        file_bytes = await download_file_from_s3(record.stopword_s3_key)
        if record.stopword_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        df["stopword_removed"] = df["stopword_removed"].apply(eval)

        # Determine if recomputation is needed
        should_recompute = True
        if (
            record.lemmatized_s3_key
            and record.stopword_updated_at
            and record.lemmatized_updated_at
            and record.lemmatized_updated_at >= record.stopword_updated_at
        ):
            should_recompute = False

        if not should_recompute:
            lemmatized_bytes = await download_file_from_s3(record.lemmatized_s3_key)
            if record.lemmatized_s3_key.endswith(".gz"):
                with gzip.GzipFile(fileobj=BytesIO(lemmatized_bytes)) as gz:
                    df = pd.read_csv(gz)
            else:
                df = pd.read_csv(BytesIO(lemmatized_bytes))

            df["stopword_removed"] = df["stopword_removed"].apply(eval)
            df["lemmatized_tokens"] = df["lemmatized_tokens"].apply(eval)

            start, end = (page - 1) * page_size, page * page_size
            return success_response(
                message=LEMMATIZATION_ALREADY_EXISTS,
                data={
                    "file_id": record.id,
                    "lemmatized_s3_url": await generate_presigned_url(
                        settings.AWS_S3_BUCKET_NAME, record.lemmatized_s3_key
                    ),
                    "total_records": len(df),
                    "columns": df.columns.tolist(),
                    "before": df["stopword_removed"].tolist()[start:end],
                    "after": df["lemmatized_tokens"].tolist()[start:end],
                    "should_recompute": False,
                },
            )

        # Delete old
        if record.lemmatized_s3_key:
            try:
                await delete_file_from_s3(record.lemmatized_s3_key)
            except Exception as e:
                logger.warning(f"Failed to delete old lemmatized file: {e}")

        # Apply lemmatization
        df["lemmatized_tokens"] = df["stopword_removed"].apply(
            lambda tokens: lemmatize_tokens(tokens)["lemmatized_tokens"]
        )

        output_buffer = BytesIO()
        with gzip.GzipFile(fileobj=output_buffer, mode="wb") as gz:
            df[["stopword_removed", "lemmatized_tokens"]].to_csv(gz, index=False)
        output_buffer.seek(0)

        new_s3_key = f"lemmatization/lemmatized_{uuid4()}.csv.gz"
        s3_url = await upload_file_to_s3(
            output_buffer, new_s3_key, content_type="application/gzip"
        )

        # Update DB
        record.lemmatized_s3_key = new_s3_key
        record.lemmatized_s3_url = s3_url
        record.lemmatized_updated_at = datetime.now()
        await db.commit()

        # Response
        start, end = (page - 1) * page_size, page * page_size
        return success_response(
            message=LEMMATIZATION_SUCCESS,
            data={
                "file_id": record.id,
                "lemmatized_s3_url": await generate_presigned_url(
                    settings.AWS_S3_BUCKET_NAME, new_s3_key
                ),
                "total_records": len(df),
                "columns": ["stopword_removed", "lemmatized_tokens"],
                "before": df["stopword_removed"].tolist()[start:end],
                "after": df["lemmatized_tokens"].tolist()[start:end],
                "should_recompute": False,
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Lemmatization failed: {e}")
        raise ServerError(code="LEMMATIZATION_FAILED", message="Lemmatization failed.")


@router.get("/lemmatize")
async def get_lemmatization_preview(
    file_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1),
    db: AsyncSession = Depends(get_db),
):
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record or not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND",
                message="Lemmatized file not found.",
            )

        file_bytes = await download_file_from_s3(record.lemmatized_s3_key)
        if record.lemmatized_s3_key.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(file_bytes)) as gz:
                df = pd.read_csv(gz)
        else:
            df = pd.read_csv(BytesIO(file_bytes))

        df["stopword_removed"] = df["stopword_removed"].apply(eval)
        df["lemmatized_tokens"] = df["lemmatized_tokens"].apply(eval)

        should_recompute = False
        if record.stopword_updated_at and record.lemmatized_updated_at:
            should_recompute = record.stopword_updated_at > record.lemmatized_updated_at

        start, end = (page - 1) * page_size, page * page_size

        return success_response(
            message="Lemmatized data preview retrieved successfully.",
            data={
                "file_id": file_id,
                "lemmatized_s3_url": await generate_presigned_url(
                    settings.AWS_S3_BUCKET_NAME, record.lemmatized_s3_key
                ),
                "total_records": len(df),
                "columns": ["stopword_removed", "lemmatized_tokens"],
                "before": df["stopword_removed"].tolist()[start:end],
                "after": df["lemmatized_tokens"].tolist()[start:end],
                "should_recompute": should_recompute,
            },
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"Failed to load lemmatized data: {e}")
        raise ServerError(
            code="GET_LEMMATIZED_FAILED",
            message="Failed to load lemmatized data.",
        )
