import ast
from typing import Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
import pandas as pd

from app.core.database import get_db
from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression
from app.core.file_handler.storage import S3Storage
from app.core.lemmatization.config import LemmatizationConfig
from app.core.lemmatization.lemmatizer import DefaultLemmatizer
from app.core.normalization.normalizer import DefaultTextNormalizer
from app.core.special_char_removal.cleaner import DefaultSpecialCleaner
from app.core.special_char_removal.config import SpecialCleanConfig
from app.core.stopword_removal.config import StopwordConfig
from app.core.stopword_removal.removal import DefaultStopwordRemover
from app.core.tokenization.config import TokenizationConfig
from app.core.tokenization.tokenizer import DefaultTokenizer
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

from app.models.db.file_record import FileRecord
from app.schemas.clean import (
    LemmatizeRequest,
    NormalizeRequest,
    SpecialCleanRequest,
    StopwordTokenRequest,
    TokenizeRequest,
)
from app.services.file_service import FileService
from app.services.lemmatization_service import LemmatizationService
from app.services.normalization_service import NormalizationService
from app.services.special_clean_service import SpecialCleanService
from app.services.stopword_service import StopwordService
from app.services.tokenization_service import TokenizationService
from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError

router = APIRouter(prefix="/api/clean", tags=["Cleaning"])
logger = logging.getLogger(__name__)


def _files() -> FileService:
    # Single place to build the FileService used by all cleaning routes
    return FileService(
        storage=S3Storage(),
        codec=CsvCodec(index=False),
        compression=GzipCompression(),
    )


def _to_list(v: Any) -> list[str]:
    """Robustly parse tokens whether stored as list or as a stringified list."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            return parsed if isinstance(parsed, list) else [v]
        except Exception:
            return [v]
    return []


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
    req: "NormalizeRequest", db: AsyncSession = Depends(get_db)
):
    """
    Normalize text for a given file. Reuses cached normalized CSV when the same broken_map was used.
    """
    try:
        # 1) Load FileRecord
        res = await db.execute(select(FileRecord).filter_by(id=req.file_id))
        record = res.scalars().first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        # 2) Use the same NormalizationService used in the pipeline
        files = _files()
        normalizer = DefaultTextNormalizer()
        norm_service = NormalizationService(files, normalizer)

        # Pass the user-supplied broken_map (or None). Service decides to reuse or recompute.
        result = await norm_service.ensure_normalized(
            db=db, record=record, override_broken_map=req.broken_map or None
        )
        df: pd.DataFrame = result.df

        # 3) Build presigned URL (use FileService/Storage, not raw helpers)
        presigned_url = await files.storage.presigned_url(
            record.normalized_s3_key, 6000
        )

        # 4) Preview rows
        before = df["review"].dropna().tolist()[:20] if "review" in df.columns else []
        after = (
            df["normalized_review"].dropna().tolist()[:20]
            if "normalized_review" in df.columns
            else []
        )

        # NB: result.recomputed tells you whether we re-used S3 or recomputed
        return success_response(
            message=NORMALIZATION_SUCCESS
            if not result.recomputed
            else NORMALIZATION_ALREADY_EXISTS,
            data={
                "file_id": record.id,
                "normalized_s3_url": presigned_url,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "broken_words": json.loads(record.normalized_broken_map or "{}"),
                "before": before,
                "after": after,
                "should_recompute": False,
                "reused": result.recomputed,
            },
        )

    except NotFoundError:
        raise
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
    """
    Paginated preview of the normalized CSV already stored for a file.
    """
    try:
        res = await db.execute(select(FileRecord).filter_by(id=file_id))
        record = res.scalars().first()

        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        if not record.normalized_s3_key:
            raise NotFoundError(
                code="NORMALIZATION_NOT_FOUND",
                message="Normalization not found for this file.",
            )

        # 1) Load the DF via FileService (handles gzip/csv transparently)
        files = _files()
        df = await files.download_df(record.normalized_s3_key)

        # 2) Presigned URL for the stored normalized CSV
        presigned_url = await files.storage.presigned_url(
            record.normalized_s3_key, expires_in=6000
        )

        start = max(0, (page - 1) * limit)
        end = start + limit

        before = (
            df["review"].dropna().tolist()[start:end] if "review" in df.columns else []
        )
        after = (
            df["normalized_review"].dropna().tolist()[start:end]
            if "normalized_review" in df.columns
            else []
        )

        return success_response(
            message=NORMALIZATION_ALREADY_EXISTS,
            data={
                "file_id": file_id,
                "normalized_s3_url": presigned_url,
                "columns": ["review", "normalized_review"],
                "broken_words": json.loads(record.normalized_broken_map or "{}"),
                "before": before,
                "after": after,
                "total_records": len(df),
                "page": page,
                "limit": limit,
                "should_recompute": False,
            },
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Normalization preview failed: {e}")
        raise ServerError(code="NORMALIZATION_FAILED", message="Normalization failed.")


@router.post("/remove-special")
async def remove_special_characters_by_file(
    req: SpecialCleanRequest, db: AsyncSession = Depends(get_db)
):
    """
    Remove special characters/numbers/emoji from the *normalized* text.
    Reuses cached result if flags match and cache is fresh.
    """
    try:
        # 1) Load record
        result = await db.execute(select(FileRecord).filter_by(id=req.file_id))
        record = result.scalars().first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        # Must have a normalized artifact for the individual endpoint flow
        if not record.normalized_s3_key:
            raise NotFoundError(
                code="NORMALIZED_FILE_NOT_FOUND", message=NORMALIZED_FILE_NOT_FOUND
            )

        # 2) Build service stack
        files = _files()
        cleaner = DefaultSpecialCleaner(
            SpecialCleanConfig(
                remove_special=req.remove_special,
                remove_numbers=req.remove_numbers,
                remove_emoji=req.remove_emoji,
            )
        )
        special_service = SpecialCleanService(files, cleaner)

        # 3) Let the service decide reuse vs recompute
        flags_dict = {
            "remove_special": req.remove_special,
            "remove_numbers": req.remove_numbers,
            "remove_emoji": req.remove_emoji,
        }
        sc_result = await special_service.ensure_special_cleaned(
            db=db, record=record, df_norm=None, override_flags=flags_dict
        )
        df = sc_result.df

        # 4) URL & preview
        presigned_url = await files.storage.presigned_url(
            sc_result.key, expires_in=6000
        )

        before = (
            df["normalized_review"].dropna().tolist()[:20]
            if "normalized_review" in df.columns
            else []
        )
        after = (
            df["special_cleaned"].dropna().tolist()[:20]
            if "special_cleaned" in df.columns
            else []
        )

        # 5) Build response
        # recomputed=True  -> we created new file  -> SPECIAL_CLEAN_SUCCESS
        # recomputed=False -> we reused existing   -> SPECIAL_CLEAN_ALREADY_EXISTS
        msg = (
            SPECIAL_CLEAN_SUCCESS
            if sc_result.recomputed
            else SPECIAL_CLEAN_ALREADY_EXISTS
        )

        removed_chars = []
        try:
            removed_chars = list(
                set(json.loads(record.special_cleaned_removed or "[]"))
            )
        except Exception:
            pass

        return success_response(
            message=msg,
            data={
                "file_id": record.id,
                "cleaned_s3_url": presigned_url,
                "total_records": len(df),
                "removed_characters": removed_chars,
                "before": before,
                "after": after,
                "flags": flags_dict,
                "should_recompute": False,
                "reused": (not sc_result.recomputed),
            },
        )

    except NotFoundError:
        raise
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
    """
    Paginated preview of the special-cleaned CSV already stored for a file.
    """
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

        if not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="SPECIAL_CLEAN_NOT_FOUND",
                message="Special cleaned file not found for this file ID.",
            )

        files = _files()
        df = await files.download_df(record.special_cleaned_s3_key)

        presigned_url = await files.storage.presigned_url(
            record.special_cleaned_s3_key, expires_in=6000
        )

        # freshness hint
        should_recompute = False
        if record.normalized_updated_at and record.special_cleaned_updated_at:
            should_recompute = (
                record.normalized_updated_at > record.special_cleaned_updated_at
            )

        # pagination
        start = max(0, (page - 1) * page_size)
        end = start + page_size

        before = (
            df["normalized_review"].dropna().tolist()[start:end]
            if "normalized_review" in df.columns
            else []
        )
        after = (
            df["special_cleaned"].dropna().tolist()[start:end]
            if "special_cleaned" in df.columns
            else []
        )

        flags = {}
        try:
            # flat flags JSON: {"remove_special": true, ...}
            flags = json.loads(record.special_cleaned_flags or "{}")
        except Exception:
            pass

        removed_chars = []
        try:
            removed_chars = list(
                set(json.loads(record.special_cleaned_removed or "[]"))
            )
        except Exception:
            pass

        return success_response(
            message=SPECIAL_CLEAN_ALREADY_EXISTS,
            data={
                "file_id": record.id,
                "cleaned_s3_url": presigned_url,
                "removed_characters": removed_chars,
                "before": before,
                "after": after,
                "total_records": len(df),
                "flags": {
                    "remove_special": bool(flags.get("remove_special", True)),
                    "remove_numbers": bool(flags.get("remove_numbers", True)),
                    "remove_emoji": bool(flags.get("remove_emoji", True)),
                },
                "page": page,
                "page_size": page_size,
                "should_recompute": should_recompute,
            },
        )

    except NotFoundError:
        raise
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
    """
    Tokenize cleaned text for a given file.

    Reuses the cached tokenized CSV if the same tokenization config was used
    and the source (special_cleaned) is not newer.
    """
    try:
        # 1) Load FileRecord and validate pre-requisite step
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()

        if not record or not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="CLEANED_FILE_NOT_FOUND", message=CLEANED_FILE_NOT_FOUND
            )

        # 2) Build service + config (allow optional method in request; default wordpunct)
        files = _files()
        cfg = TokenizationConfig(method=getattr(req, "method", "wordpunct"))
        tokenizer = DefaultTokenizer(cfg)
        tok_svc = TokenizationService(files, tokenizer)

        # 3) Service decides reuse vs recompute (handles S3 I/O + DB metadata)
        tok_res = await tok_svc.ensure_tokenized(
            db=db,
            record=record,
            df_clean=None,  # let the service load the special_cleaned DF
            override_config=None,  # or pass a dict if you want to override
        )

        df = tok_res.df
        presigned_url = await files.storage.presigned_url(tok_res.key, expires_in=6000)

        # 4) Paginated preview
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        before = (
            df["special_cleaned"].fillna("").tolist()[start:end]
            if "special_cleaned" in df.columns
            else []
        )
        after = (
            df["tokens"].iloc[start:end].apply(_to_list).tolist()
            if "tokens" in df.columns
            else []
        )

        # 5) Response
        message = (
            TOKENIZATION_SUCCESS
            if getattr(tok_res, "recomputed", True)
            else TOKENIZATION_ALREADY_EXISTS
        )
        return success_response(
            message=message,
            data={
                "file_id": record.id,
                "tokenized_s3_url": presigned_url,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "page": page,
                "page_size": page_size,
                "should_recompute": False,
                "reused": not getattr(tok_res, "recomputed", True),
            },
        )

    except NotFoundError:
        raise
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
    """
    Paginated preview of the tokenized CSV already stored for a file.
    """
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
                code="TOKENIZED_FILE_NOT_FOUND", message=TOKENIZED_FILE_NOT_FOUND
            )

        # freshness hint (if you want to show UI whether to recompute)
        should_recompute = False
        if record.tokenized_updated_at and record.special_cleaned_updated_at:
            should_recompute = (
                record.special_cleaned_updated_at > record.tokenized_updated_at
            )

        # Load DF via FileService (gz handled automatically)
        files = _files()
        df = await files.download_df(record.tokenized_s3_key)
        presigned_url = await files.storage.presigned_url(
            record.tokenized_s3_key, expires_in=6000
        )

        # Paginate
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        slice_df = df.iloc[start:end]

        before = (
            slice_df["special_cleaned"].fillna("").tolist()
            if "special_cleaned" in slice_df.columns
            else []
        )
        after = (
            slice_df["tokens"].apply(_to_list).tolist()
            if "tokens" in slice_df.columns
            else []
        )

        return success_response(
            message=TOKENIZATION_ALREADY_EXISTS,
            data={
                "file_id": file_id,
                "tokenized_s3_url": presigned_url,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "page": page,
                "page_size": page_size,
                "should_recompute": should_recompute,
            },
        )

    except NotFoundError:
        raise
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
    """
    Remove stopwords from tokenized text.
    Reuses cached stopword CSV when the same config was used and tokenized data is not newer.
    """
    try:
        # 1) Load file record and ensure tokenized step exists
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()

        if not record or not record.tokenized_s3_key:
            raise NotFoundError(
                code="TOKENIZED_FILE_NOT_FOUND", message=TOKENIZED_FILE_NOT_FOUND
            )

        # 2) Build service with config (respect user overrides)
        files = _files()
        cfg = StopwordConfig(
            language=getattr(req, "language", "english"),
            custom_stopwords=set(req.custom_stopwords or []),
            exclude_stopwords=set(req.exclude_stopwords or []),
            lowercase=getattr(req, "lowercase", True),
            preserve_negations=getattr(req, "preserve_negations", True),
        )
        remover = DefaultStopwordRemover(cfg)
        svc = StopwordService(files, remover)

        # 3) Service decides reuse vs recompute (and persists S3 + DB metadata)
        res = await svc.ensure_stopwords_removed(
            db=db,
            record=record,
            df_tokens=None,  # let service load tokenized CSV
            override_config=None,  # or pass a dict to force override
        )

        df = res.df
        presigned = await files.storage.presigned_url(res.key, expires_in=6000)

        # 4) Paginated preview
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        before = (
            df["tokens"].iloc[start:end].apply(_to_list).tolist()
            if "tokens" in df.columns
            else []
        )
        after = (
            df["stopword_removed"].iloc[start:end].apply(_to_list).tolist()
            if "stopword_removed" in df.columns
            else []
        )

        return success_response(
            message=STOPWORD_SUCCESS if res.recomputed else STOPWORD_ALREADY_EXISTS,
            data={
                "file_id": record.id,
                "stopword_s3_url": presigned,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "page": page,
                "page_size": page_size,
                "should_recompute": False,
                "config": {
                    "language": cfg.language,
                    "custom_stopwords": sorted(list(cfg.custom_stopwords)),
                    "exclude_stopwords": sorted(list(cfg.exclude_stopwords)),
                    "lowercase": cfg.lowercase,
                    "preserve_negations": cfg.preserve_negations,
                },
                "reused": not res.recomputed,
            },
        )

    except NotFoundError:
        raise
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
    """
    Paginated preview of the stopword-removed CSV already stored for a file.
    """
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

        files = _files()
        df = await files.download_df(record.stopword_s3_key)
        presigned = await files.storage.presigned_url(
            record.stopword_s3_key, expires_in=6000
        )

        # paginate
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        view = df.iloc[start:end]

        before = (
            view["tokens"].apply(_to_list).tolist() if "tokens" in view.columns else []
        )
        after = (
            view["stopword_removed"].apply(_to_list).tolist()
            if "stopword_removed" in view.columns
            else []
        )

        # Config echo (if you persist it as JSON on the record)
        cfg_obj = {}
        try:
            cfg_obj = json.loads(record.stopword_config or "{}")
        except Exception:
            cfg_obj = {}

        return success_response(
            message=STOPWORD_ALREADY_EXISTS,
            data={
                "file_id": file_id,
                "stopword_s3_url": presigned,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "page": page,
                "page_size": page_size,
                "should_recompute": should_recompute,
                "config": cfg_obj,
            },
        )

    except NotFoundError:
        raise
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
    """
    Lemmatize tokens produced by the stopword-removal step.
    Reuses cached lemmatized CSV when the same config was used and stopword data is not newer.
    """
    try:
        # 1) Load record and ensure stopword step exists
        record = (
            await db.execute(select(FileRecord).filter_by(id=req.file_id))
        ).scalar_one_or_none()

        if not record or not record.stopword_s3_key:
            raise NotFoundError(
                code="STOPWORD_FILE_NOT_FOUND", message=STOPWORD_FILE_NOT_FOUND
            )

        # 2) Build service + config
        files = _files()
        cfg = LemmatizationConfig(
            use_pos_tagging=getattr(req, "use_pos_tagging", True),
            lowercase=getattr(req, "lowercase", False),
        )
        lemmatizer = DefaultLemmatizer(cfg)
        svc = LemmatizationService(files, lemmatizer)

        # 3) Service decides reuse vs recompute (and persists S3 + DB metadata)
        res = await svc.ensure_lemmatized(
            db=db,
            record=record,
            df_stop=None,  # let service load stopword CSV
            override_cfg=None,  # or pass a dict to override
        )

        df = res.df
        presigned = await files.storage.presigned_url(res.key, expires_in=6000)

        # 4) Paginated preview
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        before = (
            df["stopword_removed"].iloc[start:end].apply(_to_list).tolist()
            if "stopword_removed" in df.columns
            else []
        )
        after = (
            df["lemmatized_tokens"].iloc[start:end].apply(_to_list).tolist()
            if "lemmatized_tokens" in df.columns
            else []
        )

        return success_response(
            message=LEMMATIZATION_SUCCESS
            if res.recomputed
            else LEMMATIZATION_ALREADY_EXISTS,
            data={
                "file_id": record.id,
                "lemmatized_s3_url": presigned,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "page": page,
                "page_size": page_size,
                "should_recompute": False,
                "config": {
                    "use_pos_tagging": cfg.use_pos_tagging,
                    "lowercase": cfg.lowercase,
                },
                "reused": not res.recomputed,
            },
        )

    except NotFoundError:
        raise
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
    """
    Paginated preview of the lemmatized CSV already stored for a file.
    """
    try:
        record = (
            await db.execute(select(FileRecord).filter_by(id=file_id))
        ).scalar_one_or_none()

        if not record or not record.lemmatized_s3_key:
            raise NotFoundError(
                code="LEMMATIZED_FILE_NOT_FOUND", message="Lemmatized file not found."
            )

        files = _files()
        df = await files.download_df(record.lemmatized_s3_key)
        presigned = await files.storage.presigned_url(
            record.lemmatized_s3_key, expires_in=6000
        )

        should_recompute = False
        if record.stopword_updated_at and record.lemmatized_updated_at:
            should_recompute = record.stopword_updated_at > record.lemmatized_updated_at

        # paginate
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        view = df.iloc[start:end]

        before = (
            view["stopword_removed"].apply(_to_list).tolist()
            if "stopword_removed" in view.columns
            else []
        )
        after = (
            view["lemmatized_tokens"].apply(_to_list).tolist()
            if "lemmatized_tokens" in view.columns
            else []
        )

        # optional: echo config if you persisted it on FileRecord
        try:
            cfg_echo = json.loads(getattr(record, "lemmatized_config", "{}") or "{}")
        except Exception:
            cfg_echo = {}

        return success_response(
            message=LEMMATIZATION_ALREADY_EXISTS,
            data={
                "file_id": file_id,
                "lemmatized_s3_url": presigned,
                "total_records": len(df),
                "columns": df.columns.tolist(),
                "before": before,
                "after": after,
                "page": page,
                "page_size": page_size,
                "should_recompute": should_recompute,
                "config": cfg_echo,
            },
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Failed to load lemmatized data: {e}")
        raise ServerError(
            code="GET_LEMMATIZED_FAILED", message="Failed to load lemmatized data."
        )
