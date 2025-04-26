from datetime import datetime
from io import BytesIO
import json
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
from app.services.s3_uploader import (
    delete_file_from_s3,
    download_file_from_s3,
    upload_file_to_s3,
)

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
        broken_map_json = json.dumps(broken_map, sort_keys=True)

        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="File ID not found")

        # ✅ Skip recompute if normalized already and broken_map hasn't changed
        if record.normalized_s3_key and record.normalized_broken_map == broken_map_json:
            logger.info("⏩ Skipping normalization — no changes in broken_map.")
            return NormalizeResponse(
                status="success",
                message="Normalization result already exists.",
                data={
                    "file_id": file_id,
                    "normalized_s3_url": record.normalized_s3_url,
                    "record_count": None,  # You can also cache this if needed
                    "columns": ["review", "normalized_review"],
                },
            )

        # 🗑️ Remove old S3 file if exists
        if record.normalized_s3_key:
            try:
                delete_file_from_s3(record.normalized_s3_key)
                logger.info(
                    f"🗑️ Deleted old normalized file: {record.normalized_s3_key}"
                )
            except Exception as err:
                logger.warning(f"⚠️ Failed to delete old normalized file: {err}")

        # 📥 Download original file
        file_bytes = download_file_from_s3(record.s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

        # 🔄 Normalize
        normalized_df = df[["review"]].dropna().copy()
        normalized_df["normalized_review"] = normalized_df["review"].apply(
            lambda r: normalize_text(r, broken_map)
        )

        # 💾 Upload new file
        output_buffer = BytesIO()
        normalized_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"normalization/normalized_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key, content_type="text/csv")

        # 📝 Update DB
        record.normalized_s3_key = new_s3_key
        record.normalized_s3_url = s3_url
        record.normalized_broken_map = broken_map_json
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

        current_flags = {
            "remove_special": req.remove_special,
            "remove_numbers": req.remove_numbers,
            "remove_emoji": req.remove_emoji,
        }

        # ✅ Skip recompute if flags match and cleaned output exists
        if record.special_cleaned_s3_key and record.special_cleaned_flags:
            prev_flags = json.loads(record.special_cleaned_flags)
            if prev_flags == current_flags:
                return SpecialCharCleanedResponse(
                    status="success",
                    message="Special cleaned file already exists and matches options.",
                    data=SpecialCharCleanedData(
                        file_id=record.id,
                        cleaned_s3_url=record.special_cleaned_s3_url,
                        record_count=None,
                        columns=[],
                        removed_characters=[],
                    ),
                )

        # 🗑️ Delete previous file if it exists
        if record.special_cleaned_s3_key:
            try:
                delete_file_from_s3(record.special_cleaned_s3_key)
            except Exception as del_err:
                logger.warning(
                    f"⚠️ Failed to delete old special cleaned file: {del_err}"
                )

        file_bytes = download_file_from_s3(record.normalized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))

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

        output_buffer = BytesIO()
        cleaned_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"cleaned/special_cleaned_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key)

        # ✅ Update DB
        record.special_cleaned_s3_key = new_s3_key
        record.special_cleaned_s3_url = s3_url
        record.special_cleaned_flags = json.dumps(current_flags)
        record.special_cleaned_updated_at = datetime.now()
        db.commit()

        return SpecialCharCleanedResponse(
            status="success",
            message="Special characters removed and saved to S3.",
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
        file_id = req.file_id
        record = db.query(FileRecord).filter_by(id=file_id).first()
        if not record or not record.special_cleaned_s3_key:
            raise HTTPException(status_code=404, detail="Cleaned file not found")

        # ✅ Skip if already tokenized after special cleaned
        if record.tokenized_updated_at and record.special_cleaned_updated_at:
            if record.tokenized_updated_at >= record.special_cleaned_updated_at:
                logger.info("⏩ Skipping tokenization — already up-to-date.")
                return TokenizedResponse(
                    status="success",
                    message="Tokenized file already up-to-date.",
                    data={
                        "file_id": file_id,
                        "tokenized_s3_url": record.tokenized_s3_url,
                        "record_count": None,
                        "columns": ["special_cleaned", "tokens"],
                    },
                )

        # 🗑️ Delete existing tokenized file
        if record.tokenized_s3_key:
            try:
                delete_file_from_s3(record.tokenized_s3_key)
            except Exception as del_err:
                logger.warning(f"⚠️ Failed to delete old tokenized file: {del_err}")

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

        current_config = json.dumps(
            {
                "custom_stopwords": req.custom_stopwords or [],
                "exclude_stopwords": req.exclude_stopwords or [],
            }
        )

        print(f"tokenized updated at::: {record.tokenized_updated_at}")
        print(f"stopword updated at::: {record.stopword_updated_at}")
        print(
            f"comparation::: {record.stopword_updated_at >= record.tokenized_updated_at}"
        )
        print(f"config change::: {record.stopword_config == current_config}")

        # 🛑 Skip if no update & config unchanged
        if (
            record.stopword_s3_key
            and record.tokenized_updated_at
            and record.stopword_updated_at
            and record.stopword_updated_at >= record.tokenized_updated_at
            and record.stopword_config == current_config
        ):
            return StopwordTokenResponse(
                status="success",
                message="Already up-to-date. No stopword removal needed.",
                data=StopwordTokenResponseData(
                    file_id=file_id,
                    tokenized_s3_url=record.tokenized_s3_url,
                    stopword_s3_url=record.stopword_s3_url,
                    record_count=None,
                    columns=["tokens", "stopword_removed"],
                ),
            )

        # 🗑️ Delete old stopword_removed file if exists
        if record.stopword_s3_key:
            try:
                delete_file_from_s3(record.stopword_s3_key)
                logger.info(f"🗑️ Deleted old stopword file: {record.stopword_s3_key}")
            except Exception as del_err:
                logger.warning(f"⚠️ Failed to delete old stopword file: {del_err}")

        # ⬇️ Download and read tokenized data
        file_bytes = download_file_from_s3(record.tokenized_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["tokens"] = df["tokens"].apply(eval)

        # 🧹 Remove stopwords
        stopword_results = df["tokens"].apply(
            lambda toks: remove_stopwords_from_tokens(
                tokens=toks,
                custom_stopwords=req.custom_stopwords or [],
                exclude_stopwords=req.exclude_stopwords or [],
            )
        )
        df["stopword_removed"] = stopword_results.map(lambda r: r["cleaned_tokens"])

        # 💾 Upload new file
        output_buffer = BytesIO()
        df[["tokens", "stopword_removed"]].to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"stopwords/stopword_removed_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key, content_type="text/csv")

        # 🔁 Update DB
        record.stopword_s3_key = new_s3_key
        record.stopword_s3_url = s3_url
        record.stopword_config = current_config
        record.stopword_updated_at = datetime.now()
        db.commit()

        return StopwordTokenResponse(
            status="success",
            message="Stopwords removed and saved to S3.",
            data=StopwordTokenResponseData(
                file_id=file_id,
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

        # ✅ Skip recomputation if no changes
        if (
            record.lemmatized_s3_key
            and record.stopword_updated_at
            and record.lemmatized_updated_at
            and record.lemmatized_updated_at >= record.stopword_updated_at
        ):
            logger.info("⏩ Skipping lemmatization — no change in stopword removal.")
            return LemmatizedResponse(
                status="success",
                message="Already lemmatized. No changes.",
                data=LemmatizedData(
                    file_id=file_id,
                    tokenized_s3_url=record.stopword_s3_url,
                    lemmatized_s3_url=record.lemmatized_s3_url,
                    record_count=None,
                    columns=["stopword_removed", "lemmatized_tokens"],
                ),
            )

        # ✅ Delete previous lemmatized file if it exists
        if record.lemmatized_s3_key:
            try:
                delete_file_from_s3(record.lemmatized_s3_key)
                logger.info(
                    f"🗑️ Deleted old lemmatized file: {record.lemmatized_s3_key}"
                )
            except Exception as del_err:
                logger.warning(
                    f"⚠️ Failed to delete previous lemmatized file: {del_err}"
                )

        # ⬇️ Download from S3 and load CSV
        file_bytes = download_file_from_s3(record.stopword_s3_key)
        df = pd.read_csv(BytesIO(file_bytes))
        df["stopword_removed"] = df["stopword_removed"].apply(eval)

        # 🧬 Apply lemmatization
        df["lemmatized_tokens"] = df["stopword_removed"].apply(
            lambda tokens: lemmatize_tokens(tokens)["lemmatized_tokens"]
        )

        # 💾 Upload new file to S3
        output_buffer = BytesIO()
        df[["stopword_removed", "lemmatized_tokens"]].to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        new_s3_key = f"lemmatization/lemmatized_{uuid4()}.csv"
        s3_url = upload_file_to_s3(output_buffer, new_s3_key, content_type="text/csv")

        # 🔁 Update DB
        record.lemmatized_s3_key = new_s3_key
        record.lemmatized_s3_url = s3_url
        record.lemmatized_updated_at = datetime.now()
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
