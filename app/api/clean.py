from fastapi import APIRouter, HTTPException, Query
from app.schemas.clean import (
    LemmatizeRequest,
    LemmatizedData,
    LemmatizedResponse,
    NormalizeData,
    NormalizeRequest,
    NormalizeResponse,
    SpecialCharCleanedData,
    SpecialCharCleanedResponse,
    StopwordTokenRequest,
    StopwordTokenResponse,
    StopwordTokenResponseData,
    TokenizedData,
    TokenizedResponse,
)
from app.services.preprocess import (
    lemmatize_tokens,
    normalize_text,
    remove_special_characters,
    remove_stopwords_from_tokens,
    tokenize_text,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clean", tags=["Cleaning"])


@router.post("/normalize", response_model=NormalizeResponse)
async def normalize_text_api(req: NormalizeRequest):
    try:
        logger.info("🔄 Normalizing input text...")
        normalized = normalize_text(req.text)

        logger.info("✅ Normalization successful.")
        return NormalizeResponse(
            status="success",
            message="Text normalized successfully",
            data=NormalizeData(original=req.text, normalized=normalized),
        )
    except Exception as e:
        logger.exception(f"❌ Normalization failed::: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to normalize text. Please try again later."
        )


@router.post("/remove-special", response_model=SpecialCharCleanedResponse)
async def remove_special_characters_api(
    req: NormalizeRequest,
    remove_special: bool = Query(True, description="Remove symbols/punctuation"),
    remove_numbers: bool = Query(True, description="Remove numeric digits"),
    remove_emoji: bool = Query(True, description="Remove emojis"),
):
    try:
        logger.info("🧹 Cleaning special characters, numbers, and emojis...")

        cleaned, removed_chars = remove_special_characters(
            text=req.text,
            remove_special=remove_special,
            remove_numbers=remove_numbers,
            remove_emoji=remove_emoji,
        )

        return SpecialCharCleanedResponse(
            status="success",
            message="Text cleaned successfully",
            data=SpecialCharCleanedData(
                original=req.text, cleaned=cleaned, removed_characters=removed_chars
            ),
        )
    except Exception as e:
        logger.exception(f"❌ Failed to clean text::: {e}")
        raise HTTPException(status_code=500, detail="Failed to clean text.")


@router.post("/tokenize", response_model=TokenizedResponse)
async def tokenize_text_api(req: NormalizeRequest):
    try:
        logger.info("✂️ Tokenizing input text...")
        tokens = tokenize_text(req.text)

        return TokenizedResponse(
            status="success",
            message="Text tokenized successfully",
            data=TokenizedData(original=req.text, tokens=tokens),
        )
    except Exception as e:
        logger.exception(f"❌ Failed to tokenize text::: {e}")
        raise HTTPException(status_code=500, detail="Failed to tokenize text.")


@router.post("/remove-stopwords", response_model=StopwordTokenResponse)
async def remove_stopwords_api(req: StopwordTokenRequest):
    try:
        logger.info("🧹 Removing stopwords from token list...")

        result = remove_stopwords_from_tokens(
            tokens=req.tokens,
            custom_stopwords=req.custom_stopwords or [],
            exclude_stopwords=req.exclude_stopwords or [],
        )

        return StopwordTokenResponse(
            status="success",
            message="Stopwords removed successfully",
            data=StopwordTokenResponseData(**result),
        )
    except Exception as e:
        logger.exception(f"❌ Failed to remove stopwords::: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to remove stopwords from token list."
        )


@router.post("/lemmatize", response_model=LemmatizedResponse)
async def lemmatize_text_api(req: LemmatizeRequest):
    try:
        logger.info("🧬 Lemmatizing token list...")

        result = lemmatize_tokens(req.tokens)

        return LemmatizedResponse(
            status="success",
            message="Text lemmatized successfully",
            data=LemmatizedData(**result),
        )
    except Exception as e:
        logger.exception(f"❌ Failed to lemmatize tokens::: {e}")
        raise HTTPException(status_code=500, detail="Failed to lemmatize tokens.")
