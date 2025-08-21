from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List

import pandas as pd

from app.models.db.file_record import FileRecord
from app.services.file_service import FileService
from app.core.tokenization.base import Tokenizer
from app.core.tokenization.config import TokenizationConfig
from app.utils.exceptions import NotFoundError


@dataclass(frozen=True)
class TokenizationResult:
    key: str
    url: str
    df: pd.DataFrame  # columns: ['special_cleaned', 'tokens']
    config: Dict  # serialized config dict
    recomputed: bool


class TokenizationService:
    """
    Orchestrates tokenization.
    - Reuse if config unchanged and not stale vs special-cleaned
    - Works with in-memory df_clean (pipeline) or loads from S3 (endpoint)
    """

    def __init__(self, files: FileService, tokenizer: Tokenizer):
        self.files = files
        self.tokenizer = tokenizer

    @staticmethod
    def _config_to_json(cfg: TokenizationConfig | Dict) -> str:
        if isinstance(cfg, TokenizationConfig):
            payload = {
                "method": cfg.method,
                "regex_pattern": cfg.regex_pattern,
                "lowercase": cfg.lowercase,
                "min_token_len": cfg.min_token_len,
                "keep_alnum_only": cfg.keep_alnum_only,
                "remove_numbers_only": cfg.remove_numbers_only,
                "drop_empty_tokens": cfg.drop_empty_tokens,
            }
        else:
            payload = cfg
        return json.dumps(payload, sort_keys=True)

    async def ensure_tokenized(
        self,
        db,
        record: FileRecord,
        *,
        df_clean: Optional[pd.DataFrame] = None,  # in-memory ['special_cleaned']
        override_config: Optional[TokenizationConfig] = None,
    ) -> TokenizationResult:
        # ✅ only require stored special_clean if df_clean is not supplied
        if df_clean is None and not record.special_cleaned_s3_key:
            raise NotFoundError(
                code="CLEANED_FILE_NOT_FOUND", message="Special cleaned file not found."
            )

        # Effective config (defaults from tokenizer)
        default_cfg = getattr(self.tokenizer, "cfg", TokenizationConfig())
        effective_cfg = override_config or default_cfg
        cfg_json = self._config_to_json(effective_cfg)

        # Reuse only if we rely on stored artifact (endpoint mode)
        can_reuse = (
            df_clean is None
            and record.tokenized_s3_key
            and (getattr(record, "tokenized_config", None) or cfg_json) == cfg_json
            and record.tokenized_updated_at
            and (
                not record.special_cleaned_updated_at
                or record.tokenized_updated_at >= record.special_cleaned_updated_at
            )
        )
        if can_reuse:
            df_tok = await self.files.download_df(record.tokenized_s3_key)
            url = await self.files.presigned_url(
                record.tokenized_s3_key, expires_in=6000
            )
            return TokenizationResult(
                key=record.tokenized_s3_key,
                url=url,
                df=df_tok,
                config=json.loads(cfg_json),
                recomputed=False,
            )

        # Compute path
        if df_clean is None:
            df_clean = await self.files.download_df(record.special_cleaned_s3_key)

        # ensure expected column
        if "special_cleaned" not in df_clean.columns:
            # backward compatibility if someone passed normalized text
            fallback_col = (
                "normalized_review"
                if "normalized_review" in df_clean.columns
                else "review"
            )
            df_clean = df_clean.assign(
                special_cleaned=df_clean[fallback_col].fillna("")
            )

        # tokenize row-by-row
        tokens_col: List[List[str]] = []
        for s in df_clean["special_cleaned"].fillna(""):
            tokens_col.append(self.tokenizer.tokenize(s))

        out = pd.DataFrame(
            {
                "special_cleaned": df_clean["special_cleaned"].fillna(""),
                "tokens": tokens_col,
            }
        )

        new_key = f"tokenization/tokens_{datetime.now(timezone.utc)}.csv.gz"
        key, url = await self.files.upload_df(out, new_key)

        # Update DB
        record.tokenized_s3_key = key
        record.tokenized_s3_url = url
        record.tokenized_config = cfg_json
        record.tokenized_updated_at = datetime.now(timezone.utc)
        await db.commit()

        return TokenizationResult(
            key=key,
            url=url,
            df=out,
            config=json.loads(cfg_json),
            recomputed=True,
        )
