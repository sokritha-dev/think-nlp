from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
import ast

import pandas as pd

from app.core.stopword_removal.base import StopwordRemover
from app.core.stopword_removal.config import StopwordConfig
from app.models.db.file_record import FileRecord
from app.services.file_service import FileService
from app.utils.exceptions import NotFoundError


@dataclass(frozen=True)
class StopwordResult:
    key: str
    url: str
    df: pd.DataFrame  # columns: ['tokens', 'stopword_removed'] (lists)
    config: Dict[str, Any]
    recomputed: bool


class StopwordService:
    """
    Orchestrates stopword removal.
    - Reuse if config unchanged and not stale vs tokenized
    - Works with in-memory df_tokens (pipeline) or loads from S3 (endpoint)
    """

    def __init__(self, files: FileService, remover: StopwordRemover):
        self.files = files
        self.remover = remover

    @staticmethod
    def _cfg_json(cfg: StopwordConfig | Dict) -> str:
        if isinstance(cfg, StopwordConfig):
            payload = {
                "language": cfg.language,
                "custom_stopwords": sorted(list(cfg.custom_stopwords)),
                "exclude_stopwords": sorted(list(cfg.exclude_stopwords)),
                "lowercase": cfg.lowercase,
                "preserve_negations": cfg.preserve_negations,
            }
        else:
            payload = cfg
        return json.dumps(payload, sort_keys=True)

    def _ensure_list(self, x):
        """
        Make sure tokens are a list.
        Accept list, JSON string, or Python literal string (for backward compatibility).
        """
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            # try JSON first
            try:
                return json.loads(x)
            except Exception:
                pass
            # then Python literal (older CSVs)
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else [str(v)]
            except Exception:
                return [x]
        return [] if x is None else [str(x)]

    async def ensure_stopwords_removed(
        self,
        db,
        record: FileRecord,
        *,
        df_tokens: Optional[pd.DataFrame] = None,  # in-memory ['tokens']
        override_config: Optional[StopwordConfig] = None,
    ) -> StopwordResult:
        # ✅ Only require stored tokenized key if df_tokens not provided
        if df_tokens is None and not record.tokenized_s3_key:
            raise NotFoundError(
                code="TOKENIZED_FILE_NOT_FOUND", message="Tokenized file not found."
            )

        # Effective config
        default_cfg = getattr(self.remover, "cfg", StopwordConfig())
        effective_cfg = override_config or default_cfg
        cfg_json = self._cfg_json(effective_cfg)

        # Reuse path (endpoint mode only)
        can_reuse = (
            df_tokens is None
            and record.stopword_s3_key
            and (getattr(record, "stopword_config", None) or cfg_json) == cfg_json
            and record.stopword_updated_at
            and (
                not record.tokenized_updated_at
                or record.stopword_updated_at >= record.tokenized_updated_at
            )
        )
        if can_reuse:
            df_sw = await self.files.download_df(record.stopword_s3_key)
            url = await self.files.presigned_url(
                record.stopword_s3_key, expires_in=6000
            )
            return StopwordResult(
                key=record.stopword_s3_key,
                url=url,
                df=df_sw,
                config=json.loads(cfg_json),
                recomputed=False,
            )

        # Compute path
        if df_tokens is None:
            df_tokens = await self.files.download_df(record.tokenized_s3_key)

        if "tokens" not in df_tokens.columns:
            raise NotFoundError(
                code="TOKENS_COL_NOT_FOUND", message="'tokens' column not found."
            )

        cleaned_col: List[List[str]] = []
        for x in df_tokens["tokens"]:
            toks = self._ensure_list(x)
            cleaned, _removed = self.remover.remove(toks)
            cleaned_col.append(cleaned)

        out = pd.DataFrame(
            {
                "tokens": df_tokens["tokens"],
                "stopword_removed": cleaned_col,
            }
        )

        new_key = f"stopwords/stopword_removed_{datetime.now(timezone.utc)}.csv.gz"
        key, url = await self.files.upload_df(out, new_key)

        # Update DB
        record.stopword_s3_key = key
        record.stopword_s3_url = url
        record.stopword_config = cfg_json
        record.stopword_updated_at = datetime.now(timezone.utc)
        await db.commit()

        return StopwordResult(
            key=key,
            url=url,
            df=out,
            config=json.loads(cfg_json),
            recomputed=True,
        )
