from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import ast
import pandas as pd

from app.core.lemmatization.base import Lemmatizer
from app.core.lemmatization.config import LemmatizationConfig
from app.models.db.file_record import FileRecord
from app.services.file_service import FileService

from app.utils.exceptions import NotFoundError


@dataclass(frozen=True)
class LemmatizationResult:
    key: str
    url: str
    df: pd.DataFrame  # ['stopword_removed', 'lemmatized_tokens'] (lists)
    config: Dict[str, Any]
    recomputed: bool


class LemmatizationService:
    """
    Orchestrates lemmatization.
    - Reuse if config unchanged and not stale vs stopword step
    - Works with in-memory df_stop (pipeline) or loads from S3 (endpoint)
    """

    def __init__(self, files: FileService, lemmatizer: Lemmatizer):
        self.files = files
        self.lemmatizer = lemmatizer

    @staticmethod
    def _cfg_json(cfg: LemmatizationConfig | Dict) -> str:
        if isinstance(cfg, LemmatizationConfig):
            payload = {
                "use_pos_tagging": cfg.use_pos_tagging,
                "lowercase": cfg.lowercase,
                "fallback_if_missing": cfg.fallback_if_missing,
            }
        else:
            payload = cfg
        return json.dumps(payload, sort_keys=True)

    def _ensure_list(self, x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            # prefer JSON; fallback to Python literal
            try:
                return json.loads(x)
            except Exception:
                pass
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else [str(v)]
            except Exception:
                return [x]
        return [] if x is None else [str(x)]

    async def ensure_lemmatized(
        self,
        db,
        record: FileRecord,
        *,
        df_stop: Optional[pd.DataFrame] = None,  # in-memory ['stopword_removed']
        override_cfg: Optional[LemmatizationConfig] = None,
    ) -> LemmatizationResult:
        # Require stored stopword file only if df_stop not supplied
        if df_stop is None and not record.stopword_s3_key:
            raise NotFoundError(
                code="STOPWORD_FILE_NOT_FOUND",
                message="Stopword-removed file not found.",
            )

        default_cfg = getattr(self.lemmatizer, "cfg", LemmatizationConfig())
        effective_cfg = override_cfg or default_cfg
        cfg_json = self._cfg_json(effective_cfg)

        # Reuse only in endpoint mode (when relying on stored artifacts)
        can_reuse = (
            df_stop is None
            and record.lemmatized_s3_key
            and (getattr(record, "lemmatized_config", None) or cfg_json) == cfg_json
            and record.lemmatized_updated_at
            and (
                not record.stopword_updated_at
                or record.lemmatized_updated_at >= record.stopword_updated_at
            )
        )
        if can_reuse:
            df_lemm = await self.files.download_df(record.lemmatized_s3_key)
            url = await self.files.presigned_url(
                record.lemmatized_s3_key, expires_in=6000
            )
            return LemmatizationResult(
                key=record.lemmatized_s3_key,
                url=url,
                df=df_lemm,
                config=json.loads(cfg_json),
                recomputed=False,
            )

        # Compute path
        if df_stop is None:
            df_stop = await self.files.download_df(record.stopword_s3_key)

        if "stopword_removed" not in df_stop.columns:
            raise NotFoundError(
                code="STOPWORD_COL_NOT_FOUND",
                message="'stopword_removed' column not found.",
            )

        lemmas: List[List[str]] = []
        for x in df_stop["stopword_removed"]:
            toks = self._ensure_list(x)
            lemmas.append(self.lemmatizer.lemmatize(toks))

        out = pd.DataFrame(
            {
                "stopword_removed": df_stop["stopword_removed"],
                "lemmatized_tokens": lemmas,
            }
        )

        new_key = (
            f"lemmatization/lemmatized_{int(datetime.utcnow().timestamp())}.csv.gz"
        )
        key, url = await self.files.upload_df(out, new_key)

        # Update DB
        record.lemmatized_s3_key = key
        record.lemmatized_s3_url = url
        # Optional but recommended: add this column to track config changes
        #   lemmatized_config = Column(Text, nullable=True)
        record.lemmatized_config = cfg_json  # type: ignore[attr-defined]
        record.lemmatized_updated_at = datetime.utcnow()
        await db.commit()

        return LemmatizationResult(
            key=key,
            url=url,
            df=out,
            config=json.loads(cfg_json),
            recomputed=True,
        )
