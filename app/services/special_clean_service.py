from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, List

import pandas as pd

from app.models.db.file_record import FileRecord
from app.services.file_service import FileService
from app.core.special_char_removal.base import SpecialCleaner
from app.core.special_char_removal.config import SpecialCleanConfig
from app.utils.exceptions import NotFoundError


@dataclass(frozen=True)
class SpecialCleanResult:
    key: str
    url: str
    df: pd.DataFrame  # columns: ['normalized_review', 'special_cleaned']
    flags: Dict[str, bool]
    removed_characters: List[str]
    recomputed: bool


class SpecialCleanService:
    """
    Orchestrates the 'remove special characters' step.

    Works in two modes:
      - Endpoint mode: pulls normalized CSV from S3 (via FileService)
      - Pipeline mode: accepts in-memory df_norm to avoid re-downloading

    Reuse conditions:
      - existing special_cleaned_s3_key
      - flags unchanged
      - special_cleaned_updated_at >= normalized_updated_at
    """

    def __init__(self, files: FileService, cleaner: SpecialCleaner):
        self.files = files
        self.cleaner = cleaner

    @staticmethod
    def _flags_to_dict(cfg: SpecialCleanConfig) -> Dict[str, bool]:
        return {
            "remove_special": cfg.remove_special,
            "remove_numbers": cfg.remove_numbers,
            "remove_emoji": cfg.remove_emoji,
        }

    @staticmethod
    def _flags_to_json(flags: Dict[str, bool]) -> str:
        return json.dumps(flags, sort_keys=True)

    async def ensure_special_cleaned(
        self,
        db,
        record: FileRecord,
        *,
        override_flags: Optional[Dict[str, bool]] = None,
        df_norm: Optional[pd.DataFrame] = None,  # <— for pipeline
    ) -> SpecialCleanResult:
        if df_norm is None and not record.normalized_s3_key:
            raise NotFoundError(
                code="NORMALIZED_FILE_NOT_FOUND", message="Normalized file not found."
            )

        # Effective flags
        default_cfg = getattr(self.cleaner, "cfg", SpecialCleanConfig())
        flags = self._flags_to_dict(default_cfg)
        if override_flags:
            flags.update(override_flags)
        flags_json = self._flags_to_json(flags)

        # Reuse path
        can_reuse = (
            record.special_cleaned_s3_key
            and record.special_cleaned_flags == flags_json
            and record.special_cleaned_updated_at
            and (
                not record.normalized_updated_at
                or record.special_cleaned_updated_at >= record.normalized_updated_at
            )
        )

        if can_reuse:
            df_clean = await self.files.download_df(record.special_cleaned_s3_key)
            url = await self.files.presigned_url(
                record.special_cleaned_s3_key, expires_in=6000
            )
            removed_list = json.loads(record.special_cleaned_removed or "[]")
            return SpecialCleanResult(
                key=record.special_cleaned_s3_key,
                url=url,
                df=df_clean,
                flags=flags,
                removed_characters=removed_list,
                recomputed=False,
            )

        # Compute path
        # Prefer in-memory df_norm (pipeline), otherwise load from S3 (endpoint)
        if df_norm is None:
            df_norm = await self.files.download_df(record.normalized_s3_key)

        if "normalized_review" not in df_norm.columns:
            # backward compatibility
            df_norm = df_norm.assign(normalized_review=df_norm["review"].fillna(""))

        rows = []
        removed_all: List[str] = []
        for text in df_norm["normalized_review"].fillna(""):
            cleaned, removed = self.cleaner.clean(text)
            rows.append({"normalized_review": text, "special_cleaned": cleaned})
            removed_all.extend(removed)

        out = pd.DataFrame(rows)
        removed_unique_sorted = sorted(list(set(removed_all)))

        new_key = f"cleaned/special_cleaned_{int(datetime.now(timezone.utc).timestamp())}.csv.gz"
        key, url = await self.files.upload_df(out, new_key)

        # Update DB once
        record.special_cleaned_s3_key = key
        record.special_cleaned_s3_url = url
        record.special_cleaned_flags = flags_json
        record.special_cleaned_removed = json.dumps(removed_unique_sorted)
        record.special_cleaned_updated_at = datetime.now(timezone.utc)
        await db.commit()

        return SpecialCleanResult(
            key=key,
            url=url,
            df=out,
            flags=flags,
            removed_characters=removed_unique_sorted,
            recomputed=True,
        )
