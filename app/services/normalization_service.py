from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict

import pandas as pd

from app.models.db.file_record import FileRecord
from app.services.file_service import FileService
from app.core.normalization.base import TextNormalizer
from app.core.normalization.config import NormalizationConfig


@dataclass(frozen=True)
class NormalizationResult:
    key: str
    url: str
    df: pd.DataFrame
    broken_map: Dict[str, str]
    recomputed: bool


class NormalizationService:
    """
    Use case / orchestrator:
      - picks config (DB value or override)
      - decides reuse vs recompute
      - does S3 I/O via FileService
      - updates DB metadata
    """

    def __init__(self, files: FileService, normalizer: TextNormalizer):
        self.files = files
        self.normalizer = normalizer

    @staticmethod
    def _serialize_cfg(cfg: NormalizationConfig | Dict[str, str]) -> str:
        """Store ONLY the flat dict in DB (no wrapper)."""
        if isinstance(cfg, NormalizationConfig):
            payload = cfg.broken_map
        else:
            payload = cfg
        # canonical JSON to make equality checks stable
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _extract_broken_map(
        db_json: Optional[str], fallback: Dict[str, str]
    ) -> Dict[str, str]:
        if not db_json:
            return fallback
        try:
            obj = json.loads(db_json)
            if isinstance(obj, dict):
                # support both legacy nested {"broken_map": {...}} and flat {...}
                return obj.get("broken_map", obj)
        except Exception:
            pass
        return fallback

    @staticmethod
    def _extract_broken_map(
        db_json: Optional[str], fallback: Dict[str, str]
    ) -> Dict[str, str]:
        if not db_json:
            return fallback
        try:
            obj = json.loads(db_json)
            bm = obj.get("broken_map", {})
            if isinstance(bm, dict):
                return bm
            return fallback
        except Exception:
            return fallback

    async def ensure_normalized(
        self,
        db,
        record: FileRecord,
        *,
        override_broken_map: Optional[Dict[str, str]] = None,
    ) -> NormalizationResult:
        """
        Ensure normalized CSV exists for this record with the intended config.
        Reuse if config matches; otherwise compute + upload + update DB.
        """

        # 1) Effective config
        # Start from DB (if present), then override if user provided.
        db_broken = self._extract_broken_map(
            record.normalized_broken_map,  # JSON string or None
            fallback=getattr(self.normalizer, "cfg", NormalizationConfig()).broken_map,
        )
        effective_broken = (
            override_broken_map if override_broken_map is not None else db_broken
        )
        cfg_json = self._serialize_cfg(effective_broken)

        # 2) Reuse if config unchanged
        if record.normalized_s3_key and record.normalized_broken_map == cfg_json:
            df_norm = await self.files.download_df(record.normalized_s3_key)
            url = await self.files.presigned_url(
                record.normalized_s3_key, expires_in=6000
            )
            return NormalizationResult(
                key=record.normalized_s3_key,
                url=url,
                df=df_norm,
                broken_map=effective_broken,
                recomputed=False,
            )

        # 3) Compute from original
        df_src = await self.files.download_df(record.s3_key)
        out = df_src.loc[:, ["review"]].copy()
        out["normalized_review"] = (
            out["review"]
            .fillna("")
            .apply(lambda s: self.normalizer.normalize(str(s).replace("\x00", "")))
        )

        # 4) Upload compressed (by extension)
        new_key = f"normalization/normalized_{datetime.now(timezone.utc)}.csv.gz"
        key, url = await self.files.upload_df(out, new_key)

        # 5) Update DB atomically
        record.normalized_s3_key = key
        record.normalized_s3_url = url
        record.normalized_broken_map = cfg_json
        record.normalized_updated_at = datetime.now(timezone.utc)
        await db.commit()

        return NormalizationResult(
            key=key,
            url=url,
            df=out,
            broken_map=effective_broken,
            recomputed=True,
        )
