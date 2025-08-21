from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from app.models.db.file_record import FileRecord
from app.services.file_service import FileService
from app.core.eda.base import EDAAnalyzer
from app.core.eda.config import EDAConfig


@dataclass(frozen=True)
class EDAResult:
    eda: Dict[str, Any]
    reused: bool


class EDAService:
    """
    Orchestrates EDA:
      - Reuse prior EDA if not stale vs lemmatized_updated_at (and, optionally, config equal)
      - Load lemmatized CSV via FileService when df_lemm is not provided
      - Persist JSON into record.eda_analysis and update record.eda_updated_at
    """

    def __init__(self, analyzer: EDAAnalyzer, files: FileService | None = None):
        self.analyzer = analyzer
        self.files = files  # optional: only needed when we must download

    @staticmethod
    def _cfg_json(cfg: EDAConfig | Dict[str, Any]) -> str:
        if isinstance(cfg, EDAConfig):
            payload = {
                "text_column": cfg.text_column,
                "top_words": cfg.top_words,
                "ngram_top_k": cfg.ngram_top_k,
                "use_sklearn_stopwords": cfg.use_sklearn_stopwords,
            }
        else:
            payload = cfg
        return json.dumps(payload, sort_keys=True)

    async def ensure_eda(
        self,
        db,
        record: FileRecord,
        *,
        df_lemm: Optional[pd.DataFrame] = None,
        override_config: Optional[EDAConfig] = None,
    ) -> EDAResult:
        effective_cfg = override_config or getattr(self.analyzer, "cfg", EDAConfig())
        cfg_json = self._cfg_json(effective_cfg)

        # Reuse if we have EDA and it's not stale vs lemmatized step
        can_reuse = (
            record.eda_analysis
            and record.eda_updated_at
            and record.lemmatized_updated_at
            and record.eda_updated_at >= record.lemmatized_updated_at
            # Optional: also compare configs if you add an 'eda_config' column:
            # and getattr(record, "eda_config", cfg_json) == cfg_json
        )
        if can_reuse:
            return EDAResult(eda=json.loads(record.eda_analysis), reused=True)

        # Need data
        if df_lemm is None:
            if not self.files:
                raise RuntimeError(
                    "EDAService requires FileService to load lemmatized CSV when df_lemm is None."
                )
            if not record.lemmatized_s3_key:
                raise FileNotFoundError("Lemmatized CSV not found for this record.")
            df_lemm = await self.files.download_df(record.lemmatized_s3_key)

        # Run analyzer
        eda_dict = self.analyzer.analyze(df_lemm)

        # Persist
        record.eda_analysis = json.dumps(eda_dict)
        record.eda_updated_at = datetime.utcnow()
        # Optional if you add the field:
        # record.eda_config = cfg_json
        await db.commit()

        return EDAResult(eda=eda_dict, reused=False)
