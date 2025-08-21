from __future__ import annotations
from io import BytesIO
import pandas as pd

from app.core.file_handler.base import DataFrameCodecBase


class CsvCodec(DataFrameCodecBase):
    def __init__(self, **to_csv_kwargs):
        # e.g., to_csv_kwargs: {"index": False}
        self._to_csv_kwargs = {"index": False, **to_csv_kwargs}

    def to_bytes(self, df: pd.DataFrame) -> bytes:
        buf = BytesIO()
        df.to_csv(buf, **self._to_csv_kwargs)
        return buf.getvalue()

    def from_bytes(self, b: bytes) -> pd.DataFrame:
        return pd.read_csv(BytesIO(b))
