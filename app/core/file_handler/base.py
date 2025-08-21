from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Optional
import pandas as pd


class StorageBase(ABC):
    """Abstract storage interface (e.g., S3, GCS, local FS)."""

    @abstractmethod
    async def download(self, key: str) -> bytes: ...

    @abstractmethod
    async def upload(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> str:
        """Upload raw bytes and return a (pre)signed URL or public URL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete an object from storage."""
        ...

    @abstractmethod
    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get a temporary download URL for the object."""
        ...


class CompressionBase(ABC):
    """Abstract compression interface (gzip, zip, none)."""

    @abstractmethod
    def compress(self, raw_bytes: bytes) -> bytes: ...

    @abstractmethod
    def decompress(self, raw_bytes: bytes) -> bytes: ...


class DataFrameCodecBase(Protocol):
    """Encode/decode DataFrames (CSV, Parquet, JSONL)."""

    def to_bytes(self, df: pd.DataFrame) -> bytes: ...
    def from_bytes(self, b: bytes) -> pd.DataFrame: ...
