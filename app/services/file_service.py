from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from app.core.file_handler.base import (
    StorageBase,
    CompressionBase,
    DataFrameCodecBase,
)


def _guess_content_type(key: str, compressed: bool, explicit: Optional[str]) -> str:
    """
    Prefer explicit content_type when provided.
    Otherwise infer basic types from extension + compression flag.
    """
    if explicit:
        return explicit
    if compressed or key.endswith(".gz"):
        return "application/gzip"
    if key.endswith(".csv"):
        return "text/csv"
    # fallthrough
    return "application/octet-stream"


@dataclass(slots=True)
class FileService:
    """
    High-level file I/O service that:
      - downloads/uploads raw bytes or DataFrames
      - (de)compresses transparently when a CompressionBase is provided
      - infers behavior from file extension when auto_by_extension=True
    """

    storage: StorageBase
    codec: DataFrameCodecBase
    compression: Optional[CompressionBase] = None
    auto_by_extension: bool = True  # if True, treat *.gz as compressed

    # ------------- internals -------------

    def _is_compressed_key(self, key: str) -> bool:
        return self.auto_by_extension and key.lower().endswith(".gz")

    # ------------- raw bytes API -------------

    async def download_raw(self, key: str) -> bytes:
        """
        Download object bytes from storage.
        If compression is configured and key looks compressed (e.g. *.gz),
        transparently DECOMPRESS and return the raw (uncompressed) bytes.
        """
        data = await self.storage.download(key)
        if self.compression and self._is_compressed_key(key):
            data = self.compression.decompress(data)
        return data

    async def upload_raw(
        self,
        data: bytes,
        key: str,
        *,
        content_type: Optional[str] = None,
        force_compress: Optional[bool] = None,
    ) -> Tuple[str, str]:
        """
        Upload raw bytes to storage.
        If compression is configured and (force_compress is True OR key looks compressed),
        COMPRESS before uploading.

        Returns: (key, url)
        """
        should_compress = bool(self.compression) and (
            (force_compress is True)
            or (force_compress is None and self._is_compressed_key(key))
        )

        to_upload = self.compression.compress(data) if should_compress else data
        ct = _guess_content_type(key, should_compress, content_type)

        url = await self.storage.upload(key, to_upload, content_type=ct)
        return key, url

    async def delete(self, key: str) -> None:
        """
        Delete an object from storage.
        Delegates to the underlying StorageBase implementation.
        """
        if hasattr(self.storage, "delete"):
            await self.storage.delete(key)
        else:
            raise NotImplementedError("Underlying storage does not support delete().")

    async def safe_delete(self, key: str) -> bool:
        """
        Best-effort delete that never raises. Returns True if deleted,
        False otherwise (and logs are expected in storage.delete()).
        """
        try:
            await self.delete(key)
            return True
        except Exception:
            return False

    # ------------- DataFrame API -------------

    async def download_df(self, key: str) -> pd.DataFrame:
        """
        Download object and return as DataFrame using the configured codec.
        Handles decompression automatically (see download_raw).
        """
        raw = await self.download_raw(key)
        return self.codec.from_bytes(raw)

    async def upload_df(
        self,
        df: pd.DataFrame,
        key: str,
        *,
        content_type: Optional[str] = None,
        force_compress: Optional[bool] = None,
    ) -> Tuple[str, str]:
        """
        Encode DataFrame with codec, optionally compress based on key/flag,
        and upload to storage.

        Returns: (key, url)
        """
        raw = self.codec.to_bytes(df)
        return await self.upload_raw(
            raw,
            key,
            content_type=content_type,
            force_compress=force_compress,
        )

    # ------------- Utilities -------------

    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Delegate to storage for a temporary GET URL."""
        return await self.storage.presigned_url(key, expires_in=expires_in)

    async def copy_with_compression(
        self,
        src_key: str,
        dest_key: str,
        *,
        dest_force_compress: Optional[bool] = None,
        dest_content_type: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Read src (auto-decompress if needed), then write to dest
        (auto-compress if dest looks compressed or dest_force_compress=True).

        This is ideal for “compress” or “decompress” operations between keys.
        """
        # Pull uncompressed payload regardless of src encoding
        raw = await self.download_raw(src_key)

        # Push with desired compression policy for the destination
        return await self.upload_raw(
            raw,
            dest_key,
            content_type=dest_content_type,
            force_compress=dest_force_compress,
        )
