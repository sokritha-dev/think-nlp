from __future__ import annotations
import gzip
from io import BytesIO
from app.core.file_handler.base import CompressionBase


class GzipCompression(CompressionBase):
    def compress(self, raw_bytes: bytes) -> bytes:
        out = BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as gz:
            gz.write(raw_bytes)
        return out.getvalue()

    def decompress(self, raw_bytes: bytes) -> bytes:
        with gzip.GzipFile(fileobj=BytesIO(raw_bytes), mode="rb") as gz:
            return gz.read()
