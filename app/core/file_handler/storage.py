from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Optional
import aioboto3
from botocore.config import Config
from app.core.config import settings
from app.core.file_handler.base import StorageBase

logger = logging.getLogger(__name__)


class S3Storage(StorageBase):
    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,  # keep None for AWS; set for MinIO/etc
        public_url_style: str = "virtual",  # "virtual" or "path"
    ):
        self.bucket = bucket or settings.AWS_S3_BUCKET_NAME
        self.region = region or settings.AWS_REGION
        self.access_key = access_key or settings.AWS_ACCESS_KEY_ID
        self.secret_key = secret_key or settings.AWS_SECRET_ACCESS_KEY
        self.endpoint_url = endpoint_url
        self.public_url_style = public_url_style

        # A little sane client config (retries, timeouts)
        self._boto_cfg = Config(
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
            s3={"addressing_style": "virtual"},  # works with AWS; set "path" for MinIO
            signature_version="s3v4",
        )

        self._session = aioboto3.Session()

    @asynccontextmanager
    async def _client(self):
        async with self._session.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            config=self._boto_cfg,
        ) as s3:
            yield s3

    # ---------- StorageBase API ----------

    async def download(self, key: str) -> bytes:
        try:
            async with self._client() as s3:
                resp = await s3.get_object(Bucket=self.bucket, Key=key)
                async with resp["Body"] as stream:
                    data = await stream.read()
                return data
        except Exception as e:
            logger.exception(f"S3 download failed: key={key}: {e}")
            raise

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload raw bytes. Returns a public URL (not presigned).
        """
        try:
            extra = {
                "ContentType": content_type or "application/octet-stream",
                "ContentDisposition": "inline",
            }

            async with self._client() as s3:
                # put_object is simple and reliable for bytes
                await s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

            return self._public_url(key)
        except Exception as e:
            logger.exception(f"S3 upload failed: key={key}: {e}")
            raise

    async def presigned_url(self, key: str, expires_in: int = 3600) -> str:
        try:
            async with self._client() as s3:
                url = await s3.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=expires_in,
                )
                return url
        except Exception as e:
            logger.exception(f"Presigned URL failed: key={key}: {e}")
            raise

    async def delete(self, key: str) -> None:
        try:
            async with self._client() as s3:
                await s3.delete_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=key)
        except Exception as e:
            logger.exception("❌ Failed to delete file from S3")
            raise Exception(f"S3 delete failed: {e}")

    # ---------- Helpers ----------

    def _public_url(self, key: str) -> str:
        """Construct a public-style URL. For private buckets, prefer presigned_url()."""
        if self.endpoint_url:
            # Custom endpoint (e.g., MinIO). Choose style.
            if self.public_url_style == "path":
                # http(s)://endpoint/bucket/key
                return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
            # virtual-hosted: http(s)://bucket.endpoint/key
            base = self.endpoint_url.replace("https://", "").replace("http://", "")
            return f"https://{self.bucket}.{base}/{key}"

        # AWS standard virtual-hosted–style URL
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
