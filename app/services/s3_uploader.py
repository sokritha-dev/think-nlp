import gzip
import time
from io import BytesIO
from typing import Union
from uuid import uuid4
import aioboto3
import pandas as pd
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def get_s3_client():
    session = aioboto3.Session()
    return session.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


async def generate_presigned_url(bucket: str, key: str, expires_in: int = 6000) -> str:
    try:
        async with get_s3_client() as s3:
            start = time.time()
            url = await s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            logger.info(f"✅ Presigned URL generated in {time.time() - start:.2f}s")
            return url
    except Exception as e:
        logger.exception("❌ Failed to generate presigned URL")
        raise RuntimeError(f"Failed to generate presigned URL: {e}")


async def upload_file_to_s3(
    file_obj: BytesIO, s3_key: str, content_type: str = "application/octet-stream"
) -> str:
    file_obj.seek(0)
    try:
        async with get_s3_client() as s3:
            start = time.time()
            await s3.upload_fileobj(
                Fileobj=file_obj,
                Bucket=settings.AWS_S3_BUCKET_NAME,
                Key=s3_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "ContentDisposition": "inline",
                },
            )
            logger.info(f"✅ Uploaded {s3_key} in {time.time() - start:.2f}s")
            url = f"https://{settings.AWS_S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            return url
    except Exception as e:
        logger.exception("❌ S3 upload failed")
        raise Exception(f"S3 upload failed: {str(e)}")


async def upload_compressed_csv_to_s3(
    df_or_bytes: Union[pd.DataFrame, bytes],
    s3_key: str,
    content_type: str = "application/gzip",
) -> str:
    try:
        start = time.time()
        compressed_buffer = BytesIO()

        if isinstance(df_or_bytes, pd.DataFrame):
            with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz_file:
                df_or_bytes.to_csv(gz_file, index=False)
        elif isinstance(df_or_bytes, bytes):
            with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz_file:
                gz_file.write(df_or_bytes)
        else:
            raise TypeError("Expected pandas.DataFrame or raw CSV bytes.")

        compressed_buffer.seek(0)
        result = await upload_file_to_s3(
            compressed_buffer,
            s3_key=s3_key,
            content_type=content_type,
        )
        logger.info(
            f"✅ Uploaded Compressed File {s3_key} in {time.time() - start:.2f}s"
        )
        return result
    except Exception as e:
        logger.exception(f"❌ Failed to upload compressed CSV to S3: {e}")
        raise


async def delete_file_from_s3(s3_key: str):
    try:
        async with get_s3_client() as s3:
            start = time.time()
            await s3.delete_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=s3_key)
            logger.info(f"✅ Deleted {s3_key} in {time.time() - start:.2f}s")
    except Exception as e:
        logger.exception("❌ Failed to delete file from S3")
        raise Exception(f"S3 delete failed: {e}")


async def download_file_from_s3(s3_key: str) -> bytes:
    try:
        async with get_s3_client() as s3:
            start = time.time()
            response = await s3.get_object(
                Bucket=settings.AWS_S3_BUCKET_NAME, Key=s3_key
            )
            async with response["Body"] as stream:
                data = await stream.read()
            logger.info(f"✅ Downloaded {s3_key} in {time.time() - start:.2f}s")
            return data
    except Exception as e:
        logger.warning(f"⚠️ Failed to download {s3_key}: {e}")
        raise


async def save_csv_to_s3(
    df: pd.DataFrame, prefix: str, suffix: str = "", compress: bool = True
):
    try:
        start = time.time()
        key = (
            f"{prefix}/{suffix}_{uuid4()}.csv.gz"
            if compress
            else f"{prefix}/{suffix}_{uuid4()}.csv"
        )
        buffer = BytesIO()

        if compress:
            with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                df.to_csv(gz, index=False)
        else:
            df.to_csv(buffer, index=False)

        buffer.seek(0)
        s3_url = await upload_file_to_s3(buffer, key)
        logger.info(f"✅ Save CSV to S3 {key} in {time.time() - start:.2f}s")

        return key, s3_url
    except Exception as e:
        logger.exception(f"❌ Failed to save CSV to S3: {e}")
        raise
