import gzip
from io import BytesIO
from typing import Union
from uuid import uuid4
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import pandas as pd
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


def generate_presigned_url(bucket: str, key: str, expires_in: int = 6000) -> str:
    s3_client = get_s3_client()

    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate presigned URL: {e}")


def upload_file_to_s3(
    file_obj: BytesIO, s3_key: str, content_type: str = "application/octet-stream"
) -> str:
    file_obj.seek(0)
    s3 = get_s3_client()
    try:
        s3.upload_fileobj(
            file_obj,
            settings.AWS_S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                "ContentType": content_type,
                "ContentDisposition": "inline",  # allows browser to render it instead of download
            },
        )
        url = f"https://{settings.AWS_S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
        return url
    except (BotoCoreError, ClientError) as e:
        logger.exception("S3 upload failed")
        raise Exception(f"S3 upload failed: {str(e)}")


def upload_compressed_csv_to_s3(
    df_or_bytes: Union[pd.DataFrame, bytes],
    s3_key: str,
    content_type: str = "application/gzip",
) -> str:
    """
    Compress and upload CSV data to S3 as a .csv.gz file.

    Args:
        df_or_bytes: A pandas DataFrame or raw CSV bytes.
        s3_key: S3 key including `.csv.gz` suffix.
        content_type: MIME type to set on the object.

    Returns:
        The S3 public URL of the uploaded file.
    """
    compressed_buffer = BytesIO()

    # Handle both DataFrame and raw bytes
    if isinstance(df_or_bytes, pd.DataFrame):
        with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz_file:
            df_or_bytes.to_csv(gz_file, index=False)
    elif isinstance(df_or_bytes, bytes):
        with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz_file:
            gz_file.write(df_or_bytes)
    else:
        raise TypeError("Expected pandas.DataFrame or raw CSV bytes.")

    compressed_buffer.seek(0)

    return upload_file_to_s3(
        compressed_buffer,
        s3_key=s3_key,
        content_type=content_type,
    )


def delete_file_from_s3(s3_key: str):
    s3 = get_s3_client()
    try:
        s3.delete_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=s3_key)
    except ClientError as e:
        logger.exception("Failed to delete file from S3")
        raise Exception(f"S3 delete failed: {e}")


def download_file_from_s3(s3_key: str) -> bytes:
    try:
        s3 = get_s3_client()
        response = s3.get_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=s3_key)
        return response["Body"].read()
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"⚠️ Failed to download {s3_key} from S3: {e}")
        raise


def save_csv_to_s3(
    df: pd.DataFrame, prefix: str, suffix: str = "", compress: bool = True
):
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
    s3_url = upload_file_to_s3(buffer, key)
    return key, s3_url
