from io import BytesIO
from typing import Optional
from uuid import uuid4
import boto3
from botocore.exceptions import BotoCoreError, ClientError
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


def save_csv_to_s3(df, folder: str, suffix: str, base_key: Optional[str] = None):
    """Helper function to save a dataframe to S3."""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    if base_key:
        # Reuse the base key but modify suffix
        new_key = base_key.replace(".csv", f"_{suffix}.csv")
    else:
        new_key = f"{folder}/{suffix}_{uuid4()}.csv"

    url = upload_file_to_s3(buffer, new_key, content_type="text/csv")
    return new_key, url
