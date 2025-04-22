from io import BytesIO
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

print(f"settings aws:: {settings.AWS_ACCESS_KEY_ID}")


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


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
