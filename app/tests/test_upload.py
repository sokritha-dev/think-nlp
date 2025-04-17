import io
import os
from fastapi.testclient import TestClient
from moto import mock_aws
import boto3

from app.main import app

client = TestClient(app)


def generate_csv(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


@mock_aws
def test_valid_csv_upload():
    # Setup test environment and session
    region = "us-east-1"
    bucket_name = "nlp-learner"

    # Ensure app uses this mocked bucket
    os.environ["AWS_REGION"] = region
    os.environ["AWS_S3_BUCKET_NAME"] = bucket_name

    # Ensure consistent default session for moto
    boto3.setup_default_session(region_name=region)

    # Create the mocked S3 bucket
    s3 = boto3.client("s3", region_name=region)
    s3.create_bucket(Bucket=bucket_name)

    # Test POST request to upload
    csv_data = "review,rating\nGreat hotel!,5\nVery bad,1\n"
    response = client.post(
        "/api/upload/",
        files={"file": ("sample.csv", generate_csv(csv_data), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["record_count"] == 2
    assert "file_url" in body["data"]
    assert "review" in body["data"]["columns"]


# -------------------------------------
# ❌ Missing required 'review' column
# -------------------------------------
@mock_aws
def test_upload_missing_review_column():
    bucket_name = "test-bucket"
    os.environ["AWS_S3_BUCKET_NAME"] = bucket_name
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    csv_data = "comment,rating\nNice!,5\nBad!,1\n"
    response = client.post(
        "/api/upload/",
        files={"file": ("missing_review.csv", generate_csv(csv_data), "text/csv")},
    )

    assert response.status_code == 400
    assert "review" in response.json()["detail"].lower()


# -------------------------------------
# ❌ Non-CSV file type (e.g. .txt)
# -------------------------------------
def test_upload_non_csv_file():
    txt_data = "this is not csv"
    response = client.post(
        "/api/upload/",
        files={"file": ("notes.txt", generate_csv(txt_data), "text/plain")},
    )
    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()


# -------------------------------------
# ❌ Invalid CSV format (e.g. syntax error)
# -------------------------------------
def test_upload_invalid_csv_format():
    csv_data = 'review,rating\n"Unclosed quote\n'
    response = client.post(
        "/api/upload/",
        files={"file": ("broken.csv", generate_csv(csv_data), "text/csv")},
    )
    assert response.status_code == 400
    assert "invalid csv format" in response.json()["detail"].lower()


# -------------------------------------
# ❌ Simulated S3 upload failure
# -------------------------------------
@mock_aws
def test_upload_s3_failure(monkeypatch):
    bucket_name = "test-bucket"
    os.environ["AWS_S3_BUCKET_NAME"] = bucket_name
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    # Simulate upload failure by forcing exception
    monkeypatch.setattr(
        "app.services.s3_uploader.upload_file_to_s3",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("simulated failure")),
    )

    csv_data = "review,rating\nNice!,5\nBad!,1\n"
    response = client.post(
        "/api/upload/",
        files={"file": ("crash.csv", generate_csv(csv_data), "text/csv")},
    )

    assert response.status_code == 500
    assert "internal server error" in response.text.lower()


# -------------------------------------
# ❌ Upload empty CSV file
# -------------------------------------
@mock_aws
def test_upload_empty_csv(monkeypatch):
    bucket_name = "test-bucket"
    os.environ["AWS_S3_BUCKET_NAME"] = bucket_name
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    empty_data = ""
    response = client.post(
        "/api/upload/",
        files={"file": ("empty.csv", generate_csv(empty_data), "text/csv")},
    )

    assert response.status_code == 400
    assert "invalid csv format" in response.json()["detail"].lower()
