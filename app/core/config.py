# app/core/config.py

from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_S3_BUCKET_NAME: str | None = None
    AWS_REGION: str = "us-east-1"

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    PGADMIN_DEFAULT_EMAIL: str | None = None
    PGADMIN_DEFAULT_PASSWORD: str | None = None

    class Config:
        # Dynamically choose the env file
        env_file = f".env.{os.getenv('ENV', 'local')}"  # default to .env.local
        env_file_encoding = "utf-8"


settings = Settings()
