# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENV: str = "local"

    SERVICE_NAME: str | None = None
    DOCKERHUB_USERNAME: str | None = None  # PRODUCTION MODE ONLY
    DROPLET_USER: str | None = None  # PRODUCTION MODE ONLY
    DROPLET_HOST: str | None = None  # PRODUCTION MODE ONLY
    APP_IMAGE: str | None = None  # PRODUCTION MODE ONLY

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

    MAX_SIZE_FILE_UPLOAD: int | None = None

    FRONTEND_ORIGIN: str | None = None
    BETTERSTACK_OPENTELEMETRY_API_KEY: str | None = None  # PRODUCTION MODE ONLY
    BETTERSTACK_API_KEY: str | None = None  # PRODUCTION MODE ONLY
    BETTERSTACK_HOST: str | None = None  # PRODUCTION MODE ONLY
    BETTERSTACK_OTLP_TRACES_ENDPOINT: str | None = None  # PRODUCTION MODE ONLY
    BETTERSTACK_OTLP_METRICS_ENDPOINT: str | None = None  # PRODUCTION MODE ONLY

    OTEL_SERVICE_NAME: str | None = None
    OTEL_SERVICE_VERSION: str | None = None
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_SAMPLE_RATIO: str | None = None
    OTEL_ENABLE_METRICS: str | None = None

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
    )

    @property
    def SQLALCHEMY_DATABASE_URI(self):
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
