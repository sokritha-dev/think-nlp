from dotenv import load_dotenv
from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from sqlalchemy.engine import URL
from alembic import context
import os

from app.core.database import Base
from app.models.db import (
    file_record,
    topic_model,
    sentiment_analysis,
)  # ensure models are imported

# Load env vars early
load_dotenv(".env.local")

# Set Alembic config
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Build dynamic DB URL
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    database=os.getenv("POSTGRES_DB"),
)

print(f"📦 Alembic using: {DATABASE_URL}")

# Set override in case other Alembic utils need it
config.set_main_option("sqlalchemy.url", str(DATABASE_URL))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=str(DATABASE_URL),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    engine = create_engine(str(DATABASE_URL), poolclass=pool.NullPool)

    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
