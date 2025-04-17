from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# ✅ Import settings AFTER dotenv is loaded
from app.core.config import settings
from app.core.database import Base
from app.models.db import file_record


# ✔️ Alembic config
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ✔️ Alembic needs metadata to detect schema
target_metadata = Base.metadata

# ✔️ Build DB URL dynamically from settings
DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
