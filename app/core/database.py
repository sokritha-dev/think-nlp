# app/core/database.py
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy.orm import declarative_base
from app.core.config import settings

Base = declarative_base()


class Database:
    def __init__(self):
        self._engine: AsyncEngine = create_async_engine(
            f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
            echo=False,
            future=True,
        )
        self._SessionLocal = async_sessionmaker(
            bind=self._engine, class_=AsyncSession, expire_on_commit=False
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def sync_engine(self):
        # The SQLAlchemy OTel instrumentor expects a sync Engine
        return self._engine.sync_engine

    async def get_db(self):
        async with self._SessionLocal() as session:
            yield session


database = Database()
get_db = database.get_db
async_session = database._SessionLocal
