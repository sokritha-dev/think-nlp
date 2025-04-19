# app/models/db/file_record.py

from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class FileRecord(Base):
    __tablename__ = "file_records"

    id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    s3_key = Column(String, nullable=False, unique=True)
    s3_url = Column(String, nullable=False)
    columns = Column(String, nullable=False)  # store as comma-separated
    record_count = Column(Integer, nullable=False)
    file_hash = Column(String, nullable=False, unique=True)  # ✅ for deduplication
    normalized_s3_key = Column(String, nullable=True)
    normalized_s3_url = Column(String, nullable=True)
    special_cleaned_s3_key = Column(String, nullable=True)
    special_cleaned_s3_url = Column(String, nullable=True)
    tokenized_s3_key = Column(String, nullable=True)
    tokenized_s3_url = Column(String, nullable=True)
    stopword_s3_key = Column(String, nullable=True)
    stopword_s3_url = Column(String, nullable=True)
    lemmatized_s3_key = Column(String, nullable=True)
    lemmatized_s3_url = Column(String, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())