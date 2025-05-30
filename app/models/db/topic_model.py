# app/models/db/topic_model.py

from sqlalchemy import Column, String, ForeignKey, Integer, DateTime
from sqlalchemy.sql import func
from app.core.database import Base
from sqlalchemy.orm import relationship


class TopicModel(Base):
    __tablename__ = "topic_model"

    id = Column(String, primary_key=True, index=True)
    file_id = Column(
        String, ForeignKey("file_records.id", ondelete="CASCADE"), nullable=False
    )
    method = Column(String, nullable=False)  # "LDA", "NMF", "BERTopic"
    topic_count = Column(Integer, nullable=True)
    s3_key = Column(String, nullable=False, unique=True)
    s3_url = Column(String, nullable=False)
    summary_json = Column(
        String, nullable=True
    )  # Can store topic/keyword info as JSON string
    label_keywords = Column(String, nullable=True)
    label_map_json = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    topic_updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    label_updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    sentiments = relationship(
        "SentimentAnalysis",
        backref="topic_model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
