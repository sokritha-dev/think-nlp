from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class SentimentAnalysis(Base):
    __tablename__ = "sentiment_analysis"

    id = Column(String, primary_key=True, index=True)
    topic_model_id = Column(
        String, ForeignKey("topic_model.id", ondelete="CASCADE"), nullable=False
    )
    method = Column(String, nullable=False)  # "vader", "bert", "textblob"

    overall_positive = Column(Float, nullable=False)
    overall_neutral = Column(Float, nullable=False)
    overall_negative = Column(Float, nullable=False)

    per_topic_json = Column(JSON, nullable=False)  # [{label, pos, neu, neg}, ...]

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
