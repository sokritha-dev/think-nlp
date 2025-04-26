from pydantic import BaseModel
from typing import List, Dict


class SentimentRequest(BaseModel):
    topic_model_id: str
    method: str  # 'vader', 'textblob', or 'bert'


class SentimentTopicBreakdown(BaseModel):
    label: str
    positive: float
    neutral: float
    negative: float


class SentimentChartData(BaseModel):
    overall: Dict[str, float]  # {'positive': %, 'neutral': %, 'negative': %}
    per_topic: List[SentimentTopicBreakdown]


class SentimentResponse(BaseModel):
    status: str
    message: str
    data: SentimentChartData
