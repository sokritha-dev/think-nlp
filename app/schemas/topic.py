from pydantic import BaseModel
from typing import List, Dict, Optional

from app.schemas.common import BaseResponse


class LDATopicRequest(BaseModel):
    file_id: str
    num_topics: Optional[int] = None


class LDATopicResponseData(BaseModel):
    file_id: str
    lda_topics_s3_url: str
    topics: List[Dict[str, str]]


class LDATopicResponse(BaseResponse):
    data: LDATopicResponseData


class TopicLabelRequest(BaseModel):
    topic_model_id: str
    label_map: Optional[Dict[int, str]] = None
    keywords: Optional[List[str]] = None


class TopicLabelTopic(BaseModel):
    topic_id: str
    keywords: str
    label: str
    confidence: float  # Change this if you need it to be rounded or string
    matched_with: Optional[str] = None  # Optional, only if you return this field


class TopicLabelResponseData(BaseModel):
    topic_model_id: str
    labeled_s3_url: str
    columns: List[str]
    record_count: int
    topics: List[TopicLabelTopic]


class TopicLabelResponse(BaseResponse):
    data: TopicLabelResponseData
