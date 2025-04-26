from pydantic import BaseModel
from app.schemas.sentiment import SentimentChartData


class FullPipelineRequest(BaseModel):
    file_id: str


class FullPipelineResponse(BaseModel):
    status: str
    message: str
    sentiment: SentimentChartData
