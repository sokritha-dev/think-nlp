from typing import List
from pydantic import BaseModel
from app.schemas.common import BaseResponse


class UploadData(BaseModel):
    file_id: str
    file_url: str
    s3_key: str
    columns: List[str]
    record_count: int


class UploadResponse(BaseResponse):
    data: UploadData
