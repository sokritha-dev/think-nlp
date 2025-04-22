from pydantic import BaseModel

from app.schemas.common import BaseResponse

class EDARequest(BaseModel):
    file_id: str

class EDAResponseData(BaseModel):
    file_id: str
    report_images: list[str]

class EDAResponse(BaseResponse):
    data: EDAResponseData
