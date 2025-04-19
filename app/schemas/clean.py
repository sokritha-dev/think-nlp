from typing import Dict, List, Optional
from pydantic import BaseModel
from app.schemas.common import BaseResponse


class NormalizeRequest(BaseModel):
    file_id: str
    broken_map: Optional[Dict[str, str]] = None


class NormalizeData(BaseModel):
    file_id: str
    normalized_s3_url: str
    record_count: int
    columns: List[str]


class NormalizeResponse(BaseResponse):
    data: NormalizeData


# Request & Response for /remove-special
class SpecialCleanRequest(BaseModel):
    file_id: str
    remove_special: bool = True
    remove_numbers: bool = True
    remove_emoji: bool = True


class SpecialCharCleanedData(BaseModel):
    file_id: str
    cleaned_s3_url: str
    record_count: int
    columns: list[str]
    removed_characters: list[str]


class SpecialCharCleanedResponse(BaseResponse):
    data: Optional[SpecialCharCleanedData] = None


# Response for /tokenize
class TokenizeRequest(BaseModel):
    file_id: str


class TokenizedData(BaseModel):
    file_id: str
    tokenized_s3_url: str
    record_count: int
    columns: list[str]


class TokenizedResponse(BaseResponse):
    data: TokenizedData


class StopwordTokenRequest(BaseModel):
    file_id: str
    custom_stopwords: Optional[List[str]] = []
    exclude_stopwords: Optional[List[str]] = []


class StopwordTokenResponseData(BaseModel):
    file_id: str
    tokenized_s3_url: str
    stopword_s3_url: str
    record_count: int
    columns: List[str]


class StopwordTokenResponse(BaseResponse):
    data: StopwordTokenResponseData


class LemmatizeRequest(BaseModel):
    file_id: str


class LemmatizedData(BaseModel):
    file_id: str
    tokenized_s3_url: str
    lemmatized_s3_url: str
    record_count: int
    columns: List[str]


class LemmatizedResponse(BaseResponse):
    data: LemmatizedData
