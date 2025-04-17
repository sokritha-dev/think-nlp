from typing import List, Optional
from pydantic import BaseModel
from app.schemas.common import BaseResponse


class NormalizeRequest(BaseModel):
    text: str


class NormalizeData(BaseModel):
    original: str
    normalized: str


class NormalizeResponse(BaseResponse):
    data: NormalizeData


# Response for /remove-special
class SpecialCharCleanedData(BaseModel):
    original: str
    cleaned: str


class SpecialCharCleanedResponse(BaseResponse):
    data: SpecialCharCleanedData


# Response for /tokenize
class TokenizedData(BaseModel):
    original: str
    tokens: List[str]


class TokenizedResponse(BaseResponse):
    data: TokenizedData


class StopwordTokenRequest(BaseModel):
    tokens: List[str]
    custom_stopwords: Optional[List[str]] = []
    exclude_stopwords: Optional[List[str]] = []


class StopwordTokenResponseData(BaseModel):
    original_tokens: List[str]
    cleaned_tokens: List[str]
    removed_stopwords: List[str]


class StopwordTokenResponse(BaseResponse):
    data: StopwordTokenResponseData


class LemmatizeRequest(BaseModel):
    tokens: List[str]


class LemmatizedData(BaseModel):
    original_tokens: List[str]
    lemmatized_tokens: List[str]
    changes: List[tuple[str, str]]


class LemmatizedResponse(BaseResponse):
    data: LemmatizedData
