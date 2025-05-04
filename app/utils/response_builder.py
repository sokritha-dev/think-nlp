from fastapi.responses import JSONResponse
from typing import Any, Optional
from pydantic import BaseModel
import datetime


def serialize_data(data: Any):
    """Helper to convert Pydantic models and datetime objects cleanly."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [serialize_data(item) for item in data]
    if isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    if isinstance(data, datetime.datetime):
        return data.isoformat()
    return data


def success_response(
    message: str, data: Optional[Any] = None, status_code: int = 200
) -> JSONResponse:
    serialized_data = serialize_data(data)

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "success",
            "message": message,
            "data": serialized_data,
        },
    )
