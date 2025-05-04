from fastapi import HTTPException, status
from typing import Optional


class APIException(HTTPException):
    """Flexible API Exception."""

    def __init__(self, status_code: int, code: str, message: Optional[str] = None):
        detail = {"code": code, "message": message or "An error occurred"}
        super().__init__(status_code=status_code, detail=detail)


class NotFoundError(APIException):
    """404 Not Found Error."""

    def __init__(self, code: str, message: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, code=code, message=message
        )


class BadRequestError(APIException):
    """400 Bad Request Error."""

    def __init__(self, code: str, message: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, code=code, message=message
        )


class ServerError(APIException):
    """500 Internal Server Error."""

    def __init__(self, code: str, message: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=code,
            message=message,
        )
