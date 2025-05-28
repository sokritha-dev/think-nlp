from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.messages.clean_messages import FILE_NOT_FOUND
from app.models.db.file_record import FileRecord
from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError

router = APIRouter(prefix="/api/file", tags=["File"])


@router.get("/status")
async def check_is_sample(
    file_id: str = Query(...), db: AsyncSession = Depends(get_db)
):
    record = (
        await db.execute(select(FileRecord).filter_by(id=file_id))
    ).scalar_one_or_none()
    if not record:
        raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

    return success_response(
        message="File sample status retrieved.",
        data={"file_id": file_id, "is_sample": record.is_sample},
    )
