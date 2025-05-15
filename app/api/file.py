from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.messages.clean_messages import FILE_NOT_FOUND
from app.models.db.file_record import FileRecord
from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError

router = APIRouter(prefix="/api/file", tags=["File"])


@router.get("/status")
def check_is_sample(file_id: str = Query(...), db: Session = Depends(get_db)):
    record = db.query(FileRecord).filter_by(id=file_id).first()
    if not record:
        raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

    return success_response(
        message="File sample status retrieved.",
        data={"file_id": file_id, "is_sample": record.is_sample},
    )
