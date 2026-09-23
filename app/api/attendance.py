from datetime import date, datetime, timezone, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import get_current_user_claims, require_roles
from app.core.audit import create_audit_log
from app.schemas.pydantic_models import AttendanceResponse, AttendanceManualCreate
from app.services.attendance_service import get_filtered_attendance, export_attendance_csv, export_attendance_excel, export_attendance_pdf
from app.attendance.engine import process_cutoff_absent_records
from app.models.domain import Attendance

IST_TZ = timezone(timedelta(hours=5, minutes=30))

router = APIRouter(prefix="/attendance", tags=["Attendance Management"])

@router.get("/today", response_model=List[AttendanceResponse])
async def get_today_attendance(
    department: Optional[str] = None,
    status: Optional[str] = None,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Fetches today's attendance records in IST timezone."""
    today = datetime.now(IST_TZ).date()
    records = await get_filtered_attendance(db, start_date=today, end_date=today, department=department, status=status)
    return records

@router.get("/history", response_model=List[AttendanceResponse])
async def get_attendance_history(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    person_id: Optional[int] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
    camera_id: Optional[int] = None,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Fetches historical attendance records with comprehensive multi-field filtering."""
    records = await get_filtered_attendance(
        db, start_date=start_date, end_date=end_date, person_id=person_id,
        department=department, status=status, camera_id=camera_id
    )
    return records

@router.post("/trigger-cutoff", dependencies=[Depends(require_roles(["ADMIN", "TEACHER"]))])
async def trigger_cutoff_job(
    target_date: Optional[date] = None,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Triggers end-of-day cutoff job to auto-generate ABSENT records for missing active persons."""
    if not target_date:
        target_date = datetime.now(IST_TZ).date()

    absent_records = await process_cutoff_absent_records(db, target_date=target_date)
    
    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="TRIGGER_ABSENT_CUTOFF_JOB", resource_type="Attendance", metadata={"date": str(target_date), "count": len(absent_records)})

    return {
        "success": True,
        "target_date": str(target_date),
        "absent_records_created": len(absent_records)
    }

@router.get("/export")
async def export_attendance_report(
    format: str = Query("csv", regex="^(csv|excel|pdf)$"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Exports attendance report in CSV, Excel, or PDF format."""
    records = await get_filtered_attendance(
        db, start_date=start_date, end_date=end_date, department=department, status=status
    )

    if format == "csv":
        csv_data = export_attendance_csv(records)
        return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=attendance_report.csv"})
    elif format == "excel":
        excel_bytes = export_attendance_excel(records)
        return Response(content=excel_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=attendance_report.xlsx"})
    elif format == "pdf":
        pdf_bytes = export_attendance_pdf(records)
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=attendance_report.pdf"})
