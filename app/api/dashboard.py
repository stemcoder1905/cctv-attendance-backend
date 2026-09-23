from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from app.core.database import get_db
from app.core.security import get_current_user_claims
from app.models.domain import Person, Attendance, Camera, RecognitionEvent
from app.schemas.pydantic_models import DashboardStats

router = APIRouter(prefix="/dashboard", tags=["Dashboard Statistics"])

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(claims: dict = Depends(get_current_user_claims), db: AsyncSession = Depends(get_db)):
    """Returns real-time KPI metrics for the Dashboard."""
    today = datetime.now(timezone.utc).date()

    # Total active registered persons
    tot_p_stmt = select(func.count(Person.id)).where(Person.is_active == True)
    tot_p_res = await db.execute(tot_p_stmt)
    total_registered = tot_p_res.scalar() or 0

    # Today's attendance status counts
    present_stmt = select(func.count(Attendance.id)).where(and_(Attendance.attendance_date == today, Attendance.status == "PRESENT"))
    present_res = await db.execute(present_stmt)
    present_today = present_res.scalar() or 0

    late_stmt = select(func.count(Attendance.id)).where(and_(Attendance.attendance_date == today, Attendance.status == "LATE"))
    late_res = await db.execute(late_stmt)
    late_today = late_res.scalar() or 0

    absent_stmt = select(func.count(Attendance.id)).where(and_(Attendance.attendance_date == today, Attendance.status == "ABSENT"))
    absent_res = await db.execute(absent_stmt)
    absent_today = absent_res.scalar() or 0

    # Today's unknown events
    unk_stmt = select(func.count(RecognitionEvent.id)).where(
        and_(
            func.date(RecognitionEvent.timestamp) == today,
            RecognitionEvent.event_type == "UNKNOWN"
        )
    )
    unk_res = await db.execute(unk_stmt)
    unknown_events_today = unk_res.scalar() or 0

    # Active cameras
    cam_stmt = select(func.count(Camera.id)).where(Camera.is_active == True)
    cam_res = await db.execute(cam_stmt)
    active_cameras = cam_res.scalar() or 0

    # Attendance rate
    attended_count = present_today + late_today
    attendance_rate = (attended_count / total_registered * 100.0) if total_registered > 0 else 0.0

    return DashboardStats(
        total_registered=total_registered,
        present_today=present_today,
        late_today=late_today,
        absent_today=absent_today,
        unknown_events_today=unknown_events_today,
        active_cameras=active_cameras,
        attendance_rate=round(attendance_rate, 1)
    )
