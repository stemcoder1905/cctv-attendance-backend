from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.config import settings as global_settings
from app.core.security import get_current_user_claims, require_roles
from app.core.audit import create_audit_log
from app.models.domain import AttendanceSettings
from app.schemas.pydantic_models import AttendanceSettingsUpdate, AttendanceSettingsResponse

router = APIRouter(prefix="/settings", tags=["System Settings"])

@router.get("", response_model=AttendanceSettingsResponse)
async def get_settings(claims: dict = Depends(get_current_user_claims), db: AsyncSession = Depends(get_db)):
    """Fetches system attendance timing policies and recognition thresholds."""
    stmt = select(AttendanceSettings).where(AttendanceSettings.is_active == True).order_by(AttendanceSettings.id.desc())
    res = await db.execute(stmt)
    setting = res.scalars().first()

    if not setting:
        return AttendanceSettingsResponse(
            id=1,
            attendance_start_time=global_settings.ATTENDANCE_START_TIME,
            late_after_time=global_settings.LATE_AFTER_TIME,
            attendance_cutoff_time=global_settings.ATTENDANCE_CUTOFF_TIME,
            timezone=global_settings.TIMEZONE,
            is_active=True,
            face_match_threshold=global_settings.FACE_MATCH_THRESHOLD,
            min_confirmation_frames=global_settings.MIN_CONFIRMATION_FRAMES
        )

    return AttendanceSettingsResponse(
        id=setting.id,
        attendance_start_time=setting.attendance_start_time,
        late_after_time=setting.late_after_time,
        attendance_cutoff_time=setting.attendance_cutoff_time,
        timezone=setting.timezone,
        is_active=setting.is_active,
        face_match_threshold=global_settings.FACE_MATCH_THRESHOLD,
        min_confirmation_frames=global_settings.MIN_CONFIRMATION_FRAMES
    )

@router.put("", response_model=AttendanceSettingsResponse, dependencies=[Depends(require_roles(["ADMIN"]))])
async def update_settings(
    settings_in: AttendanceSettingsUpdate,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Updates attendance timing rules and recognition parameters (ADMIN role required)."""
    stmt = select(AttendanceSettings).where(AttendanceSettings.is_active == True).order_by(AttendanceSettings.id.desc())
    res = await db.execute(stmt)
    setting = res.scalars().first()

    if not setting:
        setting = AttendanceSettings(
            attendance_start_time=global_settings.ATTENDANCE_START_TIME,
            late_after_time=global_settings.LATE_AFTER_TIME,
            attendance_cutoff_time=global_settings.ATTENDANCE_CUTOFF_TIME,
            timezone=global_settings.TIMEZONE,
            is_active=True
        )
        db.add(setting)

    if settings_in.attendance_start_time:
        setting.attendance_start_time = settings_in.attendance_start_time
    if settings_in.late_after_time:
        setting.late_after_time = settings_in.late_after_time
    if settings_in.attendance_cutoff_time:
        setting.attendance_cutoff_time = settings_in.attendance_cutoff_time
    if settings_in.timezone:
        setting.timezone = settings_in.timezone

    if settings_in.face_match_threshold is not None:
        global_settings.FACE_MATCH_THRESHOLD = settings_in.face_match_threshold
    if settings_in.min_confirmation_frames is not None:
        global_settings.MIN_CONFIRMATION_FRAMES = settings_in.min_confirmation_frames

    await db.commit()
    await db.refresh(setting)

    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="UPDATE_ATTENDANCE_SETTINGS", resource_type="Settings", resource_id=str(setting.id))

    return AttendanceSettingsResponse(
        id=setting.id,
        attendance_start_time=setting.attendance_start_time,
        late_after_time=setting.late_after_time,
        attendance_cutoff_time=setting.attendance_cutoff_time,
        timezone=setting.timezone,
        is_active=setting.is_active,
        face_match_threshold=global_settings.FACE_MATCH_THRESHOLD,
        min_confirmation_frames=global_settings.MIN_CONFIRMATION_FRAMES
    )
