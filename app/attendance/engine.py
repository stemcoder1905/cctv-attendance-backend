from datetime import datetime, date, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from sqlalchemy.exc import IntegrityError
from app.models.domain import Attendance, Person, AttendanceSettings, RecognitionEvent
from app.attendance.rules import evaluate_attendance_status
from app.core.logging import logger

IST_TZ = timezone(timedelta(hours=5, minutes=30))

async def get_active_attendance_settings(db: AsyncSession) -> Dict[str, Any]:
    """Fetches active attendance settings or defaults."""
    stmt = select(AttendanceSettings).where(AttendanceSettings.is_active == True).order_by(AttendanceSettings.id.desc())
    result = await db.execute(stmt)
    setting = result.scalars().first()
    if setting:
        return {
            "start_time": setting.attendance_start_time,
            "late_after_time": setting.late_after_time,
            "cutoff_time": setting.attendance_cutoff_time,
            "timezone": setting.timezone
        }
    return {
        "start_time": "08:00",
        "late_after_time": "09:15",
        "cutoff_time": "17:00",
        "timezone": "Asia/Kolkata"
    }

async def process_confirmed_recognition(
    db: AsyncSession,
    person_id: int,
    confidence: float,
    camera_id: Optional[int] = None,
    track_id: Optional[str] = None,
    snapshot_path: Optional[str] = None,
    now_dt: Optional[datetime] = None
) -> Tuple[bool, str, Optional[Attendance]]:
    """
    Processes a temporally confirmed person recognition.
    Ensures EXACTLY ONE attendance record per person per day using atomic DB logic in IST timezone.
    """
    if now_dt is None:
        now_dt = datetime.now(IST_TZ)
    
    today_date = now_dt.date()

    # 1. Check if attendance already exists for today
    stmt = select(Attendance).where(
        Attendance.person_id == person_id,
        Attendance.attendance_date == today_date
    )
    result = await db.execute(stmt)
    existing_record = result.scalars().first()

    if existing_record is not None:
        # Already marked today! Log event but ignore attendance creation.
        logger.info(f"Person ID {person_id} recognized again on {today_date} at {now_dt.strftime('%H:%M:%S')} (IST). Existing status: {existing_record.status}. Skipping attendance update.")
        
        # Log throttled recognition event
        await _log_recognition_event(db, person_id, camera_id, confidence, track_id, "KNOWN", snapshot_path)
        return False, f"Attendance already marked as {existing_record.status} today.", existing_record

    # 2. First appearance today! Determine status (PRESENT / LATE)
    settings_dict = await get_active_attendance_settings(db)
    status = evaluate_attendance_status(now_dt, late_after_time_str=settings_dict["late_after_time"])

    # 3. Create new Attendance record atomically
    try:
        new_attendance = Attendance(
            person_id=person_id,
            attendance_date=today_date,
            first_seen_time=now_dt,
            status=status,
            confidence=confidence,
            camera_id=camera_id,
            snapshot_path=snapshot_path
        )
        db.add(new_attendance)
        await db.commit()
        await db.refresh(new_attendance)

        logger.info(f"SUCCESS: Marked attendance for person_id {person_id} as {status} on {today_date} (IST) (Camera: {camera_id}).")
        
        # Log recognition event
        await _log_recognition_event(db, person_id, camera_id, confidence, track_id, "KNOWN", snapshot_path)
        
        return True, f"Attendance marked as {status}", new_attendance
    except IntegrityError:
        # Race condition safeguard: database UNIQUE(person_id, attendance_date) caught!
        await db.rollback()
        logger.warning(f"IntegrityError: Unique constraint prevented duplicate attendance for person_id {person_id} on {today_date}.")
        
        # Fetch existing record created concurrently
        result = await db.execute(stmt)
        record = result.scalars().first()
        return False, "Attendance already marked concurrently today.", record

async def process_cutoff_absent_records(db: AsyncSession, target_date: Optional[date] = None) -> List[Attendance]:
    """
    Idempotent end-of-day cutoff job using IST timezone.
    Finds all active registered persons who do NOT have an attendance record for target_date,
    and inserts ABSENT records for them.
    """
    if target_date is None:
        target_date = datetime.now(IST_TZ).date()

    now_dt = datetime.now(IST_TZ)

    # Fetch all active persons
    p_stmt = select(Person.id).where(Person.is_active == True)
    p_res = await db.execute(p_stmt)
    active_person_ids = set(p_res.scalars().all())

    # Fetch all person_ids with attendance today
    a_stmt = select(Attendance.person_id).where(Attendance.attendance_date == target_date)
    a_res = await db.execute(a_stmt)
    attended_person_ids = set(a_res.scalars().all())

    missing_person_ids = active_person_ids - attended_person_ids

    created_absent_records = []
    for p_id in missing_person_ids:
        try:
            absent_record = Attendance(
                person_id=p_id,
                attendance_date=target_date,
                first_seen_time=now_dt,
                status="ABSENT",
                confidence=None,
                camera_id=None,
                snapshot_path=None
            )
            db.add(absent_record)
            created_absent_records.append(absent_record)
        except Exception as e:
            logger.error(f"Failed to create ABSENT record for person_id {p_id}: {e}")

    if created_absent_records:
        try:
            await db.commit()
            logger.info(f"Cutoff Job: Generated {len(created_absent_records)} ABSENT attendance records for {target_date} (IST).")
        except Exception as e:
            await db.rollback()
            logger.error(f"Error committing ABSENT records: {e}")

    return created_absent_records

async def _log_recognition_event(
    db: AsyncSession,
    person_id: Optional[int],
    camera_id: Optional[int],
    confidence: float,
    track_id: Optional[str],
    event_type: str,
    snapshot_path: Optional[str]
):
    """Helper to insert recognition event."""
    try:
        event = RecognitionEvent(
            person_id=person_id,
            camera_id=camera_id,
            confidence=confidence,
            track_id=track_id,
            event_type=event_type,
            snapshot_path=snapshot_path
        )
        db.add(event)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to log recognition event: {e}")

import numpy as np
from app.models.domain import FaceEmbedding
from app.recognition.face_matcher import vector_matcher
from app.recognition.embedding_service import normalize_embedding, save_face_embedding

async def auto_register_and_mark_attendance(
    db: AsyncSession,
    probe_embedding: np.ndarray,
    confidence: float = 1.0,
    camera_id: Optional[int] = None,
    track_id: Optional[str] = None,
    face_crop_bgr: Optional[np.ndarray] = None
) -> Tuple[Person, Optional[Attendance]]:
    """
    Auto-registers an unregistered face as 'random_userX' in the Persons Directory.
    Saves face crop image and face embedding in DB, updates in-memory vector index.
    Does NOT mark attendance (PRESENT/LATE/ABSENT) for unregistered faces as per directive.
    """
    # 1. Determine next available random user number
    stmt = select(Person).where(Person.unique_person_id.like("RANDOM_USER_%"))
    result = await db.execute(stmt)
    auto_persons = result.scalars().all()
    
    num = len(auto_persons) + 1
    unique_person_id = f"RANDOM_USER_{num}"
    name = f"random_user{num}"
    
    # Safety check if unique_person_id already exists
    while True:
        check_stmt = select(Person).where(Person.unique_person_id == unique_person_id)
        res = await db.execute(check_stmt)
        if res.scalars().first() is None:
            break
        num += 1
        unique_person_id = f"RANDOM_USER_{num}"
        name = f"random_user{num}"

    # 2. Save face crop image if provided by camera worker
    rel_path = None
    if face_crop_bgr is not None and face_crop_bgr.size > 0:
        import cv2
        import os
        from app.core.config import settings as global_settings
        ref_filename = f"ref_person_{unique_person_id}.jpg"
        abs_path = os.path.join(global_settings.STORAGE_DIR, "reference_images", ref_filename)
        cv2.imwrite(abs_path, face_crop_bgr)
        rel_path = f"storage/reference_images/{ref_filename}"

    # 3. Create Person record in Persons Directory
    new_person = Person(
        unique_person_id=unique_person_id,
        name=name,
        department="Auto Registered",
        reference_image_path=rel_path,
        is_active=True
    )

    db.add(new_person)
    await db.flush()  # assign new_person.id

    # 4. Store FaceEmbedding
    norm_emb = normalize_embedding(probe_embedding).astype(np.float32)
    await save_face_embedding(db, new_person.id, norm_emb, quality_score=1.0)
    await db.refresh(new_person)

    logger.info(f"Auto-registered new unregistered face as '{name}' (ID: {new_person.id}, UniqueID: {unique_person_id}) in Persons Directory. Attendance skipped for unregistered user.")

    # 5. Update in-memory vector matcher so subsequent camera frames match this User
    vector_matcher.add_embedding(
        person_id=new_person.id,
        unique_person_id=new_person.unique_person_id,
        name=new_person.name,
        embedding=norm_emb
    )

    # Note: Attendance is NOT marked for unregistered faces (returns None)
    return new_person, None


