import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from app.models.domain import Person, Attendance
from app.attendance.engine import process_confirmed_recognition, process_cutoff_absent_records
from app.attendance.rules import evaluate_attendance_status

@pytest.mark.asyncio
async def test_present_and_late_rule_evaluation():
    # 08:55 is before late_after_time (09:15) -> PRESENT
    dt_present = datetime(2026, 8, 19, 8, 55, 0, tzinfo=timezone.utc)
    status_p = evaluate_attendance_status(dt_present, late_after_time_str="09:15")
    assert status_p == "PRESENT"

    # 09:25 is after late_after_time (09:15) -> LATE
    dt_late = datetime(2026, 8, 19, 9, 25, 0, tzinfo=timezone.utc)
    status_l = evaluate_attendance_status(dt_late, late_after_time_str="09:15")
    assert status_l == "LATE"

@pytest.mark.asyncio
async def test_first_appearance_present(test_db):
    # Register Deepak
    deepak = Person(unique_person_id="EMP001", name="Deepak Mishra", department="Engineering", is_active=True)
    test_db.add(deepak)
    await test_db.commit()
    await test_db.refresh(deepak)

    dt = datetime(2026, 8, 19, 8, 55, 0, tzinfo=timezone.utc)
    created, msg, att = await process_confirmed_recognition(
        db=test_db, person_id=deepak.id, confidence=0.94, camera_id=1, now_dt=dt
    )

    assert created is True
    assert att.status == "PRESENT"
    assert att.person_id == deepak.id

@pytest.mark.asyncio
async def test_first_appearance_late(test_db):
    # Register Rahul
    rahul = Person(unique_person_id="EMP002", name="Rahul Kumar", department="HR", is_active=True)
    test_db.add(rahul)
    await test_db.commit()
    await test_db.refresh(rahul)

    dt = datetime(2026, 8, 19, 9, 25, 0, tzinfo=timezone.utc)
    created, msg, att = await process_confirmed_recognition(
        db=test_db, person_id=rahul.id, confidence=0.91, camera_id=1, now_dt=dt
    )

    assert created is True
    assert att.status == "LATE"

@pytest.mark.asyncio
async def test_duplicate_appearances_same_day(test_db):
    # Register Deepak
    deepak = Person(unique_person_id="EMP003", name="Deepak Mishra", is_active=True)
    test_db.add(deepak)
    await test_db.commit()
    await test_db.refresh(deepak)

    dt_first = datetime(2026, 8, 19, 8, 55, 0, tzinfo=timezone.utc)
    # First detection -> Creates PRESENT row
    c1, m1, a1 = await process_confirmed_recognition(test_db, person_id=deepak.id, confidence=0.94, camera_id=1, now_dt=dt_first)
    assert c1 is True

    # 20 subsequent detections throughout the day
    for hour in range(9, 17):
        dt_subsequent = datetime(2026, 8, 19, hour, 15, 0, tzinfo=timezone.utc)
        c2, m2, a2 = await process_confirmed_recognition(test_db, person_id=deepak.id, confidence=0.96, camera_id=1, now_dt=dt_subsequent)
        assert c2 is False  # Must NOT insert duplicate row!

    # Verify database contains EXACTLY 1 attendance row for Deepak on this date
    stmt = select(Attendance).where(Attendance.person_id == deepak.id, Attendance.attendance_date == dt_first.date())
    res = await test_db.execute(stmt)
    records = res.scalars().all()
    assert len(records) == 1
    assert records[0].status == "PRESENT"

@pytest.mark.asyncio
async def test_multi_camera_deduplication(test_db):
    # Register Person
    person = Person(unique_person_id="EMP004", name="Sarah Connor", is_active=True)
    test_db.add(person)
    await test_db.commit()
    await test_db.refresh(person)

    dt = datetime(2026, 8, 19, 8, 50, 0, tzinfo=timezone.utc)
    # Camera 1 detection
    c1, _, _ = await process_confirmed_recognition(test_db, person_id=person.id, confidence=0.95, camera_id=1, now_dt=dt)
    assert c1 is True

    # Camera 2 detection later on same day
    dt_later = datetime(2026, 8, 19, 11, 30, 0, tzinfo=timezone.utc)
    c2, _, _ = await process_confirmed_recognition(test_db, person_id=person.id, confidence=0.98, camera_id=2, now_dt=dt_later)
    assert c2 is False

    # Check total rows
    stmt = select(Attendance).where(Attendance.person_id == person.id, Attendance.attendance_date == dt.date())
    res = await test_db.execute(stmt)
    records = res.scalars().all()
    assert len(records) == 1
    assert records[0].camera_id == 1  # Retains original camera reference

@pytest.mark.asyncio
async def test_cutoff_absent_generation(test_db):
    # Register 3 people: Person 1 (present), Person 2 (late), Person 3 (never detected)
    p1 = Person(unique_person_id="P1", name="Person 1", is_active=True)
    p2 = Person(unique_person_id="P2", name="Person 2", is_active=True)
    p3 = Person(unique_person_id="P3", name="Person 3", is_active=True)
    test_db.add_all([p1, p2, p3])
    await test_db.commit()

    dt = datetime(2026, 8, 19, 8, 45, 0, tzinfo=timezone.utc)
    await process_confirmed_recognition(test_db, person_id=p1.id, confidence=0.9, now_dt=dt)
    
    dt_late = datetime(2026, 8, 19, 9, 45, 0, tzinfo=timezone.utc)
    await process_confirmed_recognition(test_db, person_id=p2.id, confidence=0.9, now_dt=dt_late)

    # Run cutoff job for target date
    absent_records = await process_cutoff_absent_records(test_db, target_date=dt.date())

    assert len(absent_records) == 1
    assert absent_records[0].person_id == p3.id
    assert absent_records[0].status == "ABSENT"
