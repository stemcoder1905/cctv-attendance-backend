import pytest
import numpy as np
from sqlalchemy import select
from app.models.domain import Person, Attendance, FaceEmbedding
from app.attendance.engine import auto_register_and_mark_attendance, process_confirmed_recognition
from app.recognition.face_matcher import vector_matcher
from app.recognition.embedding_service import normalize_embedding

@pytest.mark.asyncio
async def test_auto_register_unregistered_face(test_db):
    # 1. Generate synthetic 512D face embedding for an unregistered individual
    v_unregistered = normalize_embedding(np.random.randn(512).astype(np.float32))

    # 2. Trigger auto registration (attendance is NOT marked for unregistered faces per directive)
    person, attendance = await auto_register_and_mark_attendance(
        db=test_db,
        probe_embedding=v_unregistered,
        confidence=1.0,
        camera_id=1,
        track_id="Track_Auto_001"
    )

    assert person is not None
    assert person.name.startswith("random_user")
    assert person.unique_person_id.startswith("RANDOM_USER_")
    assert person.department == "Auto Registered"

    # Attendance is skipped for unregistered faces
    assert attendance is None

    # 3. Verify vector_matcher in-memory index now finds this auto user
    match = vector_matcher.find_best_match(v_unregistered, threshold=0.55)
    assert match is not None
    assert match["person_id"] == person.id
    assert match["name"] == person.name

    # 4. Verify process_confirmed_recognition marks attendance for registered person
    is_created, msg, att_rec = await process_confirmed_recognition(
        db=test_db,
        person_id=person.id,
        confidence=0.98,
        camera_id=1,
        track_id="Track_Auto_002"
    )
    assert is_created is True
    assert att_rec is not None
    assert att_rec.person_id == person.id
    assert att_rec.status in ["PRESENT", "LATE"]

    # 5. Verify same-day re-detection DOES NOT create a duplicate attendance record
    is_created2, msg2, existing_att = await process_confirmed_recognition(
        db=test_db,
        person_id=person.id,
        confidence=0.98,
        camera_id=1,
        track_id="Track_Auto_003"
    )
    assert is_created2 is False
    assert "already marked" in msg2
    assert existing_att.id == att_rec.id

    # Verify database count for this person's attendance today is EXACTLY 1
    stmt = select(Attendance).where(Attendance.person_id == person.id)
    res = await test_db.execute(stmt)
    records = res.scalars().all()
    assert len(records) == 1
