import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from fastapi import HTTPException, status
from app.models.domain import Person, FaceEmbedding
from app.schemas.pydantic_models import PersonCreate, PersonUpdate
from app.recognition.embedding_service import validate_registration_image, save_face_embedding, load_all_active_embeddings
from app.recognition.face_matcher import vector_matcher
from app.services.storage import save_image_bytes, delete_storage_file
from app.core.logging import logger

async def create_person(db: AsyncSession, person_in: PersonCreate) -> Person:
    """Creates a new registered person entry in the database."""
    # Check for existing unique_person_id
    stmt = select(Person).where(Person.unique_person_id == person_in.unique_person_id)
    res = await db.execute(stmt)
    if res.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Person with unique ID '{person_in.unique_person_id}' already exists."
        )

    person = Person(
        unique_person_id=person_in.unique_person_id,
        name=person_in.name,
        roll_number=person_in.roll_number,
        employee_number=person_in.employee_number,
        department=person_in.department,
        email=person_in.email,
        phone=person_in.phone,
        designation=person_in.designation,
        is_active=True
    )
    db.add(person)
    await db.commit()
    await db.refresh(person)
    return person

async def add_face_image_to_person(db: AsyncSession, person_id: int, image_bytes: bytes) -> Tuple[FaceEmbedding, str]:
    """
    Validates uploaded face photo, extracts ArcFace 512D embedding, stores embedding,
    saves reference photo, and refreshes the vector similarity index.
    """
    # 1. Fetch person
    stmt = select(Person).where(Person.id == person_id)
    res = await db.execute(stmt)
    person = res.scalars().first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    # 2. Decode image bytes to OpenCV BGR
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Validate image quality (Requirement 7)
    is_valid, reason, face_data = validate_registration_image(img_bgr)
    if not is_valid or face_data is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    # 4. Save image file to storage
    rel_path = save_image_bytes(image_bytes, folder="reference_images", filename=f"person_{person_id}_{person.unique_person_id}.jpg")
    person.reference_image_path = rel_path

    # 5. Store ArcFace 512D embedding
    embedding_record = await save_face_embedding(
        db=db,
        person_id=person_id,
        embedding=face_data["embedding"],
        quality_score=face_data["quality_score"]
    )

    await db.commit()
    await db.refresh(person)

    # 6. Refresh vector search index
    all_embeddings = await load_all_active_embeddings(db)
    vector_matcher.set_index(all_embeddings)

    return embedding_record, reason

async def delete_person_and_biometrics(db: AsyncSession, person_id: int) -> bool:
    """Deletes person and cleanly purges all associated face embeddings and storage files (Requirement 35)."""
    stmt = select(Person).where(Person.id == person_id)
    res = await db.execute(stmt)
    person = res.scalars().first()
    if not person:
        return False

    # Delete storage file
    if person.reference_image_path:
        delete_storage_file(person.reference_image_path)

    # Delete DB records
    await db.delete(person)
    await db.commit()

    # Instantly purge vectors from in-memory matcher without full DB reload
    vector_matcher.remove_person_embeddings(person_id)
    return True

async def update_person(db: AsyncSession, person_id: int, person_in: PersonUpdate) -> Person:
    """Updates person fields in DB and refreshes live vector matcher names & unique IDs."""
    stmt = select(Person).where(Person.id == person_id)
    res = await db.execute(stmt)
    person = res.scalars().first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    # Check for unique_person_id collisions if unique_person_id is changed
    if person_in.unique_person_id and person_in.unique_person_id != person.unique_person_id:
        check_stmt = select(Person).where(Person.unique_person_id == person_in.unique_person_id)
        check_res = await db.execute(check_stmt)
        if check_res.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unique ID '{person_in.unique_person_id}' is already assigned to another person."
            )
        person.unique_person_id = person_in.unique_person_id

    if person_in.name is not None:
        person.name = person_in.name
    if person_in.department is not None:
        person.department = person_in.department
    if person_in.roll_number is not None:
        person.roll_number = person_in.roll_number
    if person_in.employee_number is not None:
        person.employee_number = person_in.employee_number
    if person_in.email is not None:
        person.email = person_in.email
    if person_in.phone is not None:
        person.phone = person_in.phone
    if person_in.designation is not None:
        person.designation = person_in.designation
    if person_in.is_active is not None:
        person.is_active = person_in.is_active

    await db.commit()
    await db.refresh(person)

    # Update in-memory vector matcher metadata for live camera overlays
    vector_matcher.update_person_metadata(person.id, person.name, person.unique_person_id)

    return person
