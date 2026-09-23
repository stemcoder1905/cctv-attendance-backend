from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.database import get_db
from app.core.security import get_current_user_claims, require_roles
from app.core.audit import create_audit_log
from app.models.domain import Person, FaceEmbedding
from app.schemas.pydantic_models import PersonCreate, PersonUpdate, PersonResponse
from app.services.person_service import create_person, update_person, add_face_image_to_person, delete_person_and_biometrics

router = APIRouter(prefix="/persons", tags=["Persons & Biometrics"])

@router.get("", response_model=List[PersonResponse])
async def list_persons(
    department: Optional[str] = None,
    search: Optional[str] = None,
    is_active: Optional[bool] = True,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Lists registered persons. Raw biometric embeddings are strictly omitted from response schemas."""
    stmt = select(Person, func.count(FaceEmbedding.id).label("emb_count")).outerjoin(FaceEmbedding, Person.id == FaceEmbedding.person_id)
    
    conditions = []
    if is_active is not None:
        conditions.append(Person.is_active == is_active)
    if department:
        conditions.append(Person.department.ilike(f"%{department}%"))
    if search:
        conditions.append(
            (Person.name.ilike(f"%{search}%")) | 
            (Person.unique_person_id.ilike(f"%{search}%")) |
            (Person.roll_number.ilike(f"%{search}%")) |
            (Person.employee_number.ilike(f"%{search}%"))
        )

    if conditions:
        stmt = stmt.where(*conditions)

    stmt = stmt.group_by(Person.id).order_by(Person.name.asc())
    res = await db.execute(stmt)
    rows = res.all()

    person_responses = []
    for p, emb_c in rows:
        p_resp = PersonResponse.model_validate(p)
        p_resp.embedding_count = emb_c
        person_responses.append(p_resp)

    return person_responses

@router.post("", response_model=PersonResponse, dependencies=[Depends(require_roles(["ADMIN", "TEACHER", "STAFF"]))])
async def register_person(
    person_in: PersonCreate,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Registers a new student/employee metadata record."""
    person = await create_person(db, person_in)
    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="CREATE_PERSON", resource_type="Person", resource_id=str(person.id))
    return person

@router.post("/{person_id}/face", dependencies=[Depends(require_roles(["ADMIN", "TEACHER", "STAFF"]))])
async def upload_person_face(
    person_id: int,
    file: UploadFile = File(...),
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """
    Uploads a registration face photo. Validates single face, resolution, blurriness,
    generates 512D ArcFace embedding, stores embedding, and updates reference image.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    emb_rec, message = await add_face_image_to_person(db, person_id=person_id, image_bytes=contents)
    
    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="REGISTER_FACE_BIOMETRIC", resource_type="Person", resource_id=str(person_id))

    return {
        "success": True,
        "message": message,
        "embedding_id": emb_rec.id,
        "quality_score": emb_rec.quality_score
    }

@router.put("/{person_id}", response_model=PersonResponse, dependencies=[Depends(require_roles(["ADMIN", "TEACHER", "STAFF"]))])
async def edit_person(
    person_id: int,
    person_in: PersonUpdate,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Updates registered person details (Name, Unique ID, Department, etc.)."""
    person = await update_person(db, person_id=person_id, person_in=person_in)
    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="UPDATE_PERSON", resource_type="Person", resource_id=str(person_id))
    
    # Query embedding count for response model
    emb_res = await db.execute(select(func.count(FaceEmbedding.id)).where(FaceEmbedding.person_id == person_id))
    emb_count = emb_res.scalar() or 0
    
    resp = PersonResponse.model_validate(person)
    resp.embedding_count = emb_count
    return resp

@router.delete("/{person_id}", dependencies=[Depends(require_roles(["ADMIN"]))])
async def delete_person(
    person_id: int,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Deletes person and cleanly purges all biometric embeddings and storage files."""
    deleted = await delete_person_and_biometrics(db, person_id=person_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")
    
    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="DELETE_PERSON_BIOMETRICS", resource_type="Person", resource_id=str(person_id))

    return {"success": True, "detail": f"Person ID {person_id} and biometric data purged."}
