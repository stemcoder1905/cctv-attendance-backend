from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import get_current_user_claims, require_roles, sanitize_camera_url
from app.core.audit import create_audit_log
from app.models.domain import Camera
from app.schemas.pydantic_models import CameraCreate, CameraUpdate, CameraResponse
from app.cctv.stream import CameraStreamReader
from app.cctv.camera_manager import camera_manager

router = APIRouter(prefix="/cameras", tags=["Camera Management"])

@router.get("", response_model=List[CameraResponse])
async def list_cameras(claims: dict = Depends(get_current_user_claims), db: AsyncSession = Depends(get_db)):
    """Lists registered CCTV/webcam instances. Camera credentials in RTSP URLs are masked!"""
    stmt = select(Camera).order_by(Camera.id.asc())
    res = await db.execute(stmt)
    cameras = res.scalars().all()

    camera_responses = []
    for c in cameras:
        worker = camera_manager.get_worker(c.id)
        current_status = worker.stream_reader.status if worker else c.status
        
        c_resp = CameraResponse(
            id=c.id,
            camera_name=c.camera_name,
            location=c.location,
            stream_url=sanitize_camera_url(c.stream_url),
            is_active=c.is_active,
            status=current_status,
            created_at=c.created_at,
            updated_at=c.updated_at
        )
        camera_responses.append(c_resp)

    return camera_responses

@router.post("", response_model=CameraResponse, dependencies=[Depends(require_roles(["ADMIN"]))])
async def add_camera(
    camera_in: CameraCreate,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Adds a new CCTV / RTSP / webcam instance (ADMIN role required)."""
    stmt = select(Camera).where(Camera.camera_name == camera_in.camera_name)
    res = await db.execute(stmt)
    if res.scalars().first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Camera name '{camera_in.camera_name}' already exists.")

    new_camera = Camera(
        camera_name=camera_in.camera_name,
        location=camera_in.location,
        stream_url=camera_in.stream_url,
        is_active=camera_in.is_active,
        status="OFFLINE"
    )
    db.add(new_camera)
    await db.commit()
    await db.refresh(new_camera)

    # Sync camera manager workers
    await camera_manager.sync_cameras_from_db(db)

    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="ADD_CAMERA", resource_type="Camera", resource_id=str(new_camera.id))

    return CameraResponse(
        id=new_camera.id,
        camera_name=new_camera.camera_name,
        location=new_camera.location,
        stream_url=sanitize_camera_url(new_camera.stream_url),
        is_active=new_camera.is_active,
        status=new_camera.status,
        created_at=new_camera.created_at,
        updated_at=new_camera.updated_at
    )

@router.delete("/{camera_id}", dependencies=[Depends(require_roles(["ADMIN"]))])
async def delete_camera(
    camera_id: int,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Deletes a camera instance."""
    stmt = select(Camera).where(Camera.id == camera_id)
    res = await db.execute(stmt)
    camera = res.scalars().first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")

    await db.delete(camera)
    await db.commit()

    # Sync camera manager workers
    await camera_manager.sync_cameras_from_db(db)

    user_id = int(claims["sub"])
    await create_audit_log(db, user_id=user_id, action="DELETE_CAMERA", resource_type="Camera", resource_id=str(camera_id))

    return {"success": True, "detail": f"Camera ID {camera_id} deleted."}

@router.post("/{camera_id}/test")
async def test_camera_connection(camera_id: int, db: AsyncSession = Depends(get_db)):
    """Tests connection to RTSP stream or webcam without starting permanent worker."""
    stmt = select(Camera).where(Camera.id == camera_id)
    res = await db.execute(stmt)
    camera = res.scalars().first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")

    reader = CameraStreamReader(camera_id=camera.id, stream_url=camera.stream_url, camera_name=camera.camera_name)
    success = reader._open_stream()
    status_res = reader.status
    reader._release_cap()

    return {
        "camera_id": camera.id,
        "camera_name": camera.camera_name,
        "connection_successful": success,
        "status": status_res
    }
