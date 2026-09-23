import asyncio
from typing import Dict, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.domain import Camera
from app.cctv.camera_worker import CameraWorker
from app.core.logging import logger

class CameraManager:
    """Multi-camera orchestrator managing active CameraWorker instances."""

    def __init__(self):
        self.workers: Dict[int, CameraWorker] = {}

    async def sync_cameras_from_db(self, db: AsyncSession):
        """Fetches active cameras from DB and starts missing workers or stops removed ones."""
        stmt = select(Camera).where(Camera.is_active == True)
        result = await db.execute(stmt)
        active_cameras = result.scalars().all()
        
        active_ids = {c.id for c in active_cameras}

        # Stop workers for cameras no longer active
        existing_ids = list(self.workers.keys())
        for c_id in existing_ids:
            if c_id not in active_ids:
                logger.info(f"Deactivating worker for camera ID {c_id}...")
                self.workers[c_id].stop()
                del self.workers[c_id]

        # Start workers for newly added active cameras
        for cam in active_cameras:
            if cam.id not in self.workers:
                logger.info(f"Starting worker for camera [{cam.camera_name}] (ID: {cam.id})...")
                worker = CameraWorker(cam.id, cam.camera_name, cam.stream_url)
                worker.start()
                self.workers[cam.id] = worker

    def get_worker(self, camera_id: int) -> Optional[CameraWorker]:
        return self.workers.get(camera_id)

    def stop_all(self):
        for c_id, worker in self.workers.items():
            worker.stop()
        self.workers.clear()
        logger.info("All camera workers stopped.")

camera_manager = CameraManager()
