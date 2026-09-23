import os
import uuid
import cv2
import time
from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.core.logging import logger

def save_image_bytes(image_bytes: bytes, folder: str = "reference_images", filename: Optional[str] = None) -> str:
    """Saves raw image bytes to secure local storage and returns relative path."""
    target_dir = Path(settings.STORAGE_DIR) / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"{uuid.uuid4().hex}.jpg"

    file_path = target_dir / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # Return relative path for DB reference
    return f"{folder}/{filename}"

def save_cv2_frame(frame: cv2.typing.MatLike, folder: str = "snapshots", filename: Optional[str] = None) -> str:
    """Saves OpenCV BGR image frame to storage."""
    target_dir = Path(settings.STORAGE_DIR) / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"snap_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"

    file_path = target_dir / filename
    cv2.imwrite(str(file_path), frame)

    return f"{folder}/{filename}"

def get_absolute_storage_path(relative_path: str) -> Optional[Path]:
    """Resolves relative path to absolute storage path securely."""
    if not relative_path:
        return None
    
    clean_rel = relative_path.lstrip("/\\")
    abs_path = (Path(settings.STORAGE_DIR) / clean_rel).resolve()
    
    # Path traversal protection
    storage_root = Path(settings.STORAGE_DIR).resolve()
    if not str(abs_path).startswith(str(storage_root)):
        logger.warning(f"Path traversal attempt blocked: {relative_path}")
        return None

    if abs_path.exists():
        return abs_path
    return None

def delete_storage_file(relative_path: str) -> bool:
    """Deletes storage file securely."""
    abs_path = get_absolute_storage_path(relative_path)
    if abs_path and abs_path.is_file():
        try:
            abs_path.unlink()
            logger.info(f"Deleted storage file: {relative_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {relative_path}: {e}")
    return False

def run_retention_cleanup():
    """Cleans up expired snapshots and unknown snapshots based on settings."""
    storage_root = Path(settings.STORAGE_DIR)
    
    # Cleanup unknown snapshots older than UNKNOWN_FACE_RETENTION_DAYS
    unk_dir = storage_root / "unknown_snapshots"
    if unk_dir.exists():
        cutoff_secs = time.time() - (settings.UNKNOWN_FACE_RETENTION_DAYS * 86400)
        for p in unk_dir.glob("*.jpg"):
            if p.stat().st_mtime < cutoff_secs:
                try:
                    p.unlink()
                except Exception:
                    pass

    # Cleanup attendance snapshots older than SNAPSHOT_RETENTION_DAYS
    snap_dir = storage_root / "snapshots"
    if snap_dir.exists():
        cutoff_secs = time.time() - (settings.SNAPSHOT_RETENTION_DAYS * 86400)
        for p in snap_dir.glob("*.jpg"):
            if p.stat().st_mtime < cutoff_secs:
                try:
                    p.unlink()
                except Exception:
                    pass
