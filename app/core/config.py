import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "AI CCTV Face Recognition Attendance System"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./attendance.db",
        description="Async SQLAlchemy database URL (PostgreSQL asyncpg or SQLite aiosqlite)"
    )

    # Auth & Security
    JWT_SECRET: str = "cctv_attendance_super_secret_jwt_key_2026_dev"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # Camera & AI Video Processing
    CAMERA_PROCESSING_FPS: float = 5.0
    FACE_MATCH_THRESHOLD: float = 0.55
    MIN_CONFIRMATION_FRAMES: int = 3
    DETECTION_INTERVAL: int = 1

    # Attendance Rules (HH:MM)
    ATTENDANCE_START_TIME: str = "08:00"
    LATE_AFTER_TIME: str = "09:15"
    ATTENDANCE_CUTOFF_TIME: str = "17:00"
    TIMEZONE: str = "Asia/Kolkata"

    # Data Retention (Days)
    UNKNOWN_FACE_RETENTION_DAYS: int = 7
    SNAPSHOT_RETENTION_DAYS: int = 30
    RECOGNITION_EVENT_RETENTION_DAYS: int = 30
    AUDIT_LOG_RETENTION_DAYS: int = 90

    # Storage Path
    STORAGE_DIR: str = str(BASE_DIR / "storage")

settings = Settings()

# Ensure storage subdirectories exist
for folder in ["reference_images", "snapshots", "unknown_snapshots"]:
    path = Path(settings.STORAGE_DIR) / folder
    path.mkdir(parents=True, exist_ok=True)
