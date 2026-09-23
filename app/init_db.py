from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.domain import User, AttendanceSettings, Camera
from app.core.security import hash_password
from app.core.logging import logger

async def seed_initial_data(db: AsyncSession):
    """Seeds initial admin user, attendance timing policies, and default webcam camera if empty."""
    # 1. Seed Admin User
    user_stmt = select(User).where(User.email == "admin@system.com")
    res = await db.execute(user_stmt)
    if not res.scalars().first():
        admin_user = User(
            name="System Administrator",
            email="admin@system.com",
            password_hash=hash_password("admin123"),
            role="ADMIN",
            is_active=True
        )
        db.add(admin_user)
        logger.info("Seeded default admin user: admin@system.com / admin123")

    # 2. Seed Default Attendance Settings
    settings_stmt = select(AttendanceSettings)
    res_s = await db.execute(settings_stmt)
    if not res_s.scalars().first():
        default_settings = AttendanceSettings(
            attendance_start_time="08:00",
            late_after_time="09:15",
            attendance_cutoff_time="17:00",
            timezone="Asia/Kolkata",
            is_active=True
        )
        db.add(default_settings)
        logger.info("Seeded default attendance timing settings (08:00 - 09:15 - 17:00)")

    # 3. Seed Default Webcam Camera for development
    cam_stmt = select(Camera).where(Camera.camera_name == "Webcam 0")
    res_c = await db.execute(cam_stmt)
    if not res_c.scalars().first():
        webcam = Camera(
            camera_name="Webcam 0",
            location="Development Local Stream",
            stream_url="0",
            is_active=True,
            status="OFFLINE"
        )
        db.add(webcam)
        logger.info("Seeded default Camera 0 (webcam / local test stream)")

    await db.commit()
