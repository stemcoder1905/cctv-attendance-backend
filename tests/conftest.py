import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.database import Base
from app.models.domain import User, Person, AttendanceSettings, Camera
from app.core.security import hash_password

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def test_db():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSession = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with TestSession() as session:
        # Seed test admin user & settings
        admin = User(
            name="Test Admin",
            email="admin@test.com",
            password_hash=hash_password("admin123"),
            role="ADMIN",
            is_active=True
        )
        settings = AttendanceSettings(
            attendance_start_time="08:00",
            late_after_time="09:15",
            attendance_cutoff_time="17:00",
            timezone="Asia/Kolkata",
            is_active=True
        )
        cam1 = Camera(camera_name="Main Gate", location="Gate 1", stream_url="rtsp://user:pass123@192.168.1.100:554/stream", is_active=True)
        cam2 = Camera(camera_name="Reception", location="Building A", stream_url="rtsp://admin:secret456@192.168.1.101:554/live", is_active=True)
        
        session.add_all([admin, settings, cam1, cam2])
        await session.commit()

        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
