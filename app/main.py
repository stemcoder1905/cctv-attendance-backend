import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.database import engine, Base, AsyncSessionLocal
from app.api import auth, persons, attendance, cameras, dashboard, settings as settings_api, audit, websocket
from app.recognition.embedding_service import load_all_active_embeddings
from app.recognition.face_matcher import vector_matcher
from app.cctv.camera_manager import camera_manager
from app.attendance.scheduler import attendance_scheduler

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing AI CCTV Face Recognition Attendance System Backend...")

    # 1. Create database tables if not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized.")

    # 2. Seed default admin user & settings if table is empty
    async with AsyncSessionLocal() as db:
        from app.init_db import seed_initial_data
        await seed_initial_data(db)

    # 3. Load existing face embeddings into vector similarity matcher
    async with AsyncSessionLocal() as db:
        embeddings = await load_all_active_embeddings(db)
        vector_matcher.set_index(embeddings)

    # 4. Start active camera workers
    async with AsyncSessionLocal() as db:
        await camera_manager.sync_cameras_from_db(db)

    # 5. Start background scheduler
    attendance_scheduler.start()

    logger.info("Backend startup complete. System ready.")
    yield

    # Shutdown
    logger.info("Shutting down backend services...")
    attendance_scheduler.stop()
    camera_manager.stop_all()
    await engine.dispose()
    logger.info("Backend shutdown complete.")

app = FastAPI(
    title=settings.APP_NAME,
    description="Production AI-Powered CCTV Face Recognition Attendance System with InsightFace & ArcFace",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(persons.router, prefix="/api/v1")
app.include_router(attendance.router, prefix="/api/v1")
app.include_router(cameras.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(settings_api.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(websocket.router)

# Serve secure storage media files (with authentication protection in production)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")

from fastapi.responses import FileResponse

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.ENV
    }

@app.get("/download-thesis-pdf")
async def download_thesis_pdf():
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "THESIS_PRESENTATION.pdf")
    abs_pdf = os.path.abspath(pdf_path)
    if os.path.exists(abs_pdf):
        return FileResponse(abs_pdf, media_type="application/pdf", filename="THESIS_PRESENTATION.pdf")
    return {"error": "PDF file not found"}

