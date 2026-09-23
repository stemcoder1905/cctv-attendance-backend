# Production AI CCTV Face Recognition Attendance System

An enterprise-grade, privacy-first automated attendance management system built from scratch using **InsightFace (buffalo_l)**, **ArcFace 512-dimensional normalized embeddings**, **FastAPI**, **PostgreSQL** (with `pgvector` or in-memory vector index), and a real-time **React dashboard** with WebSocket live video overlays.

---

## 🌟 Core System Highlights

- **Automated Continuous CCTV Recognition**: Recognizes registered individuals in real-time without requiring manual button presses or user interaction.
- **Biometric Core**: Powered strictly by **InsightFace** and **ArcFace** normalized 512D embeddings. No reliance on legacy dlib or `face_recognition` packages.
- **Temporal Match Confirmation**: Requires $N$ consecutive consistent matches across frames before confirming identity to eliminate single-frame false positives.
- **One Attendance Per Person Per Day**: Database-level unique constraint (`UNIQUE(person_id, attendance_date)`) and race-condition-proof atomic insertions.
- **PRESENT / LATE / ABSENT Business Logic**:
  - First-seen time $\le$ `late_after_time` (e.g. 09:15) $\rightarrow$ **PRESENT**
  - First-seen time $>$ `late_after_time` $\rightarrow$ **LATE**
  - End-of-day cutoff job $\rightarrow$ Idempotent auto-generation of **ABSENT** records for missing active individuals.
- **Privacy-First Security Architecture**:
  - Argon2 password hashing (`argon2-cffi`).
  - Raw 512D face embeddings are **never** exposed over API responses.
  - Camera stream credentials (passwords in RTSP URLs) are automatically masked (`rtsp://user:****@host:port`).
  - Private local storage for reference photos and attendance snapshots.
  - Data retention policies for cleanup of expired snapshots and logs.
- **Multi-Camera Engine**: Independent concurrent stream processing workers for Camera 1 (Main Gate), Camera 2 (Reception), etc., feeding into the unified recognition pipeline.
- **Real-Time React Dashboard**: Live CCTV WebSocket feed with bounding box, name, confidence %, and attendance status overlays. Export reports to CSV, Excel, and PDF.

---

## 📐 System Pipeline Architecture

```
CCTV / Webcam / RTSP Stream
           │
           ▼
CameraStreamReader (Non-blocking Thread & Reconnect Backoff)
           │
           ▼
CameraWorker (Frame Sampler: 5-10 Processing FPS)
           │
           ▼
InsightFace Engine (buffalo_l Detection & 5 Landmark Alignment)
           │
           ▼
ArcFace Model (512-Dimensional ArcFace Vector Extraction)
           │
           ▼
L2 Normalization (||v|| = 1.0)
           │
           ▼
VectorSimilarityMatcher (Cosine Similarity Matrix Search)
           │
           ▼
MultiFaceTracker (IoU + Centroid Track ID Assignment)
           │
           ▼
TemporalMatchConfirmator (MIN_CONFIRMATION_FRAMES verification)
           │
           ▼
Attendance Rules Engine (Checks UNIQUE(person_id, attendance_date))
           │
 ┌─────────┴────────────────────────┐
 ▼                                  ▼
First Appearance Today       Subsequent Appearance Today
 ├── PRESENT (<= 09:15)             └── Ignored (No DB update)
 └── LATE (> 09:15)
           │
           ▼
Cutoff Job (17:00) -> Auto-generate ABSENT for missing active persons
```

---

## 🛠️ Repository Directory Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application entrypoint
│   │   ├── api/                        # REST & WebSocket API Routers
│   │   │   ├── auth.py                 # Login, JWT, User accounts
│   │   │   ├── persons.py              # Person registration & photo quality check
│   │   │   ├── attendance.py           # Today, History, Export, Cutoff trigger
│   │   │   ├── cameras.py              # Camera setup & RTSP connection test
│   │   │   ├── dashboard.py            # Summary stats & metrics
│   │   │   ├── settings.py             # Timing & threshold configs
│   │   │   ├── audit.py                # Security audit log view
│   │   │   └── websocket.py            # Live video WebSocket stream
│   │   ├── core/                       # Configuration, Security, DB, Audit
│   │   ├── models/                     # SQLAlchemy domain models
│   │   ├── schemas/                    # Pydantic request/response schemas
│   │   ├── recognition/                # InsightFace, ArcFace, Vector search & Tracker
│   │   ├── cctv/                       # OpenCV Stream Reader & Camera Worker
│   │   ├── attendance/                 # Business logic, Rules & Cutoff Scheduler
│   │   └── services/                   # Storage, Person Registration & Export Services
│   ├── tests/                          # Automated Pytest suite
│   ├── init_db.py                      # Database seeding script
│   └── requirements.txt                # Backend dependencies
├── frontend/                           # React Dashboard (Vite + Tailwind CSS)
│   ├── src/
│   │   ├── components/                 # Navbar, Sidebar, UI elements
│   │   ├── pages/                      # Dashboard, Live CCTV, Persons, Reports, etc.
│   │   ├── services/                   # Axios API client
│   │   ├── context/                    # Auth Context
│   │   ├── App.jsx                     # Router & Protected Guards
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── storage/                            # Secure local media storage
├── .env.example                        # Safe environment template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Setup & Installation Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- OpenCV & C++ build tools (for InsightFace / ONNX Runtime)

### 1. Backend Setup

```bash
# Navigate to project root
cd d:\cctv_attendance_management_system

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
source venv/bin/activate      # On Linux/macOS

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# Create local environment configuration
cp .env.example .env

# Initialize database schema and seed default admin user
python -c "import asyncio; from app.core.database import engine, Base; from app.init_db import seed_initial_data; from app.core.database import AsyncSessionLocal; async def init(): async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all); async with AsyncSessionLocal() as db: await seed_initial_data(db); asyncio.run(init())"
```

Default seeded credentials:
- **Email**: `admin@system.com`
- **Password**: `admin123`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite React development server
npm run dev
```

The React dashboard will be available at `http://localhost:3000`.

### 3. Running Backend Application

```bash
# Start FastAPI application with Uvicorn
cd d:\cctv_attendance_management_system
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation available at `http://localhost:8000/docs`.

---

## 🧪 Running Automated Test Suite

Run the full automated pytest suite covering attendance rules, duplicate protection, multi-camera deduplication, cutoff ABSENT generation, vector similarity search, and security controls:

```bash
.\venv\Scripts\python.exe -m pytest backend/tests -v
```

---

## 📷 CCTV & Camera Configuration

- **Webcam / Development Mode**: Set `stream_url` to `0` in Camera setup.
- **RTSP IP Camera / NVR**: Set `stream_url` to `rtsp://username:password@camera-ip:554/stream`.
- **RTSP Passwords**: Passwords in RTSP URLs are sanitized in API outputs (`rtsp://username:****@camera-ip:554/stream`).

---

## 🔒 Security & Biometric Privacy Policy

1. **No Embedding Leakage**: Raw ArcFace 512D embeddings are excluded from all public Pydantic DTO responses.
2. **Argon2 Password Security**: Admin and user passwords are automatically hashed with Argon2.
3. **Private Biometric Media**: Reference face photos and attendance snapshots are stored in non-public storage directories.
4. **Data Retention Jobs**: Automated cleanup of unknown snapshot files after $X$ days (`UNKNOWN_FACE_RETENTION_DAYS`).
5. **Security Audit Logs**: All administrative actions (user login, person registration, face photo upload, camera edits) are recorded in the `audit_logs` table.
