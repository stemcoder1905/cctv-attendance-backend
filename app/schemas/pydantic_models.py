from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, ConfigDict

# Auth Schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "STAFF"  # ADMIN, TEACHER, STAFF, VIEWER

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime

# Person Schemas
class PersonCreate(BaseModel):
    unique_person_id: str
    name: str
    roll_number: Optional[str] = None
    employee_number: Optional[str] = None
    department: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    designation: Optional[str] = None

class PersonUpdate(BaseModel):
    name: Optional[str] = None
    unique_person_id: Optional[str] = None
    roll_number: Optional[str] = None
    employee_number: Optional[str] = None
    department: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    designation: Optional[str] = None
    is_active: Optional[bool] = None

class PersonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    unique_person_id: str
    name: str
    roll_number: Optional[str] = None
    employee_number: Optional[str] = None
    department: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    designation: Optional[str] = None
    reference_image_path: Optional[str] = None
    is_active: bool
    embedding_count: int = 0
    created_at: datetime
    updated_at: datetime

# Camera Schemas
class CameraCreate(BaseModel):
    camera_name: str
    location: Optional[str] = None
    stream_url: str
    is_active: bool = True

class CameraUpdate(BaseModel):
    camera_name: Optional[str] = None
    location: Optional[str] = None
    stream_url: Optional[str] = None
    is_active: Optional[bool] = None
    status: Optional[str] = None

class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_name: str
    location: Optional[str] = None
    stream_url: str  # Sanitized URL with masked credentials
    is_active: bool
    status: str
    created_at: datetime
    updated_at: datetime

# Attendance Schemas
from typing import List, Optional, Dict, Any, Union

class AttendanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    person_id: int
    person_name: Optional[str] = None
    unique_person_id: Optional[str] = None
    roll_number: Optional[str] = None
    employee_number: Optional[str] = None
    department: Optional[str] = None
    attendance_date: Union[date, str]
    first_seen_time: Union[datetime, str]
    status: str  # PRESENT, LATE, ABSENT
    confidence: Optional[float] = None

    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    snapshot_path: Optional[str] = None

class AttendanceManualCreate(BaseModel):
    person_id: int
    attendance_date: date
    status: str  # PRESENT, LATE, ABSENT
    notes: Optional[str] = None

# Attendance Settings Schemas
class AttendanceSettingsUpdate(BaseModel):
    attendance_start_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    late_after_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    attendance_cutoff_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    timezone: Optional[str] = None
    face_match_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confirmation_frames: Optional[int] = Field(None, ge=1, le=10)

class AttendanceSettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    attendance_start_time: str
    late_after_time: str
    attendance_cutoff_time: str
    timezone: str
    is_active: bool
    face_match_threshold: float
    min_confirmation_frames: int

# Dashboard Stats Schema
class DashboardStats(BaseModel):
    total_registered: int
    present_today: int
    late_today: int
    absent_today: int
    unknown_events_today: int
    active_cameras: int
    attendance_rate: float

# Audit Log Schema
class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    timestamp: datetime
    ip_address: Optional[str] = None
    metadata_json: Optional[str] = None
