from datetime import datetime, date, timezone
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, Text, Date, DateTime, Time,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from app.core.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="VIEWER")  # ADMIN, TEACHER, STAFF, VIEWER
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    audit_logs = relationship("AuditLog", back_populates="user")


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    unique_person_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False, index=True)
    roll_number = Column(String(50), nullable=True, index=True)
    employee_number = Column(String(50), nullable=True, index=True)
    department = Column(String(100), nullable=True, index=True)
    email = Column(String(150), nullable=True)
    phone = Column(String(30), nullable=True)
    designation = Column(String(100), nullable=True)
    reference_image_path = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    face_embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")
    attendances = relationship("Attendance", back_populates="person", cascade="all, delete-orphan")
    recognition_events = relationship("RecognitionEvent", back_populates="person", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_json = Column(Text, nullable=False)  # JSON array of 512 floats
    embedding_model = Column(String(50), default="ArcFace", nullable=False)
    embedding_version = Column(String(50), default="buffalo_l_v1", nullable=False)
    quality_score = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    person = relationship("Person", back_populates="face_embeddings")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camera_name = Column(String(100), nullable=False, unique=True)
    location = Column(String(150), nullable=True)
    stream_url = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    status = Column(String(30), default="OFFLINE", nullable=False)  # ONLINE, OFFLINE, DISCONNECTED, ERROR
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    attendances = relationship("Attendance", back_populates="camera")
    recognition_events = relationship("RecognitionEvent", back_populates="camera")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    attendance_date = Column(Date, nullable=False, index=True)
    first_seen_time = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), nullable=False, index=True)  # PRESENT, LATE, ABSENT
    confidence = Column(Float, nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    snapshot_path = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    person = relationship("Person", back_populates="attendances")
    camera = relationship("Camera", back_populates="attendances")

    __table_args__ = (
        UniqueConstraint("person_id", "attendance_date", name="uq_person_daily_attendance"),
        Index("idx_attendance_date_person", "attendance_date", "person_id"),
    )


class AttendanceSettings(Base):
    __tablename__ = "attendance_settings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    attendance_start_time = Column(String(10), default="08:00", nullable=False)  # HH:MM
    late_after_time = Column(String(10), default="09:15", nullable=False)        # HH:MM
    attendance_cutoff_time = Column(String(10), default="17:00", nullable=False) # HH:MM
    timezone = Column(String(50), default="Asia/Kolkata", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=True, index=True) # None for UNKNOWN
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    track_id = Column(String(50), nullable=True)
    event_type = Column(String(30), default="RECOGNITION", nullable=False)  # KNOWN, UNKNOWN
    snapshot_path = Column(String(255), nullable=True)

    person = relationship("Person", back_populates="recognition_events")
    camera = relationship("Camera", back_populates="recognition_events")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    ip_address = Column(String(50), nullable=True)
    metadata_json = Column(Text, nullable=True)

    user = relationship("User", back_populates="audit_logs")
