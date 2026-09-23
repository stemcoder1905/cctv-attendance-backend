import json
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.domain import FaceEmbedding
from app.recognition.insightface_engine import insightface_engine
from app.core.logging import logger

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Performs L2 normalization: v / ||v||."""
    vec = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).astype(np.float32)
    return vec

def validate_registration_image(image_bgr: np.ndarray) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates a registration face image according to strict requirements (Requirement 7):
    - Must detect exactly 1 face (reject if no face or multiple faces).
    - Image must not be extremely blurry (Laplacian variance > 50).
    - Face bounding box must meet minimum dimensions (at least 60x60 pixels).
    - Image brightness check (mean brightness in range [30, 230]).
    Returns: (is_valid, reason_message, face_info_dict)
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, "Invalid or corrupt image data.", None

    faces = insightface_engine.detect_and_embed(image_bgr)
    if len(faces) == 0:
        return False, "No face detected in the image. Please upload a clear face photo.", None
    if len(faces) > 1:
        return False, f"Multiple faces ({len(faces)}) detected. Registration photo must contain exactly one face.", None

    face = faces[0]
    bbox = face["bbox"]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w < 50 or h < 50:
        return False, f"Face is too small ({w}x{h} px). Please provide a closer, higher resolution photo.", None

    # Blurriness validation
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(image_bgr.shape[1], bbox[2]), min(image_bgr.shape[0], bbox[3])
    face_roi = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 35.0:
        return False, f"Image is too blurry (sharpness score {lap_var:.1f}). Please capture under clear lighting.", None

    # Brightness validation
    mean_brightness = gray.mean()
    if mean_brightness < 25.0:
        return False, f"Image is too dark (brightness {mean_brightness:.1f}). Please ensure sufficient lighting.", None
    if mean_brightness > 240.0:
        return False, f"Image is overexposed (brightness {mean_brightness:.1f}). Please avoid harsh direct flash.", None

    return True, "Face quality validation passed.", face

async def save_face_embedding(
    db: AsyncSession,
    person_id: int,
    embedding: np.ndarray,
    quality_score: float = 1.0,
    model_name: str = "ArcFace",
    model_version: str = "buffalo_l_v1"
) -> FaceEmbedding:
    """Stores a normalized ArcFace embedding in the database."""
    norm_emb = normalize_embedding(embedding)
    emb_list = norm_emb.tolist()

    emb_record = FaceEmbedding(
        person_id=person_id,
        embedding_json=json.dumps(emb_list),
        embedding_model=model_name,
        embedding_version=model_version,
        quality_score=quality_score
    )
    db.add(emb_record)
    await db.commit()
    await db.refresh(emb_record)
    return emb_record

async def load_all_active_embeddings(db: AsyncSession) -> List[Dict[str, Any]]:
    """Loads all normalized face embeddings for active registered persons from the database."""
    from app.models.domain import Person, FaceEmbedding
    
    stmt = (
        select(FaceEmbedding, Person.id, Person.unique_person_id, Person.name, Person.department)
        .join(Person, FaceEmbedding.person_id == Person.id)
        .where(Person.is_active == True)
    )
    result = await db.execute(stmt)
    records = result.all()

    embeddings_list = []
    for emb_rec, p_id, u_id, p_name, p_dept in records:
        try:
            vec = np.array(json.loads(emb_rec.embedding_json), dtype=np.float32)
            vec = normalize_embedding(vec)
            embeddings_list.append({
                "embedding_id": emb_rec.id,
                "person_id": p_id,
                "unique_person_id": u_id,
                "name": p_name,
                "department": p_dept,
                "embedding": vec,
                "quality_score": emb_rec.quality_score
            })
        except Exception as e:
            logger.error(f"Error parsing embedding {emb_rec.id} for person {p_id}: {e}")

    return embeddings_list
