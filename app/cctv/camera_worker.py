import asyncio
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from app.core.config import settings
from app.cctv.stream import CameraStreamReader
from app.recognition.insightface_engine import insightface_engine
from app.recognition.face_matcher import vector_matcher
from app.recognition.tracker import FaceTrackerManager
from app.attendance.engine import process_confirmed_recognition, auto_register_and_mark_attendance
from app.core.database import AsyncSessionLocal

from app.core.logging import logger

class CameraWorker:
    """Async worker processing camera frames through AI recognition & tracking pipeline."""

    def __init__(self, camera_id: int, camera_name: str, stream_url: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.stream_url = stream_url
        
        self.stream_reader = CameraStreamReader(camera_id, stream_url, camera_name)
        self.tracker = FaceTrackerManager()
        
        self._task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        
        # Latest annotated frame for WebSocket streaming
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_detections: List[Dict[str, Any]] = []

    def start(self):
        """Starts worker stream reader and processing task."""
        if not self.is_running:
            self.is_running = True
            self.stream_reader.start()
            self._task = asyncio.create_task(self._worker_loop())
            logger.info(f"CameraWorker [{self.camera_name}] (ID: {self.camera_id}) started.")

    def stop(self):
        """Stops worker task and reader."""
        self.is_running = False
        if self._task:
            self._task.cancel()
        self.stream_reader.stop()
        logger.info(f"CameraWorker [{self.camera_name}] stopped.")

    async def _worker_loop(self):
        target_fps = settings.CAMERA_PROCESSING_FPS
        frame_interval = 1.0 / max(1.0, target_fps)
        last_process_time = 0.0

        while self.is_running:
            now = time.time()
            if now - last_process_time < frame_interval:
                await asyncio.sleep(0.01)
                continue

            last_process_time = now
            has_frame, frame, _ = self.stream_reader.read_latest_frame()

            if not has_frame or frame is None:
                await asyncio.sleep(0.1)
                continue

            try:
                # 1. Run InsightFace detection & 512D ArcFace embedding extraction
                detected_faces = insightface_engine.detect_and_embed(frame)

                # 2. For each detected face, query vector similarity search
                threshold = settings.FACE_MATCH_THRESHOLD
                for face in detected_faces:
                    best_match = vector_matcher.find_best_match(face["embedding"], threshold=threshold)
                    face["candidate"] = best_match

                # 3. Update multi-object face tracker with temporal confirmation
                min_frames = settings.MIN_CONFIRMATION_FRAMES
                tracked_faces = self.tracker.update_tracks(detected_faces, min_confirmation_frames=min_frames)

                # 4. Check for newly confirmed identities and mark attendance
                for face in tracked_faces:
                    if face.get("is_confirmed"):
                        identity = face.get("confirmed_identity")
                        track_id = face.get("track_id")

                        if identity:
                            person_id = identity["person_id"]
                            confidence = identity["confidence"]

                            # Trigger async database attendance creation
                            async with AsyncSessionLocal() as db:
                                await process_confirmed_recognition(
                                    db=db,
                                    person_id=person_id,
                                    confidence=confidence,
                                    camera_id=self.camera_id,
                                    track_id=track_id,
                                    snapshot_path=None
                                )
                        else:
                            # Confirmed UNREGISTERED face! Auto-register as 'random_userX' & mark attendance
                            embedding = face.get("embedding")
                            if embedding is not None:
                                face_crop = None
                                bbox = face.get("bbox")
                                if bbox and frame is not None and frame.size > 0:
                                    h_f, w_f = frame.shape[:2]
                                    pad = 10
                                    x1, y1 = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
                                    x2, y2 = min(w_f, bbox[2] + pad), min(h_f, bbox[3] + pad)
                                    if x2 > x1 and y2 > y1:
                                        face_crop = frame[y1:y2, x1:x2].copy()

                                async with AsyncSessionLocal() as db:
                                    new_person, _ = await auto_register_and_mark_attendance(
                                        db=db,
                                        probe_embedding=embedding,
                                        confidence=1.0,
                                        camera_id=self.camera_id,
                                        track_id=track_id,
                                        face_crop_bgr=face_crop
                                    )
                                    confirmed_id_dict = {
                                        "person_id": new_person.id,
                                        "unique_person_id": new_person.unique_person_id,
                                        "name": new_person.name,
                                        "confidence": 1.0
                                    }
                                    face["confirmed_identity"] = confirmed_id_dict
                                    if track_id and track_id in self.tracker.tracks:
                                        self.tracker.tracks[track_id].confirmed_identity = confirmed_id_dict

                # 5. Annotate frame for live CCTV display
                annotated = self._annotate_frame(frame.copy(), tracked_faces)
                self.latest_annotated_frame = annotated
                self.latest_detections = tracked_faces

            except Exception as e:
                logger.error(f"Error in CameraWorker [{self.camera_name}]: {e}")

            await asyncio.sleep(0.01)

    def _annotate_frame(self, frame: np.ndarray, tracked_faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draws visual bounding boxes, labels, confidence, and attendance overlay on CCTV frame."""
        for face in tracked_faces:
            bbox = face["bbox"]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            track_id = face.get("track_id", "")
            is_confirmed = face.get("is_confirmed", False)
            identity = face.get("confirmed_identity")

            if is_confirmed and identity:
                name = identity["name"]
                conf_pct = identity["confidence"] * 100.0
                status_text = "PRESENT / LATE"
                color = (0, 255, 0)  # Green for confirmed recognized person
                label = f"{name} ({conf_pct:.1f}%) | {track_id}"
            else:
                candidate = face.get("candidate")
                if candidate:
                    name = candidate["name"]
                    conf_pct = candidate["similarity"] * 100.0
                    color = (0, 255, 255)  # Yellow for pending confirmation
                    label = f"Verifying {name} ({conf_pct:.1f}%)"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = f"UNKNOWN | {track_id}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label background box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame
