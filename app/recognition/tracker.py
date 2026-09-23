import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)

class TrackObject:
    def __init__(self, track_id: str, bbox: List[int]):
        self.track_id = track_id
        self.bbox = bbox
        self.last_updated = time.time()
        self.missed_frames = 0
        
        # Recognition state
        self.confirmed_identity: Optional[Dict[str, Any]] = None  # {person_id, name, unique_person_id}
        self.confirmation_history: List[Dict[str, Any]] = []      # [{person_id, similarity}]
        self.is_confirmed: bool = False
        self.attendance_marked: bool = False

    def add_match_candidate(self, candidate: Optional[Dict[str, Any]], min_confirmation_frames: int = 3):
        """Accumulates matching candidates across consecutive frames for temporal confirmation."""
        if self.is_confirmed:
            return

        if candidate is None:
            # Low confidence / unknown frame
            self.confirmation_history.append({"person_id": None, "similarity": 0.0})
        else:
            self.confirmation_history.append(candidate)

        # Keep history window
        if len(self.confirmation_history) > 10:
            self.confirmation_history.pop(0)

        # Check if last min_confirmation_frames contain consistent person_id
        recent = self.confirmation_history[-min_confirmation_frames:]
        if len(recent) >= min_confirmation_frames:
            first_person = recent[0].get("person_id")
            if first_person is not None and all(item.get("person_id") == first_person for item in recent):
                # Identity confirmed after N consistent temporal matches!
                avg_sim = float(np.mean([item["similarity"] for item in recent]))
                self.confirmed_identity = {
                    "person_id": first_person,
                    "unique_person_id": recent[0]["unique_person_id"],
                    "name": recent[0]["name"],
                    "confidence": round(avg_sim, 4)
                }
                self.is_confirmed = True
            elif first_person is None and all(item.get("person_id") is None for item in recent):
                # Unregistered face consistently tracked for N frames -> Confirm for auto-registration!
                self.confirmed_identity = None
                self.is_confirmed = True


class FaceTrackerManager:
    """Multi-face tracker assigning persistent Track IDs and performing temporal identity confirmation."""

    def __init__(self, max_missed_frames: int = 15, iou_threshold: float = 0.3):
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.tracks: Dict[str, TrackObject] = {}
        self.next_track_number = 1

    def update_tracks(self, detected_faces: List[Dict[str, Any]], min_confirmation_frames: int = 3) -> List[Dict[str, Any]]:
        """
        Updates active tracks with newly detected bounding boxes.
        Returns face dicts annotated with track_id and confirmed_identity.
        """
        updated_results = []
        unmatched_detections = list(range(len(detected_faces)))
        existing_track_ids = list(self.tracks.keys())

        # Match existing tracks with detected bboxes via IoU
        matched_pairs = []
        if existing_track_ids and detected_faces:
            iou_matrix = np.zeros((len(existing_track_ids), len(detected_faces)))
            for i, t_id in enumerate(existing_track_ids):
                for j, face in enumerate(detected_faces):
                    iou_matrix[i, j] = compute_iou(self.tracks[t_id].bbox, face["bbox"])

            # Greedy assignment based on max IoU
            for _ in range(min(len(existing_track_ids), len(detected_faces))):
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                if max_iou >= self.iou_threshold:
                    t_id = existing_track_ids[max_idx[0]]
                    d_idx = max_idx[1]
                    matched_pairs.append((t_id, d_idx))
                    iou_matrix[max_idx[0], :] = -1
                    iou_matrix[:, max_idx[1]] = -1

        matched_det_indices = set()
        matched_track_ids = set()

        for t_id, d_idx in matched_pairs:
            matched_track_ids.add(t_id)
            matched_det_indices.add(d_idx)
            track = self.tracks[t_id]
            face = detected_faces[d_idx]
            
            track.bbox = face["bbox"]
            track.missed_frames = 0
            track.last_updated = time.time()

            # Process identity recognition candidate
            candidate = face.get("candidate")
            track.add_match_candidate(candidate, min_confirmation_frames=min_confirmation_frames)

            annotated_face = face.copy()
            annotated_face["track_id"] = track.track_id
            annotated_face["confirmed_identity"] = track.confirmed_identity
            annotated_face["is_confirmed"] = track.is_confirmed
            updated_results.append(annotated_face)

        # Create new tracks for unmatched detections
        for j, face in enumerate(detected_faces):
            if j not in matched_det_indices:
                track_id = f"Track_{self.next_track_number:03d}"
                self.next_track_number += 1
                
                track = TrackObject(track_id=track_id, bbox=face["bbox"])
                candidate = face.get("candidate")
                track.add_match_candidate(candidate, min_confirmation_frames=min_confirmation_frames)

                self.tracks[track_id] = track
                
                annotated_face = face.copy()
                annotated_face["track_id"] = track_id
                annotated_face["confirmed_identity"] = track.confirmed_identity
                annotated_face["is_confirmed"] = track.is_confirmed
                updated_results.append(annotated_face)

        # Increment missed frames for unmatched active tracks and purge dead tracks
        dead_tracks = []
        for t_id, track in self.tracks.items():
            if t_id not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    dead_tracks.append(t_id)

        for t_id in dead_tracks:
            del self.tracks[t_id]

        return updated_results
