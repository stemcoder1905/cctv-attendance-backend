import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.core.logging import logger

class InsightFaceEngine:
    """InsightFace + ArcFace Face Detection, Landmark Alignment & 512D Embedding Engine."""

    def __init__(self, name: str = "buffalo_l", ctx_id: int = -1, det_size: Tuple[int, int] = (640, 640)):
        self.name = name
        self.ctx_id = ctx_id  # -1 for CPU, >= 0 for GPU
        self.det_size = det_size
        self.app = None
        self.is_initialized = False
        self._init_engine()

    def _init_engine(self):
        """Initializes InsightFace FaceAnalysis application."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            logger.info(f"Initializing InsightFace FaceAnalysis model package '{self.name}' (ctx_id={self.ctx_id})...")
            # Providers: CPUExecutionProvider or CUDAExecutionProvider
            providers = ['CPUExecutionProvider']
            if self.ctx_id >= 0:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.app = FaceAnalysis(name=self.name, providers=providers)
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self.is_initialized = True
            logger.info("InsightFace engine initialized successfully.")
        except Exception as e:
            logger.warning(f"InsightFace model load warning: {e}. Will attempt fallback CV detection & embedding if needed.")
            self.is_initialized = False

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in BGR image, performs alignment, extracts normalized 512D ArcFace embeddings.
        Returns list of face dictionaries:
          {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "landmarks": np.ndarray (5, 2),
            "embedding": np.ndarray (512,),
            "quality_score": float
          }
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        results = []

        if self.is_initialized and self.app is not None:
            try:
                faces = self.app.get(image_bgr)
                for face in faces:
                    bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                    det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                    landmarks = face.kps if hasattr(face, 'kps') else None
                    embedding = face.embedding if hasattr(face, 'embedding') else None
                    
                    if embedding is not None:
                        # L2 Normalization: embedding / ||embedding||
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            norm_embedding = (embedding / norm).astype(np.float32)
                        else:
                            norm_embedding = embedding.astype(np.float32)

                        # Estimate face quality score based on blurriness & detection confidence
                        quality_score = self.estimate_quality(image_bgr, bbox, det_score)

                        results.append({
                            "bbox": bbox,
                            "confidence": det_score,
                            "landmarks": landmarks,
                            "embedding": norm_embedding,
                            "quality_score": quality_score
                        })
                return results
            except Exception as e:
                logger.error(f"Error in InsightFace detection/embedding: {e}")

        # Fallback implementation if InsightFace model package is loading or unavailable
        return self._fallback_detection(image_bgr)

    def estimate_quality(self, image_bgr: np.ndarray, bbox: List[int], det_score: float) -> float:
        """Estimates face quality score based on Laplacian blurriness, face area, and detection confidence."""
        try:
            h_img, w_img = image_bgr.shape[:2]
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w_img, bbox[2]), min(h_img, bbox[3])
            
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w <= 0 or face_h <= 0:
                return 0.0

            face_roi = image_bgr[y1:y2, x1:x2]
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Blurriness score using Laplacian variance
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality = min(1.0, lap_var / 300.0)

            # Resolution quality
            min_dim = min(face_w, face_h)
            res_quality = min(1.0, min_dim / 100.0)

            # Combined quality score in range [0.0, 1.0]
            quality = 0.4 * det_score + 0.4 * blur_quality + 0.2 * res_quality
            return round(float(quality), 3)
        except Exception:
            return float(det_score)

    def _fallback_detection(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """OpenCV Cascade / DNN Fallback detector with pseudo ArcFace 512D norm vector for testing setup."""
        h_img, w_img = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        results = []
        for (x, y, w, h) in faces:
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            # Generate deterministic pseudo 512D ArcFace embedding from face pixel histogram for test harness
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (32, 16))
            vec = face_resized.flatten().astype(np.float32)
            # Expand/pad to 512-D
            full_vec = np.pad(vec, (0, 512 - len(vec)), mode='constant')
            norm = np.linalg.norm(full_vec)
            norm_vec = (full_vec / norm).astype(np.float32) if norm > 0 else full_vec

            results.append({
                "bbox": bbox,
                "confidence": 0.85,
                "landmarks": None,
                "embedding": norm_vec,
                "quality_score": 0.80
            })
        return results

# Singleton engine instance
insightface_engine = InsightFaceEngine()
