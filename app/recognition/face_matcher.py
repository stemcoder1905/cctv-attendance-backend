import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import threading
from app.core.logging import logger
from app.recognition.embedding_service import normalize_embedding

class VectorSimilarityMatcher:
    """High-speed in-memory vector index for normalized ArcFace 512D embeddings."""

    def __init__(self):
        self._lock = threading.RLock()
        self.person_ids: List[int] = []
        self.unique_person_ids: List[str] = []
        self.names: List[str] = []
        self.matrix: Optional[np.ndarray] = None  # Shape (N, 512)

    def set_index(self, embeddings_data: List[Dict[str, Any]]):
        """Builds or refreshes the in-memory vector matrix from loaded database embeddings."""
        with self._lock:
            if not embeddings_data:
                self.person_ids = []
                self.unique_person_ids = []
                self.names = []
                self.matrix = None
                logger.info("Vector index cleared (0 embeddings).")
                return

            p_ids = []
            u_ids = []
            names = []
            vecs = []

            for item in embeddings_data:
                p_ids.append(item["person_id"])
                u_ids.append(item["unique_person_id"])
                names.append(item["name"])
                vecs.append(item["embedding"])

            self.person_ids = p_ids
            self.unique_person_ids = u_ids
            self.names = names
            # Stack into (N, 512) float32 matrix
            self.matrix = np.vstack(vecs).astype(np.float32)
            logger.info(f"Vector index updated with {len(p_ids)} registered face embeddings.")

    def search(self, probe_embedding: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Searches for best matching candidate vector(s) using Cosine Similarity.
        Returns sorted list of matches:
        [
           {
              "person_id": int,
              "unique_person_id": str,
              "name": str,
              "similarity": float
           }
        ]
        """
        with self._lock:
            if self.matrix is None or len(self.person_ids) == 0:
                return []

            norm_probe = normalize_embedding(probe_embedding)  # (512,)
            # Dot product of normalized vectors gives Cosine Similarity
            similarities = np.dot(self.matrix, norm_probe)  # (N,)

            # Top K indices sorted in descending order
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                results.append({
                    "person_id": self.person_ids[idx],
                    "unique_person_id": self.unique_person_ids[idx],
                    "name": self.names[idx],
                    "similarity": round(sim, 4)
                })

            return results

    def add_embedding(self, person_id: int, unique_person_id: str, name: str, embedding: np.ndarray):
        """Dynamically adds a single face embedding to the live vector index."""
        with self._lock:
            norm_emb = normalize_embedding(embedding).astype(np.float32)
            self.person_ids.append(person_id)
            self.unique_person_ids.append(unique_person_id)
            self.names.append(name)
            if self.matrix is None or len(self.matrix) == 0:
                self.matrix = np.array([norm_emb], dtype=np.float32)
            else:
                self.matrix = np.vstack([self.matrix, norm_emb]).astype(np.float32)
            logger.info(f"Dynamically added person '{name}' ({unique_person_id}) to vector index. Total embeddings: {len(self.person_ids)}")

    def find_best_match(self, probe_embedding: np.ndarray, threshold: float = 0.55) -> Optional[Dict[str, Any]]:
        """
        Finds single best matching identity above configurable similarity threshold.
        Returns match dict if similarity >= threshold, else None.
        """
        matches = self.search(probe_embedding, top_k=1)
        if matches and matches[0]["similarity"] >= threshold:
            return matches[0]
        return None


    def update_person_metadata(self, person_id: int, name: str, unique_person_id: str):
        """Updates in-memory name and unique_person_id for all occurrences of person_id."""
        with self._lock:
            updated_count = 0
            for i, p_id in enumerate(self.person_ids):
                if p_id == person_id:
                    self.names[i] = name
                    self.unique_person_ids[i] = unique_person_id
                    updated_count += 1
            if updated_count > 0:
                logger.info(f"Updated vector index metadata for person ID {person_id}: name='{name}', unique_person_id='{unique_person_id}'.")


    def remove_person_embeddings(self, person_id: int):
        """Instantly removes all embedding vectors for person_id from live matrix without full DB reload."""
        with self._lock:
            if not self.person_ids or self.matrix is None:
                return
            keep_indices = [i for i, p_id in enumerate(self.person_ids) if p_id != person_id]
            if len(keep_indices) == len(self.person_ids):
                return
            self.person_ids = [self.person_ids[i] for i in keep_indices]
            self.unique_person_ids = [self.unique_person_ids[i] for i in keep_indices]
            self.names = [self.names[i] for i in keep_indices]
            if keep_indices:
                self.matrix = self.matrix[keep_indices]
            else:
                self.matrix = None
            logger.info(f"Instantly purged embeddings for person ID {person_id}. Remaining: {len(self.person_ids)}")


# Global vector matcher singleton
vector_matcher = VectorSimilarityMatcher()
