import pytest
import numpy as np
from app.recognition.embedding_service import normalize_embedding, validate_registration_image
from app.recognition.face_matcher import VectorSimilarityMatcher

def test_l2_normalization():
    vec = np.array([3.0, 4.0] + [0.0]*510, dtype=np.float32)
    norm_vec = normalize_embedding(vec)
    norm = np.linalg.norm(norm_vec)
    assert abs(norm - 1.0) < 1e-5

def test_vector_similarity_search():
    matcher = VectorSimilarityMatcher()
    
    # 2 registered synthetic 512D vectors
    v1 = normalize_embedding(np.random.randn(512).astype(np.float32))
    v2 = normalize_embedding(np.random.randn(512).astype(np.float32))

    data = [
        {"person_id": 101, "unique_person_id": "EMP101", "name": "Alice", "embedding": v1},
        {"person_id": 102, "unique_person_id": "EMP102", "name": "Bob", "embedding": v2}
    ]
    matcher.set_index(data)

    # Search with v1 probe -> Should match Alice with similarity ~1.0
    match = matcher.find_best_match(v1, threshold=0.55)
    assert match is not None
    assert match["person_id"] == 101
    assert match["name"] == "Alice"
    assert match["similarity"] >= 0.99

def test_low_confidence_unknown_rejection():
    matcher = VectorSimilarityMatcher()
    v1 = normalize_embedding(np.random.randn(512).astype(np.float32))
    data = [{"person_id": 101, "unique_person_id": "EMP101", "name": "Alice", "embedding": v1}]
    matcher.set_index(data)

    # Search with completely orthogonal probe -> Should return None (UNKNOWN)
    v_random = normalize_embedding(np.random.randn(512).astype(np.float32))
    match = matcher.find_best_match(v_random, threshold=0.95)
    assert match is None
