import pytest
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token, sanitize_camera_url
from app.schemas.pydantic_models import PersonResponse

def test_argon2_password_hashing():
    raw_pass = "SecureP@ssword2026"
    hashed = hash_password(raw_pass)
    assert hashed != raw_pass
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

def test_jwt_token_claims():
    data = {"sub": "42", "role": "ADMIN", "name": "Admin User"}
    token = create_access_token(data)
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "ADMIN"

def test_camera_url_credential_masking():
    url_with_creds = "rtsp://admin:super_secret_password_123@192.168.1.50:554/h264Preview"
    sanitized = sanitize_camera_url(url_with_creds)
    assert "super_secret_password_123" not in sanitized
    assert "admin:****@" in sanitized

def test_biometric_embedding_privacy_in_schemas():
    # Verify PersonResponse schema fields do not expose raw embeddings
    fields = PersonResponse.model_fields.keys()
    assert "embedding" not in fields
    assert "face_embedding" not in fields
    assert "embedding_json" not in fields
