import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import logger

async def create_audit_log(
    db: AsyncSession,
    user_id: Optional[int],
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Records an administrative audit log in the database."""
    from app.models.domain import AuditLog
    
    # Sanitize metadata to remove any potential sensitive keys
    safe_metadata = {}
    if metadata:
        for k, v in metadata.items():
            if any(secret_key in k.lower() for secret_key in ["password", "token", "embedding", "secret", "key"]):
                safe_metadata[k] = "[REDACTED]"
            else:
                safe_metadata[k] = v

    try:
        log_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            ip_address=ip_address,
            metadata_json=json.dumps(safe_metadata) if safe_metadata else None,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(log_entry)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to record audit log: {e}")
