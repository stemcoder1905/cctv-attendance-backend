from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import get_current_user_claims, require_roles
from app.models.domain import AuditLog, User
from app.schemas.pydantic_models import AuditLogResponse

router = APIRouter(prefix="/audit-logs", tags=["Audit Logging"])

@router.get("", response_model=List[AuditLogResponse], dependencies=[Depends(require_roles(["ADMIN"]))])
async def list_audit_logs(
    limit: int = 100,
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """Fetches system audit logs for administrative security review (ADMIN role required)."""
    stmt = (
        select(AuditLog, User.name.label("user_name"))
        .outerjoin(User, AuditLog.user_id == User.id)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    rows = res.all()

    logs = []
    for log, u_name in rows:
        resp = AuditLogResponse.model_validate(log)
        resp.user_name = u_name or "System"
        logs.append(resp)

    return logs
