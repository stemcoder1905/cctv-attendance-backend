from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import verify_password, hash_password, create_access_token, get_current_user_claims, require_roles
from app.core.audit import create_audit_log
from app.models.domain import User
from app.schemas.pydantic_models import LoginRequest, TokenResponse, UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Authenticates user with email and Argon2 password hash, returning JWT access token."""
    stmt = select(User).where(User.email == credentials.email.lower())
    res = await db.execute(stmt)
    user = res.scalars().first()

    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User account is deactivated.")

    # Create JWT token
    token = create_access_token(data={"sub": str(user.id), "email": user.email, "role": user.role, "name": user.name})

    # Log audit event
    client_ip = request.client.host if request.client else None
    await create_audit_log(db, user_id=user.id, action="USER_LOGIN", resource_type="User", resource_id=str(user.id), ip_address=client_ip)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user)
    }

@router.get("/me", response_model=UserResponse)
async def get_me(claims: dict = Depends(get_current_user_claims), db: AsyncSession = Depends(get_db)):
    """Fetches details of the currently authenticated user."""
    user_id = int(claims["sub"])
    stmt = select(User).where(User.id == user_id)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user

@router.post("/users", response_model=UserResponse, dependencies=[Depends(require_roles(["ADMIN"]))])
async def create_user(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """Creates a new administrative user account (ADMIN role required)."""
    stmt = select(User).where(User.email == user_in.email.lower())
    res = await db.execute(stmt)
    if res.scalars().first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

    new_user = User(
        name=user_in.name,
        email=user_in.email.lower(),
        password_hash=hash_password(user_in.password),
        role=user_in.role.upper(),
        is_active=True
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user
