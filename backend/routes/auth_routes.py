from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import get_db
from models import User
from pydantic import BaseModel
from fastapi_jwt_auth import AuthJWT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

@router.post("/register", status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="User already exists")
    hashed_password = pwd_context.hash(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
    # Log the incoming request data
    logger.info(f"Login attempt with data: {user.dict()}")

    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        logger.error("User not found")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not pwd_context.verify(user.password, db_user.password):
        logger.error("Incorrect password")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # If successful, generate access token
    access_token = Authorize.create_access_token(subject=str(db_user.id))
    logger.info(f"Access token generated for user id: {db_user.id}")

    return {"access_token": access_token}
