# from flask import Blueprint, request, jsonify
# from models import db, User
# from flask_jwt_extended import create_access_token
# import datetime

# auth = Blueprint('auth', __name__)

# @auth.route('/register', methods=['POST'])
# def register():
#     data = request.get_json()
#     username = data.get('username')
#     email = data.get('email')
#     password = data.get('password')

#     if User.query.filter_by(email=email).first():
#         return jsonify(message="User already exists"), 400

#     new_user = User(username=username, email=email, password=password)
#     db.session.add(new_user)
#     db.session.commit()

#     return jsonify(message="User registered successfully"), 201

# @auth.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     email = data.get('email')
#     password = data.get('password')

#     user = User.query.filter_by(email=email).first()

#     if user and bcrypt.check_password_hash(user.password, password):
#         token = create_access_token(identity=user.id, expires_delta=datetime.timedelta(hours=1))
#         return jsonify(token=token), 200
#     else:
#         return jsonify(message="Invalid credentials"), 401

# from flask import Blueprint, request, jsonify
# from models import db, User
# from flask_jwt_extended import create_access_token
# from flask_bcrypt import Bcrypt
# import datetime

# auth = Blueprint('auth', __name__)
# bcrypt = Bcrypt()

# @auth.route('/register', methods=['POST'])
# def register():
#     data = request.get_json()
#     username = data.get('username')
#     email = data.get('email')
#     password = data.get('password')

#     if not username or not email or not password:
#         return jsonify(message="All fields are required"), 400

#     # Check if the user already exists
#     if User.query.filter_by(email=email).first():
#         return jsonify(message="User already exists"), 400

#     # Hash the password before storing it
#     hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
#     new_user = User(username=username, email=email, password=hashed_password)
#     db.session.add(new_user)
#     db.session.commit()

#     return jsonify(message="User registered successfully"), 201

# @auth.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     email = data.get('email')
#     password = data.get('password')

#     # Check if the email is registered
#     user = User.query.filter_by(email=email).first()

#     if not user:
#         return jsonify(message="Email not registered"), 404

#     # Verify the password using bcrypt
#     if bcrypt.check_password_hash(user.password, password):
#         token = create_access_token(identity=user.id, expires_delta=datetime.timedelta(hours=1))
#         return jsonify(token=token), 200
#     else:
#         return jsonify(message="Invalid credentials"), 401

# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from passlib.context import CryptContext
# # from fastapi_jwt_auth import AuthJWT
# from flask_jwt_extended import create_access_token, jwt_required

# from database import SessionLocal
# from models import User
# from pydantic import BaseModel

# router = APIRouter()
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# class UserCreate(BaseModel):
#     username: str
#     email: str
#     password: str

# class UserLogin(BaseModel):
#     email: str
#     password: str

# @router.post("/register", status_code=201)
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="User already exists")
#     hashed_password = pwd_context.hash(user.password)
#     new_user = User(username=user.username, email=user.email, password=hashed_password)
#     db.add(new_user)
#     db.commit()
#     return {"message": "User registered successfully"}

# @router.post("/login")
# def login(user: UserLogin, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
#     db_user = db.query(User).filter(User.email == user.email).first()
#     if not db_user or not pwd_context.verify(user.password, db_user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     access_token = Authorize.create_access_token(subject=db_user.id)
#     return {"access_token": access_token}

# from flask import Blueprint, request, jsonify
# from models import db, User
# from flask_jwt_extended import create_access_token
# from passlib.context import CryptContext
# from sqlalchemy.orm import Session

# auth = Blueprint('auth', __name__)
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Register endpoint
# @auth.route('/register', methods=['POST'])
# def register():
#     data = request.get_json()
#     username = data.get('username')
#     email = data.get('email')
#     password = data.get('password')

#     if not username or not email or not password:
#         return jsonify(message="All fields are required"), 400

#     # Check if the user already exists
#     if User.query.filter_by(email=email).first():
#         return jsonify(message="User already exists"), 400

#     # Hash the password before storing it
#     hashed_password = pwd_context.hash(password)
#     new_user = User(username=username, email=email, password=hashed_password)
#     db.session.add(new_user)
#     db.session.commit()

#     return jsonify(message="User registered successfully"), 201

# # Login endpoint
# @auth.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     email = data.get('email')
#     password = data.get('password')

#     # Check if the email is registered
#     user = User.query.filter_by(email=email).first()

#     if not user or not pwd_context.verify(password, user.password):
#         return jsonify(message="Invalid credentials"), 401

#     # Create an access token
#     access_token = create_access_token(identity=user.id)
#     return jsonify(access_token=access_token), 200

# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from passlib.context import CryptContext
# from database import get_db
# from models import User
# from pydantic import BaseModel

# router = APIRouter()
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# class UserCreate(BaseModel):
#     username: str
#     email: str
#     password: str

# class UserLogin(BaseModel):
#     email: str
#     password: str

# @router.post("/register", status_code=201)
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="User already exists")
#     hashed_password = pwd_context.hash(user.password)
#     new_user = User(username=user.username, email=user.email, password=hashed_password)
#     db.add(new_user)
#     db.commit()
#     return {"message": "User registered successfully"}

# @router.post("/login")
# def login(user: UserLogin, db: Session = Depends(get_db)):
#     db_user = db.query(User).filter(User.email == user.email).first()
#     if not db_user or not pwd_context.verify(user.password, db_user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     # For simplicity, let's generate a dummy token
#     access_token = "some_dummy_token"
#     return {"access_token": access_token}

# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from passlib.context import CryptContext
# from database import get_db
# from models import User
# from pydantic import BaseModel, BaseSettings  # Correctly importing BaseSettings
# from fastapi_jwt_auth import AuthJWT
# from config import settings
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# router = APIRouter()
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# class UserCreate(BaseModel):
#     username: str
#     email: str
#     password: str

# class UserLogin(BaseModel):
#     email: str
#     password: str

# # Configuration class for JWT settings
# class AuthJWTSettings(BaseSettings):
#     authjwt_secret_key: str = settings.AUTHJWT_SECRET_KEY

# @AuthJWT.load_config
# def get_config():
#     return AuthJWTSettings()

# @router.post("/register", status_code=201)
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="User already exists")
#     hashed_password = pwd_context.hash(user.password)
#     new_user = User(username=user.username, email=user.email, password=hashed_password)
#     db.add(new_user)
#     db.commit()
#     return {"message": "User registered successfully"}

# # @router.post("/login")
# # def login(user: UserLogin, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
# #     db_user = db.query(User).filter(User.email == user.email).first()
# #     if not db_user or not pwd_context.verify(user.password, db_user.password):
# #         raise HTTPException(status_code=401, detail="Invalid credentials")

# #     access_token = Authorize.create_access_token(subject=str(db_user.id))
# #     return {"access_token": access_token}

# @router.post("/login")
# def login(user: UserLogin, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
#     # Log the incoming request data
#     # logger.info(f"Login attempt for email: {user.email}")
#     logger.info(f"Login attempt with data: {user.dict()}")

#     db_user = db.query(User).filter(User.email == user.email).first()
#     if not db_user:
#         logger.error("User not found")
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     if not pwd_context.verify(user.password, db_user.password):
#         logger.error("Incorrect password")
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     # If successful, generate access token
#     access_token = Authorize.create_access_token(subject=str(db_user.id))
#     logger.info(f"Access token generated for user id: {db_user.id}")

#     return {"access_token": access_token}

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
