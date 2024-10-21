from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

# SQLAlchemy Database URL
SQLALCHEMY_DATABASE_URI = settings.SQLALCHEMY_DATABASE_URI

# Creating the SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URI)

# Creating a configured session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for getting the session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
