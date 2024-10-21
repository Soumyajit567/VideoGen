import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    AUTHJWT_SECRET_KEY: str = os.environ.get('SECRET_KEY') or 'e7f2b8d3c4a6f1b9e2c3f0a1d6e8b7a2c4d3e9f8a7b2c6f1e4d2a3b8c7f0e5'
    SQLALCHEMY_DATABASE_URI: str = os.environ.get('DATABASE_URL') or 'postgresql://postgres:Soumyajit123#@localhost:5432/videogen_db'

    class Config:
        env_file = ".env"

settings = Settings()
