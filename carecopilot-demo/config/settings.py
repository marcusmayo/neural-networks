import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Use Docker postgres service for database connection
    DATABASE_URL: str = os.environ.get('DATABASE_URL', 'postgresql://carecopilot:demo123@postgres:5432/carecopilot')
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

settings = Settings()
