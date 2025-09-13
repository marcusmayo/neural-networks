"""
CareCopilot Configuration Settings
Optimized for local PostgreSQL setup
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CareCopilot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Local PostgreSQL Database (instead of AWS RDS)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "carecopilot"
    DB_USER: str = "carecopilot"
    DB_PASSWORD: str = "carecopilot123"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # AWS S3 (Keep for demo purposes)
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = os.getenv("CARECOPILOT_BUCKET", "carecopilot-data-1757768714")
    
    # ML Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Small, fast model
    EMBEDDING_DIMENSION: int = 384
    
    # Vector Search
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_SEARCH_RESULTS: int = 10
    
    # Text Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_TEXT_LENGTH: int = 10000
    
    # PHI Detection Patterns
    PHI_PATTERNS: list = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{10,}\b',  # Phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
    ]
    
    # FHIR Settings
    FHIR_SERVER_URL: str = "http://localhost:8080/fhir"  # Local HAPI FHIR server
    FHIR_VERSION: str = "R4"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/carecopilot.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
