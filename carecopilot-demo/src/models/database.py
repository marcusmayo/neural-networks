"""
Database models and connection setup for CareCopilot
Using local PostgreSQL with pgvector extension
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime
from config.settings import settings

# Database setup
engine = create_engine(settings.database_url, echo=settings.DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    """Stores medical documents with vector embeddings"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String(50), index=True)
    document_type = Column(String(100))  # discharge_summary, progress_note, etc.
    title = Column(String(500))
    content = Column(Text)
    processed_content = Column(Text)  # PHI-redacted content
    s3_key = Column(String(500))  # S3 location
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    document_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentChunk(Base):
    """Stores document chunks for RAG retrieval"""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), index=True)
    chunk_index = Column(Integer)
    content = Column(Text)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    chunk_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    """Patient information (synthetic data only)"""
    __tablename__ = "patients"
    
    id = Column(String(50), primary_key=True)  # Patient ID
    name = Column(String(200))
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    mrn = Column(String(50), unique=True)  # Medical Record Number
    patient_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    """Logs all queries for audit and analytics"""
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100))
    query_text = Column(Text)
    query_type = Column(String(50))  # rag, fhir_conversion
    response_time_ms = Column(Float)
    results_count = Column(Integer)
    confidence_score = Column(Float)
    query_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class FHIRConversion(Base):
    """Logs FHIR conversion attempts"""
    __tablename__ = "fhir_conversions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_text = Column(Text)
    fhir_bundle = Column(JSON)
    entities_extracted = Column(JSON)
    validation_status = Column(Boolean)
    validation_errors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database connection functions
def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def init_db():
    """Initialize database with extensions and tables"""
    from sqlalchemy import text
    
    try:
        # Ensure pgvector extension is enabled
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("✅ pgvector extension enabled")
        
        # Create tables
        create_tables()
        
        # Create vector indices for better performance
        with engine.connect() as conn:
            # Index for document embeddings
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """))
            
            # Index for chunk embeddings
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """))
            
            conn.commit()
            print("✅ Vector indices created for optimal search performance")
        
        print("🎉 Database initialization complete!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    init_db()
