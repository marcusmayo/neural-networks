"""
Data Processing Pipeline for CareCopilot
"""
import re
import json
import boto3
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

from config.settings import settings
from src.models.database import Document, DocumentChunk, Patient, SessionLocal

class PHIRedactor:
    def __init__(self):
        self.phi_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b\d{10,}\b', '[PHONE]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'MRN\d+', '[MRN]'),
            (r'PT\d+', '[PATIENT_ID]'),
        ]
    
    def redact_phi(self, text: str) -> str:
        redacted_text = text
        for pattern, replacement in self.phi_patterns:
            redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.IGNORECASE)
        return redacted_text

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks

class EmbeddingGenerator:
    def __init__(self):
        print(f"🔄 Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("✅ Embedding model loaded successfully")
    
    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

class DataProcessor:
    def __init__(self):
        self.phi_redactor = PHIRedactor()
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        try:
            self.s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
        except:
            print("⚠️ AWS S3 client initialization failed, continuing without S3")
            self.s3_client = None
    
    def upload_to_s3(self, content: str, key: str) -> bool:
        if not self.s3_client:
            return False
        try:
            self.s3_client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=content, ServerSideEncryption='AES256')
            return True
        except Exception as e:
            print(f"⚠️ S3 upload failed: {e}")
            return False
    
    def process_document(self, document_data: Dict[str, Any], db: Session) -> None:
        print(f"🔄 Processing document: {document_data['title']}")
        
        content = document_data["content"]
        patient_data = document_data["patient"]
        
        # Store patient if not exists
        existing_patient = db.query(Patient).filter(Patient.id == patient_data["id"]).first()
        if not existing_patient:
            patient = Patient(
                id=patient_data["id"],
                name=patient_data["name"],
                date_of_birth=datetime.fromisoformat(patient_data["date_of_birth"]),
                gender=patient_data["gender"],
                mrn=patient_data["mrn"],
                metadata=patient_data["metadata"]
            )
            db.add(patient)
        
        # PHI redaction
        redacted_content = self.phi_redactor.redact_phi(content)
        doc_embedding = self.embedding_generator.generate_embedding(redacted_content[:1000])
        
        # Upload to S3 (optional)
        s3_key = f"documents/{document_data['patient_id']}/{document_data['id']}.txt"
        s3_success = self.upload_to_s3(content, s3_key) if self.s3_client else False
        
        # Create document record
        document = Document(
            id=document_data["id"],
            patient_id=document_data["patient_id"],
            document_type=document_data["document_type"],
            title=document_data["title"],
            content=content,
            processed_content=redacted_content,
            s3_key=s3_key if s3_success else None,
            embedding=doc_embedding,
            metadata={**document_data["metadata"], "s3_uploaded": s3_success}
        )
        
        db.add(document)
        db.flush()
        
        # Create chunks
        chunks = self.chunker.chunk_text(redacted_content)
        chunk_embeddings = self.embedding_generator.generate_embeddings_batch(chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_record = DocumentChunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk,
                embedding=embedding,
                metadata={"word_count": len(chunk.split())}
            )
            db.add(chunk_record)
        
        print(f"✅ Processed document with {len(chunks)} chunks")
    
    def process_dataset(self, dataset_file: str) -> None:
        print(f"🔄 Loading dataset from {dataset_file}")
        
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
        
        print(f"📊 Processing {len(dataset)} documents...")
        
        db = SessionLocal()
        try:
            processed_count = 0
            for doc_data in dataset:
                try:
                    self.process_document(doc_data, db)
                    processed_count += 1
                    if processed_count % 10 == 0:
                        db.commit()
                        print(f"💾 Progress: {processed_count}/{len(dataset)}")
                except Exception as e:
                    print(f"❌ Failed to process {doc_data.get('title', 'Unknown')}: {e}")
            
            db.commit()
            print(f"🎉 Processing complete! Processed: {processed_count}")
            
        finally:
            db.close()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            doc_count = db.query(Document).count()
            chunk_count = db.query(DocumentChunk).count()
            patient_count = db.query(Patient).count()
            
            doc_types = db.query(Document.document_type, func.count(Document.document_type).label('count')).group_by(Document.document_type).all()
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "total_patients": patient_count,
                "document_types": dict(doc_types),
                "average_chunks_per_document": chunk_count / doc_count if doc_count > 0 else 0
            }
        finally:
            db.close()

def main():
    import os
    os.makedirs("data/synthetic", exist_ok=True)
    
    processor = DataProcessor()
    synthetic_file = "data/synthetic/synthetic_medical_data.json"
    
    if not os.path.exists(synthetic_file):
        print("❌ Synthetic data not found. Please run synthetic_data_generator.py first")
        return
    
    processor.process_dataset(synthetic_file)
    stats = processor.get_processing_stats()
    print("\n📊 Processing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
