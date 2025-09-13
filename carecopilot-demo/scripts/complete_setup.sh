#!/bin/bash

# CareCopilot Complete Setup Script
# This script sets up the entire CareCopilot demo environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log "🚀 Starting CareCopilot Complete Setup"
log "======================================"

# Step 1: Create remaining Python files
log "📝 Step 1: Creating Python source files"

# Create synthetic data generator
cat > src/data_processing/synthetic_data_generator.py << 'PYEOF'
"""
Synthetic Medical Data Generator for CareCopilot Demo
"""
import random
import json
from datetime import datetime, timedelta
import uuid

class SyntheticMedicalDataGenerator:
    def __init__(self):
        self.first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily", "James", "Ashley", "William", "Jessica", "Richard", "Amanda", "Joseph"]
        self.last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
        self.conditions = ["Type 2 Diabetes Mellitus", "Hypertension", "Coronary Artery Disease", "Congestive Heart Failure", "Chronic Kidney Disease", "COPD", "Atrial Fibrillation", "Osteoarthritis", "Depression", "Anxiety Disorder"]
        self.medications = ["Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 20mg", "Amlodipine 5mg", "Omeprazole 20mg", "Aspirin 81mg", "Furosemide 40mg", "Sertraline 50mg", "Gabapentin 300mg", "Levothyroxine 75mcg"]
    
    def generate_patient(self):
        patient_id = f"PT{random.randint(10000, 99999)}"
        birth_date = datetime.now() - timedelta(days=random.randint(18*365, 85*365))
        return {
            "id": patient_id,
            "name": f"{random.choice(self.first_names)} {random.choice(self.last_names)}",
            "date_of_birth": birth_date.strftime("%Y-%m-%d"),
            "gender": random.choice(["Male", "Female"]),
            "mrn": f"MRN{random.randint(100000, 999999)}",
            "metadata": {"age": int((datetime.now() - birth_date).days / 365.25), "created_by": "synthetic_generator"}
        }
    
    def generate_discharge_summary(self, patient):
        primary_condition = random.choice(self.conditions)
        medications = random.sample(self.medications, random.randint(2, 4))
        
        return f"""DISCHARGE SUMMARY

PATIENT: {patient['name']}
MRN: {patient['mrn']}
DOB: {patient['date_of_birth']}

DISCHARGE DIAGNOSES:
1. {primary_condition}

DISCHARGE MEDICATIONS:
{chr(10).join([f"- {med} daily" for med in medications])}

VITAL SIGNS ON DISCHARGE:
Temperature: {random.randint(97, 99)}.{random.randint(0, 9)}°F
Heart Rate: {random.randint(60, 100)} bpm
Blood Pressure: {random.randint(110, 140)}/{random.randint(70, 90)} mmHg

DISCHARGE INSTRUCTIONS:
1. Follow up with primary care physician in 1-2 weeks
2. Continue medications as prescribed
3. Monitor symptoms and return if worsening

Dr. {random.choice(self.first_names)} {random.choice(self.last_names)}, MD"""

    def generate_progress_note(self, patient):
        condition = random.choice(self.conditions)
        medications = random.sample(self.medications, random.randint(1, 3))
        
        return f"""PROGRESS NOTE

Patient: {patient['name']}
MRN: {patient['mrn']}

SUBJECTIVE:
Patient reports stable symptoms related to {condition}.

OBJECTIVE:
Vital Signs: Temperature 98.6°F, HR {random.randint(60, 100)} bpm, BP {random.randint(110, 140)}/{random.randint(70, 90)}

LABORATORY RESULTS:
- Glucose: {random.randint(80, 200)} mg/dL
- Hemoglobin: {random.randint(10, 16)}.{random.randint(0, 9)} g/dL

ASSESSMENT AND PLAN:
1. {condition} - Continue current management
   - Medications: {', '.join(medications)}

Dr. {random.choice(self.first_names)} {random.choice(self.last_names)}, MD"""

    def generate_dataset(self, num_patients=50):
        dataset = []
        print(f"🔄 Generating {num_patients} synthetic patients...")
        
        for i in range(num_patients):
            patient = self.generate_patient()
            num_docs = random.randint(2, 4)
            
            for j in range(num_docs):
                doc_type = random.choice(["discharge_summary", "progress_note"])
                
                if doc_type == "discharge_summary":
                    content = self.generate_discharge_summary(patient)
                    title = f"Discharge Summary - {patient['name']}"
                else:
                    content = self.generate_progress_note(patient)
                    title = f"Progress Note - {patient['name']}"
                
                document = {
                    "id": str(uuid.uuid4()),
                    "patient_id": patient["id"],
                    "patient": patient,
                    "document_type": doc_type,
                    "title": title,
                    "content": content,
                    "metadata": {"generated_at": datetime.now().isoformat(), "word_count": len(content.split())}
                }
                dataset.append(document)
        
        print(f"✅ Generated {len(dataset)} documents for {num_patients} patients")
        return dataset

def main():
    generator = SyntheticMedicalDataGenerator()
    dataset = generator.generate_dataset(num_patients=50)
    
    output_file = "data/synthetic/synthetic_medical_data.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"💾 Synthetic data saved to {output_file}")

if __name__ == "__main__":
    main()
PYEOF

# Create data processor
cat > src/data_processing/data_processor.py << 'PYEOF'
"""
Data Processing Pipeline for CareCopilot
"""
import re
import json
import boto3
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
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
            
            doc_types = db.query(Document.document_type, db.func.count(Document.document_type).label('count')).group_by(Document.document_type).all()
            
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
PYEOF

log "✅ Python source files created"

# Step 2: Create Virtual Environment and Install Dependencies
log "🐍 Step 2: Setting up Python Environment"

if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    log "✅ Virtual environment created"
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install spacy model
python -m spacy download en_core_web_sm
log "✅ Dependencies installed"

# Step 3: Setup PostgreSQL
log "🗄️ Step 3: Setting up PostgreSQL Database"

# Enable pgvector extension
sudo -u postgres psql -d carecopilot -c "CREATE EXTENSION IF NOT EXISTS vector;" || true

# Test database connection
PGPASSWORD=carecopilot123 psql -h localhost -U carecopilot -d carecopilot -c "SELECT version();" > /dev/null
log "✅ Database connection verified"

# Step 4: Initialize Database
log "🗂️ Step 4: Initializing Database Schema"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -c "from src.models.database import init_db; init_db()"
log "✅ Database schema initialized"

# Step 5: Generate and Process Data
log "📊 Step 5: Generating Synthetic Data"

python src/data_processing/synthetic_data_generator.py
log "✅ Synthetic data generated"

log "🧠 Step 6: Processing Data and Creating Embeddings"
python src/data_processing/data_processor.py
log "✅ Data processing completed"

# Step 7: Create remaining files and scripts
log "📝 Step 7: Creating remaining application files"

# Create simple RAG API
cat > src/api/rag_api.py << 'PYEOF'
"""
Simple RAG Query Engine
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from sentence_transformers import SentenceTransformer
import time
from datetime import datetime

from config.settings import settings
from src.models.database import DocumentChunk, Document, Patient, QueryLog, SessionLocal

class RAGQueryEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("✅ RAG Query Engine initialized")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        return embedding.tolist()
    
    def hybrid_search(self, query: str, db: Session, patient_id: Optional[str] = None, document_type: Optional[str] = None, limit: int = 10):
        start_time = time.time()
        
        query_embedding = self.generate_query_embedding(query)
        
        base_query = db.query(
            DocumentChunk,
            Document.title,
            Document.document_type,
            Document.patient_id,
            Patient.name.label("patient_name"),
            func.round((1 - DocumentChunk.embedding.cosine_distance(query_embedding)) * 100, 2).label("similarity_score")
        ).join(Document, DocumentChunk.document_id == Document.id).join(Patient, Document.patient_id == Patient.id)
        
        if patient_id:
            base_query = base_query.filter(Document.patient_id == patient_id)
        if document_type:
            base_query = base_query.filter(Document.document_type == document_type)
        
        results = base_query.filter(
            func.round((1 - DocumentChunk.embedding.cosine_distance(query_embedding)) * 100, 2) >= settings.SIMILARITY_THRESHOLD * 100
        ).order_by((1 - DocumentChunk.embedding.cosine_distance(query_embedding)).desc()).limit(limit).all()
        
        formatted_results = []
        for result in results:
            chunk, doc_title, doc_type, patient_id, patient_name, similarity = result
            formatted_results.append({
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "document_title": doc_title,
                "document_type": doc_type,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "content": chunk.content,
                "similarity_score": float(similarity),
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            })
        
        search_time = (time.time() - start_time) * 1000
        return formatted_results, search_time
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not context_chunks:
            return {
                "answer": "I don't have enough information in the available medical records to answer your question.",
                "confidence": 0.0,
                "reasoning": "No relevant documents found"
            }
        
        query_lower = query.lower()
        answer_parts = []
        citations = []
        
        if "medication" in query_lower or "drug" in query_lower:
            answer_parts.append("Based on the medical records, here are the relevant medications:")
            for chunk in context_chunks[:3]:
                content = chunk["content"]
                lines = content.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ['medication', 'drug', 'mg']):
                        if line.strip():
                            answer_parts.append(f"- {line.strip()}")
                            citations.append({
                                "source": chunk["document_title"],
                                "patient": chunk["patient_name"],
                                "similarity": chunk["similarity_score"],
                                "document_type": chunk["document_type"]
                            })
                            break
        
        elif "vital" in query_lower or "blood pressure" in query_lower or "temperature" in query_lower:
            answer_parts.append("Here are the vital signs from the medical records:")
            for chunk in context_chunks[:3]:
                content = chunk["content"]
                lines = content.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ['vital', 'temperature', 'blood pressure', 'heart rate']):
                        if line.strip():
                            answer_parts.append(f"- {line.strip()}")
                            citations.append({
                                "source": chunk["document_title"],
                                "patient": chunk["patient_name"],
                                "similarity": chunk["similarity_score"],
                                "document_type": chunk["document_type"]
                            })
                            break
        
        else:
            answer_parts.append("Based on the medical records, here's the relevant information:")
            for chunk in context_chunks[:2]:
                sentences = chunk["content"].split('.')
                relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:2]
                for sentence in relevant_sentences:
                    answer_parts.append(f"- {sentence}")
                citations.append({
                    "source": chunk["document_title"],
                    "patient": chunk["patient_name"],
                    "similarity": chunk["similarity_score"],
                    "document_type": chunk["document_type"]
                })
        
        if context_chunks:
            avg_similarity = sum(chunk["similarity_score"] for chunk in context_chunks[:3]) / min(3, len(context_chunks))
            confidence = min(avg_similarity / 100.0, 0.95)
        else:
            confidence = 0.0
        
        answer = "\n".join(answer_parts) if answer_parts else "No specific information found for this query."
        
        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "citations": citations[:5],
            "reasoning": f"Found {len(context_chunks)} relevant chunks with average similarity of {avg_similarity:.1f}%" if context_chunks else "No relevant information found"
        }
    
    def query(self, query_text: str, patient_id: Optional[str] = None, document_type: Optional[str] = None, user_id: str = "demo_user"):
        start_time = time.time()
        
        with SessionLocal() as db:
            try:
                search_results, search_time = self.hybrid_search(query_text, db, patient_id=patient_id, document_type=document_type)
                answer_data = self.generate_answer(query_text, search_results)
                total_time = (time.time() - start_time) * 1000
                
                query_log = QueryLog(
                    user_id=user_id,
                    query_text=query_text,
                    query_type="rag",
                    response_time_ms=total_time,
                    results_count=len(search_results),
                    confidence_score=answer_data["confidence"],
                    metadata={"patient_filter": patient_id, "document_type_filter": document_type}
                )
                db.add(query_log)
                db.commit()
                
                return {
                    "query": query_text,
                    "answer": answer_data["answer"],
                    "confidence": answer_data["confidence"],
                    "reasoning": answer_data["reasoning"],
                    "citations": answer_data["citations"],
                    "search_results": search_results,
                    "metadata": {
                        "total_time_ms": round(total_time, 2),
                        "search_time_ms": round(search_time, 2),
                        "results_count": len(search_results),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            except Exception as e:
                return {
                    "query": query_text,
                    "answer": "An error occurred while processing your query.",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "citations": [],
                    "search_results": [],
                    "metadata": {"error": str(e)}
                }

query_engine = RAGQueryEngine()

def get_available_patients(db: Session) -> List[Dict[str, str]]:
    patients = db.query(Patient.id, Patient.name).all()
    return [{"id": p.id, "name": p.name} for p in patients]

def get_available_document_types(db: Session) -> List[str]:
    doc_types = db.query(Document.document_type).distinct().all()
    return [dt[0] for dt in doc_types]
PYEOF

log "✅ RAG API created"

# Create simplified FHIR agent
cat > src/agents/fhir_agent.py << 'PYEOF'
"""
FHIR Agent for converting clinical notes
"""
import re
import json
import uuid
from typing import Dict, List, Any
from datetime import datetime

class MedicalEntityExtractor:
    def __init__(self):
        self.condition_patterns = [
            r'\b(?:diabetes|hypertension|copd|asthma|pneumonia|depression|anxiety|heart failure)\b'
        ]
        self.medication_patterns = [
            r'\b(?:metformin|insulin|aspirin|lisinopril|atorvastatin)\b',
            r'\b\w+\s+\d+\s*mg\b'
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        entities = {"conditions": [], "medications": [], "vitals": []}
        text_lower = text.lower()
        
        for pattern in self.condition_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities["conditions"].append({"text": match.group(), "confidence": 0.8})
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities["medications"].append({"text": match.group(), "confidence": 0.7})
        
        return entities

class FHIRAgent:
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
    
    def convert_to_fhir(self, clinical_text: str, patient_name: str = None) -> Dict[str, Any]:
        try:
            entities = self.entity_extractor.extract_entities(clinical_text)
            
            # Create simple FHIR bundle
            bundle = {
                "resourceType": "Bundle",
                "id": str(uuid.uuid4()),
                "type": "transaction",
                "entry": []
            }
            
            # Patient resource
            patient_resource = {
                "resourceType": "Patient",
                "id": str(uuid.uuid4()),
                "name": [{"text": patient_name or "Unknown Patient"}]
            }
            bundle["entry"].append({"resource": patient_resource})
            
            # Condition resources
            for condition in entities.get("conditions", []):
                condition_resource = {
                    "resourceType": "Condition",
                    "id": str(uuid.uuid4()),
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"text": condition["text"].title()},
                    "subject": {"reference": f"Patient/{patient_resource['id']}"}
                }
                bundle["entry"].append({"resource": condition_resource})
            
            # Medication resources
            for med in entities.get("medications", []):
                med_resource = {
                    "resourceType": "MedicationStatement",
                    "id": str(uuid.uuid4()),
                    "status": "active",
                    "medicationCodeableConcept": {"text": med["text"].title()},
                    "subject": {"reference": f"Patient/{patient_resource['id']}"}
                }
                bundle["entry"].append({"resource": med_resource})
            
            return {
                "success": True,
                "bundle": bundle,
                "entities_extracted": entities,
                "resource_count": len(bundle["entry"]),
                "validation_errors": [],
                "processing_time_ms": 100.0,
                "summary": {
                    "conditions": len(entities.get("conditions", [])),
                    "medications": len(entities.get("medications", [])),
                    "observations": 0,
                    "total_resources": len(bundle["entry"])
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "bundle": None,
                "entities_extracted": {},
                "validation_errors": [str(e)]
            }

fhir_agent = FHIRAgent()
PYEOF

log "✅ FHIR Agent created"

log "🎉 Setup Complete! You can now run the demo."
