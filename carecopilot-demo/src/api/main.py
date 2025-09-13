"""
CareCopilot FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.api.rag_api import query_engine
from src.agents.fhir_agent import fhir_agent

app = FastAPI(title="CareCopilot API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RAGQueryRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None
    document_type: Optional[str] = None

class FHIRConversionRequest(BaseModel):
    clinical_text: str
    patient_name: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/query")
async def rag_query(request: RAGQueryRequest):
    return query_engine.query(request.query)

@app.post("/api/v1/fhir/convert")
async def fhir_convert(request: FHIRConversionRequest):
    return fhir_agent.convert_to_fhir(request.clinical_text, request.patient_name)

@app.get("/api/v1/sample/note")
async def get_sample_note():
    return {
        "sample_note": """Patient has Type 2 diabetes mellitus and hypertension. 
Current medications include Metformin 500mg twice daily and Lisinopril 10mg daily.
Blood pressure today: 140/90 mmHg. Patient reports good medication compliance.""",
        "patient_name": "John Smith"
    }
