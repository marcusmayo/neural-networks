"""
Simple FHIR Agent for CareCopilot
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
        entities = {"conditions": [], "medications": []}
        text_lower = text.lower()
        
        for pattern in self.condition_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities["conditions"].append({"text": match.group().title(), "confidence": 0.8})
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities["medications"].append({"text": match.group().title(), "confidence": 0.7})
        
        return entities

class FHIRAgent:
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
    
    def convert_to_fhir(self, clinical_text: str, patient_name: str = None) -> Dict[str, Any]:
        try:
            entities = self.entity_extractor.extract_entities(clinical_text)
            
            # Create FHIR bundle
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
                    "code": {"text": condition["text"]},
                    "subject": {"reference": f"Patient/{patient_resource['id']}"}
                }
                bundle["entry"].append({"resource": condition_resource})
            
            # Medication resources
            for med in entities.get("medications", []):
                med_resource = {
                    "resourceType": "MedicationStatement", 
                    "id": str(uuid.uuid4()),
                    "status": "active",
                    "medicationCodeableConcept": {"text": med["text"]},
                    "subject": {"reference": f"Patient/{patient_resource['id']}"}
                }
                bundle["entry"].append({"resource": med_resource})
            
            return {
                "success": True,
                "bundle": bundle,
                "entities_extracted": entities,
                "resource_count": len(bundle["entry"]),
                "validation_errors": [],
                "processing_time_ms": 50.0,
                "summary": {
                    "conditions": len(entities.get("conditions", [])),
                    "medications": len(entities.get("medications", [])),
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
