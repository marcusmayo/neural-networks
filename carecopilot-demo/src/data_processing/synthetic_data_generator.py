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
