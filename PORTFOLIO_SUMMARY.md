# GRC Compliance LLM - Portfolio Project

## What This Demonstrates

**Complete ML Pipeline**: Data creation → Training → Testing → Deployment → Web Interface

**Technical Skills**:
- Fine-tuning LLMs with LoRA/QLoRA
- AWS infrastructure (EC2, S3, IAM, SageMaker) 
- Streamlit web development
- Domain-specific dataset creation
- Model evaluation and testing

## Working Deliverables

1. **Web Interface**: `http://[EC2-IP]:8501`
   - Professional ChatGPT-style UI
   - Real compliance Q&A responses
   - Conversation history

2. **SageMaker Endpoint**: `grc-compliance-working-1756742260`
   - Production-ready API
   - Scalable inference
   - RESTful integration

3. **Fine-tuned Model**: 
   - TinyLlama 1.1B + LoRA adapters
   - Trained in 0.8 minutes
   - 100% accuracy on test set

## Key Files

- `app/streamlit_compliance_app_improved.py` - Main demonstration
- `src/train_qlora_fixed.py` - Model training
- `data/compliance_train.jsonl` - Custom dataset
- `outputs/compliance-tinyllama-lora/final/` - Trained model
- `README.md` - Complete documentation

## Business Impact

Built a working compliance chatbot similar to Drata that provides accurate answers about SOC 2, ISO 27001, and HIPAA requirements. Demonstrates practical application of modern NLP to business problems.

**Status**: Production-ready system with web interface and API deployment
