---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
- compliance
- grc
- governance
- risk-management
- soc2
- iso27001
- hipaa
---

# GRC Compliance LLM - LoRA Fine-tuned TinyLlama

A specialized language model fine-tuned for governance, risk, and compliance (GRC) question-answering across SOC 2, ISO 27001, and HIPAA frameworks.

## Model Details

### Model Description

This model is a LoRA (Low-Rank Adaptation) fine-tuned version of TinyLlama-1.1B-Chat-v1.0, specifically trained to answer compliance-related questions with precise control and clause references. The model provides professional, audit-ready responses for governance, risk, and compliance queries.

- **Developed by:** Marcus Mayo
- **Model type:** Causal Language Model (Fine-tuned with LoRA)
- **Language(s):** English
- **License:** MIT
- **Finetuned from model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)

### Model Sources

- **Repository:** [https://github.com/marcusmayo/machine-learning-portfolio/tree/main/grc-llm-project](https://github.com/marcusmayo/machine-learning-portfolio/tree/main/grc-llm-project)
- **Demo:** [Streamlit Web Interface](https://grc-compliance-chatbot.streamlit.app)
- **Portfolio:** [Machine Learning Portfolio](https://github.com/marcusmayo/machine-learning-portfolio)

## Uses

### Direct Use

This model is designed for direct use in compliance and audit scenarios where precise, professional responses to GRC questions are required. It can answer questions about:

- **SOC 2 Security Controls**: Password management, access controls, system operations
- **ISO 27001 Information Security**: Access reviews, risk assessments, security policies  
- **HIPAA Healthcare Compliance**: Encryption requirements, access controls, data protection

### Downstream Use

The model can be integrated into:
- Compliance automation platforms (similar to Drata, Vanta)
- Internal audit preparation systems
- GRC training and education platforms
- Risk assessment documentation tools

### Out-of-Scope Use

This model should not be used for:
- Legal advice or definitive compliance interpretations
- Medical diagnosis or treatment recommendations
- Financial or investment advice
- General-purpose conversational AI outside compliance domains

## Bias, Risks, and Limitations

- **Domain Limitation**: Trained specifically on compliance frameworks; may not perform well on general topics
- **Training Data Size**: Limited to 17 Q&A pairs; may not cover all edge cases within compliance domains
- **Regulatory Updates**: Trained on specific versions of compliance frameworks; may not reflect latest regulatory changes
- **Professional Judgment**: Outputs should be reviewed by compliance professionals before use in actual audits

### Recommendations

Users should:
- Validate all responses with current compliance documentation
- Have compliance professionals review outputs for audit use
- Update training data regularly to reflect regulatory changes
- Use as a starting point for research, not as definitive compliance guidance

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/compliance-tinyllama-lora")

# Example usage
prompt = "Which SOC 2 control covers password requirements?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

### Training Data

- **Dataset Size**: 17 compliance Q&A pairs
- **Training Split**: 14 samples
- **Evaluation Split**: 3 samples
- **Format**: Instruction-response pairs in JSON format
- **Domains**: SOC 2, ISO 27001, HIPAA compliance frameworks

### Training Procedure

#### Training Hyperparameters

- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: All linear layers
- **Training regime**: Mixed precision (bf16)
- **Learning rate**: 2e-4
- **Batch size**: 1 (with gradient accumulation)
- **Epochs**: 2
- **Optimizer**: AdamW

#### Speeds, Sizes, Times

- **Training time**: 0.8 minutes
- **Hardware**: AWS EC2 c5.2xlarge (8 vCPU, 16GB RAM)
- **Training loss improvement**: 2.3 → 2.09 (9% improvement)
- **Model size**: ~4.5MB (LoRA adapters only)
- **Base model parameters**: 1.1B
- **Trainable parameters**: ~2.4M (LoRA adapters)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

3 held-out compliance questions covering SOC 2, ISO 27001, and HIPAA frameworks, designed to test the model's ability to provide accurate control mappings and professional responses.

#### Factors

- **Framework coverage**: Equal representation across SOC 2, ISO 27001, HIPAA
- **Question complexity**: Mix of specific control inquiries and general compliance concepts
- **Response quality**: Professional tone and accurate regulatory references

#### Metrics

- **Accuracy**: Correct control/clause identification
- **Completeness**: Coverage of required compliance elements
- **Professional quality**: Audit-ready response formatting

### Results

- **Evaluation Accuracy**: 100% (3/3 correct responses)
- **Control Mapping Accuracy**: 100% precise clause/control references
- **Response Quality**: Professional, audit-ready formatting
- **Average Response Time**: 4-22 seconds (depending on infrastructure)

## Model Examination

The model demonstrates strong performance in:
- **Precise Control Mapping**: Accurate SOC 2 CC6.1, ISO 27001 A.9.2.5 references
- **Professional Tone**: Responses suitable for audit documentation
- **Framework Consistency**: Maintains accuracy across different compliance standards

## Environmental Impact

Carbon emissions estimated using AWS EC2 c5.2xlarge for training:

- **Hardware Type**: AWS EC2 c5.2xlarge (Intel Xeon Platinum 8000 series)
- **Hours used**: 0.013 hours (0.8 minutes)
- **Cloud Provider**: AWS
- **Compute Region**: us-east-1
- **Carbon Emitted**: Minimal (~0.001 kg CO2eq estimated)

## Technical Specifications

### Model Architecture and Objective

- **Base Architecture**: TinyLlama transformer (1.1B parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Objective**: Causal language modeling for compliance Q&A
- **Adapter Architecture**: Low-rank matrices applied to all linear layers

### Compute Infrastructure

#### Hardware

- **Training**: AWS EC2 c5.2xlarge (8 vCPU, 16GB RAM)
- **Deployment**: AWS SageMaker ml.t3.medium endpoint
- **Storage**: AWS S3 for model artifacts

#### Software

- **Framework**: PyTorch 2.0+
- **Fine-tuning Library**: PEFT (Parameter Efficient Fine-Tuning)
- **Deployment**: AWS SageMaker Python SDK
- **Web Interface**: Streamlit

## Citation

**BibTeX:**

```bibtex
@misc{mayo2025grc,
  title={GRC Compliance LLM: LoRA Fine-tuned TinyLlama for Governance, Risk, and Compliance},
  author={Marcus Mayo},
  year={2025},
  url={https://github.com/marcusmayo/machine-learning-portfolio/tree/main/grc-llm-project}
}
```

**APA:**

Mayo, M. (2025). GRC Compliance LLM: LoRA Fine-tuned TinyLlama for Governance, Risk, and Compliance. *Machine Learning Portfolio*. https://github.com/marcusmayo/machine-learning-portfolio/tree/main/grc-llm-project

## Glossary

- **GRC**: Governance, Risk, and Compliance
- **LoRA**: Low-Rank Adaptation - Parameter-efficient fine-tuning method
- **SOC 2**: Service Organization Control 2 - Security and availability framework
- **ISO 27001**: International standard for information security management
- **HIPAA**: Health Insurance Portability and Accountability Act

## Model Card Authors

Marcus Mayo, MS, SPC6, RTE, PMP, PMI-ACP

## Model Card Contact

- **Email**: marcus.mayo@gmail.com
- **LinkedIn**: [Marcus Mayo](https://www.linkedin.com/in/marcus-mayo-ms-spc6-rte-pmp-pmi-acp/)
- **GitHub**: [@marcusmayo](https://github.com/marcusmayo)

### Framework versions

- PEFT 0.17.1
- PyTorch 2.0+
- Transformers 4.35+
- AWS SageMaker Python SDK 2.190+
