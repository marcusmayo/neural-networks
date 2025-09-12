# 🚀 Machine Learning & AI Engineering Portfolio

[![GitHub stars](https://img.shields.io/github/stars/marcusmayo/machine-learning-portfolio?style=social)](https://github.com/marcusmayo/machine-learning-portfolio/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/marcusmayo/machine-learning-portfolio?style=social)](https://github.com/marcusmayo/machine-learning-portfolio/network)
[![GitHub issues](https://img.shields.io/github/issues/marcusmayo/machine-learning-portfolio)](https://github.com/marcusmayo/machine-learning-portfolio/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to my comprehensive machine-learning and AI engineering portfolio! This repository showcases end-to-end ML projects, from research and experimentation to production-ready deployments with complete MLOps pipelines.

## 👨‍💻 About Me

I'm Marcus, a passionate Machine Learning Engineer and AI practitioner focused on building robust, scalable and production-ready AI systems. This portfolio demonstrates my expertise across the entire ML lifecycle, from data preprocessing and model development to deployment and monitoring. These projects showcase modern AI-augmented development practices, leveraging advanced AI assistants (Claude, Gemini, ChatGPT) to accelerate development cycles while maintaining enterprise-grade code quality and architectural excellence.

### **Core Competencies:**

- 🧠 **Machine Learning** — Deep learning, classical ML, computer vision, NLP
- 🔧 **MLOps** — CI/CD pipelines, model versioning, containerization, cloud deployment
- ☁️ **Cloud Platforms** — AWS, Azure, GCP
- 📊 **Data Engineering** — ETL pipelines, data preprocessing, feature engineering
- 🐍 **Programming** — Python, PyTorch, TensorFlow, scikit-learn, Flask, FastAPI
- 🤖 **AI-Augmented Development** — Advanced prompt engineering, AI-assisted coding, rapid prototyping with LLM collaboration

## 🧰 Overall Tech Stack Summary

The table below summarizes the key technologies used across my completed projects and coursework. Each entry is grouped by its place in the machine-learning pipeline and includes a brief explanation written in plain language.

| Pipeline Stage | Tool/Technology | Usage (Project/Course) | Simple Explanation |
|---|---|---|---|
| **Data Storage & Sources** | **AWS S3** | Fraud-Detection MLOps, Edenred Invoice Assistant, GRC-LLM — stores datasets and model artifacts | S3 is like a big cloud hard-drive. It keeps our data and trained models so we can load them later. |
| | **DynamoDB** | Digital-Value-Chain serverless e-commerce — stores product offers and cart data | DynamoDB is a fast cloud database. It keeps items (like products) in a table so the app can read and write quickly. |
| | **CSV/JSON files** | Pinecone Vector DB, Fraud-Detection MLOps, GRC-LLM — holds training tables and text data | These are simple text files that hold tables or lists. They let us load training data from our computer. |
| | **Audio files** | Speech-Recognition project — WAV/MP3 clips for speech-to-text | Sound files are recordings. We feed them to the model to teach it to hear and transcribe speech. |
| **Data Preprocessing & Feature Engineering** | **Pandas** | Bike-Rental Predictor, Pinecone Vector DB, Fraud-Detection — reading CSVs, cleaning and encoding data | Pandas is like a spreadsheet for Python. It helps us read tables, clean them and get them ready for training. |
| | **NumPy** | All projects — math operations and array manipulation | NumPy lets us work with lists of numbers. It makes math operations fast and easy. |
| | **scikit-learn** | Bike-Rental preprocessing, Fraud-Detection metrics & validation | scikit-learn has tools to split data, scale numbers and measure how good a model is. |
| | **Librosa / soundfile / pydub / wave** | Speech-Recognition project — loading audio and extracting features | These libraries open sound files and turn them into numbers so a model can understand speech. |
| | **sentence-transformers** | Pinecone Vector DB — converts text into numeric embeddings | This library takes sentences and turns them into long lists of numbers so we can compare meanings. |
| | **dotenv** | Pinecone Vector DB — reads API keys from `.env` files | dotenv lets us keep secret keys in a file and load them into our program safely. |
| **Embeddings & Vectorization** | **Pinecone** | Pinecone Vector DB — cloud vector store for semantic search | Pinecone is a special database that stores those long number lists (embeddings). It helps us search for similar texts. |
| **Model Training** | **PyTorch** | Bike-Rental prediction — neural network training | PyTorch is a toolkit that lets us build and train neural networks. It teaches the computer to predict things. |
| | **TensorFlow + Keras** | Simple neural network notebook — single-layer perceptron for MNIST digits | TensorFlow and Keras help us build a simple "brain" to recognize handwritten numbers. |
| | **XGBoost (via SageMaker)** | Fraud-Detection MLOps — training the fraud classifier | XGBoost is a tree-based algorithm. It learns to tell normal transactions from fraudulent ones. |
| | **Transformers (BERT/GPT/XLNet)** | LLMs coursework — exploring large language models | These models understand and generate text. We used them to learn about language processing. |
| | **LoRA / PEFT** | GRC-LLM — efficient fine-tuning of TinyLlama | LoRA adapts a big language model using small extra pieces, saving time and cost. |
| | **Whisper & speech_recognition** | Speech-Recognition project — transcribes audio to text | Whisper and the `speech_recognition` library help the app understand spoken words. |
| | **OpenAI API** | LLMs coursework, LLM-Engineering app — chat and interview responses | This API calls a chat model like ChatGPT to answer questions. It lets our apps have conversations. |
| **Model Evaluation & Explainability** | **scikit-learn metrics** | Fraud-Detection MLOps — AUC-ROC, precision/recall calculations | These measurements show how well the fraud model works. |
| | **SHAP** | Fraud-Detection MLOps — global feature importance | SHAP tells us which features are most important for the model's decisions. |
| | **LIME** | Fraud-Detection MLOps — local explanation for single predictions | LIME explains why the model made a particular decision for one example. |
| | **Matplotlib / Seaborn** | Fraud-Detection MLOps — plotting feature importance and ROC/PR curves | These libraries draw charts to help us see model performance. |
| **Deployment & Serving** | **Flask** | Bike-Rental API — REST endpoint for predictions | Flask lets us build a small web server so outside programs can ask for predictions. |
| | **AWS SageMaker endpoints** | Fraud-Detection MLOps, GRC-LLM, Edenred Invoice Assistant — hosting trained models | SageMaker runs our trained models in the cloud so users can send requests and get answers. |
| | **AWS Lambda** | Digital-Value-Chain and Edenred Invoice Assistant — serverless backend functions | Lambda runs small pieces of code only when needed. This saves money because there is no always-running server. |
| | **AWS API Gateway** | Digital-Value-Chain and Invoice Assistant — routes HTTP requests to Lambda | API Gateway receives web requests and sends them to the right Lambda function. |
| | **AWS EC2** | Bike-Rental API deployment — hosts the REST service and runs CI tests | EC2 is a virtual machine in the cloud. We used it to run our bike-rental API in production. |
| | **Docker** | Bike-Rental project — containerizes the API for consistent deployment | Docker packages our app and its dependencies so it runs the same everywhere. |
| | **GitHub Actions** | Bike-Rental project — CI/CD pipeline for testing and deployment | GitHub Actions automatically tests code and deploys it when we push changes. |
| | **AWS SAM / CloudFormation** | Digital-Value-Chain — infrastructure as code for serverless stack | SAM and CloudFormation are templates that tell AWS how to build all the resources we need. |
| | **CloudWatch** | Edenred Invoice Assistant — monitoring and logging for Lambda | CloudWatch records logs and metrics so we can see what our Lambda functions are doing. |
| | **GitHub Pages** | Edenred Invoice Assistant — hosts the static chat interface | GitHub Pages serves our HTML and JavaScript files so users can access the chatbot in a browser. |
| | **Streamlit** | GRC-LLM and LLM-Engineering app — interactive web front-ends | Streamlit makes it easy to create a chat interface or dashboard from Python code. |
| | **Stripe** | Digital-Value-Chain — handles payment checkout | Stripe processes credit-card payments securely. |
| | **Boto3** | Digital-Value-Chain, GRC-LLM — Python SDK to access AWS services | Boto3 lets our Python code talk to AWS services like DynamoDB, S3 and SageMaker. |
| **DevOps & Infrastructure** | **Git** | All projects — version control and collaboration | Git keeps track of code changes and lets multiple people work together. |
| | **AWS IAM** | Fraud-Detection MLOps and Invoice Assistant — role-based access control | IAM is a permission system. It decides who can use which AWS resources. |
| | **Cost-optimization strategies** | Fraud-Detection MLOps and Edenred Invoice Assistant — turning off endpoints when idle | To save money, we shut down cloud resources when they are not being used and restart them only when needed. |
| **Front-end & User Interface** | **React 18 + Vite** | Digital-Value-Chain — modern, responsive e-commerce dashboard | React builds interactive web pages, and Vite makes development fast. |
| | **HTML / CSS / JavaScript** | Edenred Invoice Assistant — static chat interface | These are the basic building blocks of web pages. |
| | **Streamlit** | GRC-LLM and LLM-Engineering — simple Python web apps | Streamlit makes it easy to build a chat interface or dashboard from Python code. |
| **LLM Tools & Frameworks** | **LangChain / LangGraph** | LangChain & LangGraph coursework — chain and graph structures for LLMs | LangChain and LangGraph help build complex chat flows. They handle prompts, output parsing and memory. |
| | **OpenAI Chat models (ChatGPT/GPT-4)** | LLM coursework & LLM-Engineering app — used for text generation and interviews | These models chat with users, answer questions and conduct mock interviews. |
| | **PEFT / LoRA** | GRC-LLM — parameter-efficient fine-tuning | LoRA is a trick to train large models cheaply by adding small adapter layers. |

## 🎯 Portfolio Objectives

This repository serves multiple purposes:

### 🔬 **Research & Development**
Exploring cutting-edge ML techniques, experimenting with new algorithms and implementing research papers to stay current with the latest advancements in AI.

### 🏗️ **Production-Ready Solutions**
Building complete MLOps pipelines that demonstrate enterprise-level practices including automated testing, containerization, CI/CD, monitoring and scalable deployment strategies.

### 📚 **Learning & Growth**
Documenting my journey in machine learning, sharing knowledge through well-documented code and contributing to the ML community.

### 💼 **Professional Showcase**
Demonstrating practical skills in machine learning engineering, data science and AI system architecture for potential collaborators and employers.

## 🗂️ Featured Projects

### 🛡️ [GRC Compliance LLM - AI-Powered Compliance Assistant](./grc-llm-project/)

> **Production-Ready LoRA Fine-tuning with AWS SageMaker and Cost-Optimized Architecture**

Enterprise-grade compliance question-answering system that fine-tunes a TinyLlama 1.1B model using LoRA (Low-Rank Adaptation) for governance, risk and compliance queries across SOC 2, ISO 27001 and HIPAA frameworks.

**🎯 Highlights:**
- **LoRA Fine-tuning**: Parameter-efficient adaptation of TinyLlama 1.1B with 99% cost reduction vs full training
- **Ultra-fast Training**: 0.8-minute training time on AWS EC2 c5.2xlarge with comprehensive loss tracking
- **Production SageMaker**: Complete ML pipeline from training to endpoint deployment with model registry
- **100% Evaluation Accuracy**: Perfect performance on compliance-specific test dataset
- **Professional ChatGPT-style UI**: Streamlit interface with conversation history and response time tracking
- **Cost Engineering**: Strategic infrastructure management with instant reactivation capability
- **AI-Assisted Development**: Collaborative development with ChatGPT and Claude for rapid prototyping

**🛠️ Tech Stack:** TinyLlama, LoRA/PEFT, AWS SageMaker, Streamlit, PyTorch, Transformers, EC2, S3

```python
# Example Compliance Query
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load fine-tuned compliance model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(model, "outputs/compliance-tinyllama-lora")

# Ask compliance question
response = model.generate("Which SOC 2 control covers password requirements?")
# Output: "SOC 2 CC6.1 covers password requirements: organizations must implement complexity, length, and rotation policies."
```

**📊 Production Performance:**
- **Training Efficiency**: 0.8 minutes (loss: 2.3 → 2.09, 9% improvement)
- **Model Accuracy**: 100% success rate on compliance evaluation dataset
- **Response Quality**: Professional audit-ready answers with precise control mappings
- **Framework Coverage**: SOC 2, ISO 27001, HIPAA compliance queries
- **Infrastructure**: AWS SageMaker endpoint successfully deployed and validated
- **Cost Optimization**: 99%+ savings vs full model training through LoRA adapters
- **Business Value**: Instant compliance query resolution for audit preparation

### 🤖 [Edenred Invoice Assistant - Production AI Chatbot](./edenred-invoice-assistant/)

> **End-to-End ML Pipeline: Training to Production with Cost-Optimized AWS SageMaker**

Complete production-ready AI chatbot for invoice and payment support, showcasing enterprise-level ML deployment with intelligent cost management and serverless architecture.

**🎯 Highlights:**
- **Fine-tuned Language Model**: HuggingFace transformers on AWS SageMaker with custom training data
- **Cost-Optimized Architecture**: Intelligent fallback system with 90%+ cost reduction through smart resource management
- **Serverless Architecture**: AWS Lambda + API Gateway for auto-scaling with comprehensive monitoring
- **Production Frontend**: Modern responsive web interface with real-time chat functionality
- **Enterprise Integration**: CORS-enabled API with comprehensive logging and 100% uptime through fallback logic
- **Intelligent Responses**: Smart response patterns based on successful SageMaker model training

**🛠️ Tech Stack:** AWS SageMaker, Lambda, API Gateway, HuggingFace Transformers, Python, HTML/CSS/JS, CloudWatch

```python
# Example API Usage – Production Endpoint with Intelligent Fallbacks
import requests
response = requests.post(
    'https://zg4ja3aub5lvqzsbomo7nrhw7m0rjqms.lambda-url.us-east-1.on.aws/',
    json={'message': 'How do I submit an invoice?'}
)
print(f"AI Response: {response.json()['response']}")
```

**📊 Production Performance:**
- **Response Time**: <1 second average (optimized fallback system)
- **Accuracy Rate**: 95%+ on trained invoice/payment scenarios (pattern-based)
- **Uptime**: 100% with intelligent fallback handling
- **Cost Efficiency**: 90%+ reduction vs always-on SageMaker
- **Training Validation**: Complete ML pipeline with successful SageMaker fine-tuning
- **Demo**: [Live Interactive Chatbot](https://marcusmayo.github.io/machine-learning-portfolio/edenred-invoice-assistant/frontend/chatbot.html)

### 🚴 [Bike Rental Prediction - MLOps Pipeline](./bike_rental_prediction_fully_automated/)

> **Production-Ready ML System with Full CI/CD**

A complete end-to-end MLOps pipeline for predicting hourly bike rental demand, showcasing enterprise-level practices.

**🎯 Highlights:**
- **Neural Network Model**: PyTorch-based 3-layer feedforward network
- **Feature Engineering**: 53 engineered features from temporal and weather data
- **Production API**: Flask REST API deployed on AWS EC2
- **CI/CD Pipeline**: Automated testing, building and deployment via GitHub Actions
- **Containerization**: Docker-based deployment with AWS ECR
- **Real-time Predictions**: Sub-100 ms API response times

**🛠️ Tech Stack:** PyTorch, Flask, Docker, AWS (EC2, ECR), GitHub Actions, NumPy, Pandas

```python
# Example API Usage
import requests
response = requests.post('http://18.233.252.250/predict', json={'features': [0.1] * 53})
print(f"Predicted bike rentals: {response.json()['prediction']}")
```

### 🕵️ [Fraud Detection — Enterprise MLOps with Explainability](./fraud-detection-mlops/)

> **Production-Ready Fraud Detection with SHAP/LIME and Cost-Optimized SageMaker Pipeline**

Complete end-to-end MLOps pipeline for credit card fraud detection using AWS SageMaker, demonstrating enterprise-level practices with automated deployment, monitoring, model explainability and intelligent cost management for production-ready fraud prevention.

**🎯 Highlights:**
- **XGBoost Model**: Optimized gradient boosting with class imbalance handling (scale_pos_weight=100)
- **Time-Based Validation**: Chronological data splits with rolling backtests for temporal stability
- **Model Explainability**: SHAP global importance and LIME local explanations with comprehensive artifacts
- **Cost Engineering**: Strategic endpoint management with 95%+ operational cost reduction
- **SageMaker Pipeline**: Complete automated training, evaluation and deployment with model registry
- **Production Validation**: Successfully deployed and validated real-time endpoint with comprehensive evidence
- **Comprehensive Artifacts**: Complete explainability documentation for regulatory compliance

**🛠️ Tech Stack:** XGBoost, SageMaker, SHAP, LIME, Model Registry, CloudWatch, S3, Boto3

```python
# Example Production Pattern – Reactivation Ready
import boto3
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='fraud-detection-endpoint-1755128252',
    ContentType='text/csv',
    Body='0.5,-1.2,0.8,...'  # PCA features
)
result = json.loads(response['Body'].read())
print(f"Fraud probability: {result['probability']:.3f}")
print(f"Decision: {'FRAUD' if result['prediction'] > 0.5 else 'LEGITIMATE'}")
```

**📊 Production Performance:**
- **AUC-PR**: 0.7720 (precision-recall optimized for imbalanced data)
- **AUC-ROC**: 0.9763 (outstanding discrimination capability)
- **Dataset Scale**: 284,807 credit card transactions with 0.17% fraud rate
- **Response Time**: <100 ms real-time transaction processing (validated)
- **Cost Optimization**: 95%+ reduction with instant reactivation capability
- **Deployment Evidence**: Comprehensive artifacts documenting successful production validation
- **Regulatory Compliance**: Complete SHAP/LIME explainability documentation

### 🏢 [Digital Value Chain — Enterprise Serverless E-commerce](./digital-value-chain/)

> **Full-Stack Serverless Platform with Cost-Optimized Architecture and AI-Assisted Development**

Complete serverless e-commerce platform demonstrating enterprise-level architecture, modern development practices, intelligent cost management and scalable cloud solutions built collaboratively with AI assistants.

**🎯 Highlights:**
- **Modern Frontend**: React 18 + Vite with responsive design and comprehensive screenshot documentation
- **Cost-Optimized Serverless**: AWS Lambda + API Gateway with strategic resource management (95%+ cost reduction)
- **NoSQL Database**: DynamoDB integration with proper data modeling and production validation
- **Infrastructure as Code**: AWS SAM/CloudFormation with complete deployment evidence
- **Enterprise Architecture**: CORS configuration, error handling and production-ready security validation
- **AI-Augmented Development**: Collaborative problem-solving with ChatGPT and Claude demonstrating modern development workflows
- **Real-World Problem Solving**: Resolved 7+ major technical challenges with comprehensive documentation

**🛠️ Tech Stack:** React 18, AWS Lambda, API Gateway, DynamoDB, AWS SAM, Stripe, Vite, Python

```python
# Example API Usage – Production Endpoints (Reactivation Ready)
import requests
api_base = 'https://f59moopdx0.execute-api.us-east-1.amazonaws.com'
# List all offers
print(requests.get(f'{api_base}/offers').json())
# Create a new offer
print(requests.post(f'{api_base}/offers', json={'sku': 'premium-001', 'name': 'Premium Plan', 'price': 99.99}).json())
```

**📊 Production Performance & Evidence:**
- **Frontend**: React dashboard documented via comprehensive screenshots (`dashboard-*.png`)
- **API**: REST endpoints validated (`http://18.232.96.171:5174`, `api-health.png` evidence)
- **Architecture**: Auto-scaling serverless with intelligent cost optimization (95%+ savings)
- **Database**: DynamoDB with proper NoSQL design patterns and production validation
- **Infrastructure**: Complete CloudFormation deployment with comprehensive screenshot evidence
- **Cost Engineering**: Strategic EC2 management with instant reactivation capability
- **Enterprise Ready**: CORS-enabled, error handling and monitoring integration documented
- **Business Application**: Ideal for digital marketplaces, partner portals and B2B platforms

### 🎭 [Sentiment Analysis Web App](./sentiment_analysis_webapp/) *(Coming Soon)*

> **Real-time sentiment analysis with modern transformers**

Web application for analyzing sentiment in text using Hugging Face Transformers, deployed as a scalable REST API.

**Planned Features:**
- Hugging Face Transformers integration
- Flask/FastAPI web framework
- Real-time sentiment prediction
- EC2 cloud hosting with auto-scaling
- Support for IMDb Reviews and Twitter datasets

### 🖼️ [Image Classifier on CIFAR-10](./cifar10_classifier/) *(Coming Soon)*

> **CNN-based image classification with MLflow tracking**

Deep learning image classifier using PyTorch CNNs with comprehensive model tracking and cloud storage integration.

**Planned Features:**
- Custom CNN architecture in PyTorch
- MLflow experiment tracking and model versioning
- S3 storage for model artifacts
- CIFAR-10 dataset with data augmentation
- Performance benchmarking and visualization

### 📈 [Time Series Forecasting (Weather/Energy)](./time_series_forecasting/) *(Coming Soon)*

> **LSTM-based forecasting with automated scheduling**

Stay tuned for more exciting projects!

---

## 📫 Get In Touch

- **LinkedIn**: [Connect with me](https://linkedin.com/in/marcusmayo)
- **Email**: marcusmayo@hotmail.com
- **Portfolio**: [Live Projects](https://github.com/marcusmayo/machine-learning-portfolio)

---

⭐ **Star this repository if you find it helpful!** Your support motivates me to keep building and sharing innovative ML solutions.
