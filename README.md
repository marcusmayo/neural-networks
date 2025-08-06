# 🚲 Bike Rental Prediction with Neural Network & MLOps Pipeline

This project demonstrates how to build, train, test, and deploy a **simple neural network** using **PyTorch**, and automate the entire ML lifecycle with an **MLOps pipeline** using **GitHub Actions**, **Docker**, and **FastAPI**.

---

## 🧠 Project Summary

We use a feedforward neural network (multi-layer perceptron) to predict **bike rental counts** based on hourly weather and time features using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

---

## 📦 What's Included

| Folder / File | Purpose |
|---------------|---------|
| `src/`        | Preprocessing, model definition, training, and testing scripts |
| `app/`        | FastAPI server to deploy the trained model as an API |
| `models/`     | Stores the trained PyTorch model |
| `data/`       | Placeholder for dataset file `hour.csv` |
| `Dockerfile`  | Containerizes the FastAPI app |
| `requirements.txt` | Lists required Python packages |
| `.github/workflows/train.yml` | GitHub Actions workflow for full automation |

---

## ⚙
