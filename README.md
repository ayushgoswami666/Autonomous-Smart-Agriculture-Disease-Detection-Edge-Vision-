# 🌿 Autonomous Smart Agriculture Disease Detection System
### Edge Vision + Minimal Sensors | Team Winters (T-66)

> **GLA University | B.Tech CS (AIML & IIoT) | Version v1.0 | Jan 2026**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Edge AI](https://img.shields.io/badge/Edge%20AI-Raspberry%20Pi-red?logo=raspberry-pi)
![Domain](https://img.shields.io/badge/Domain-AgriTech-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📌 Problem Statement

Crop diseases often spread rapidly before farmers can detect them. Manual inspection is:
- Slow and inaccurate
- Requires expert knowledge
- Leads to excessive pesticide use and yield loss
- High cost of existing solutions

## 🎯 Solution

An AI-powered smart agriculture system that uses **Computer Vision + Edge AI** to:
- Detect crop diseases **early** from leaf images
- Run inference **on-device** (Raspberry Pi) — no cloud required
- Provide **targeted treatment recommendations**
- Support farmers via a **GenAI chatbot (RAG-based)**

---

## 🏗️ Architecture Overview

```
[Camera Module]
      ↓
[Image Preprocessing]
      ↓
[CNN Disease Detection Model]  ←── Trained on PlantVillage Dataset
      ↓
[Edge Inference (Raspberry Pi)]
      ↓
[Web Dashboard (Streamlit/React)]
      ↓
[Treatment Recommendation Engine]
```

---

## 🚀 Key Features

| Feature | Description | Priority |
|---|---|---|
| 📷 Image Capture & Upload | Capture or upload crop leaf images | Must |
| 🧠 CNN Disease Detection | Classify diseases using AI model (≥85% accuracy) | Must |
| ⚡ Edge AI Inference | On-device prediction in ≤3 seconds | Must |
| 💊 Treatment Recommendation | Disease-specific treatment guidance | Must |
| 📊 Result Visualization | Confidence scores & severity display | Should |
| 📜 Detection History | Store & view past detection records | Could |

---

## 👥 Team Composition

| Member | Role | Responsibilities |
|---|---|---|
| Ayush | Product Lead | Scope, requirements, mentor coordination |
| Kush | Tech Lead & ML Backend | Model architecture, CNN training, backend |
| Ashwani | Frontend & Integration | Dashboard UI, model integration, testing, docs |

---

## 🛠️ Tech Stack

- **ML Framework**: TensorFlow / Keras
- **Model**: CNN (MobileNetV2 / custom)
- **Backend**: Flask / FastAPI
- **Frontend**: Streamlit / React
- **Edge Device**: Raspberry Pi 4
- **Dataset**: PlantVillage (38 disease classes)
- **Language**: Python 3.9+

---

## 📁 Repository Structure

```
├── src/
│   ├── model/          # CNN model training & evaluation
│   ├── api/            # Flask/FastAPI backend services
│   ├── frontend/       # Streamlit/React dashboard
│   └── edge/           # Raspberry Pi edge deployment
├── data/
│   ├── raw/            # Raw dataset (gitignored)
│   └── processed/      # Preprocessed images
├── models/             # Saved trained models (.h5 / TFLite)
├── notebooks/          # Jupyter notebooks for experiments
├── tests/              # Unit and integration tests
├── scripts/            # Utility scripts
├── docs/               # Project documentation & diagrams
└── .github/workflows/  # CI/CD pipelines
```

---

## ⚙️ Quick Start

### Prerequisites
```bash
Python 3.9+
pip install -r requirements.txt
```

### Run Training
```bash
python src/model/train.py --epochs 20 --dataset data/processed/
```

### Run API Server
```bash
python src/api/app.py
```

### Run Dashboard
```bash
streamlit run src/frontend/app.py
```

### Run on Edge (Raspberry Pi)
```bash
python src/edge/inference.py --model models/disease_model.tflite
```

---

## 📊 Success Metrics

| Objective | KPI | Target |
|---|---|---|
| Model Accuracy | Validation accuracy | ≥ 85% |
| Inference Speed | p95 detection time on edge | ≤ 3 seconds |
| Detection Reliability | Correct classification rate | ≥ 80% |
| Usability | Critical UI defects | 0 |

---

## 📅 Project Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1 | Jan 28 – Feb 3 | Requirements Freeze |
| 2 | Feb 4 – Feb 10 | Architecture & Setup |
| 3–4 | Feb 11 – Feb 24 | Data Prep + Model Training |
| 5–6 | Feb 25 – Mar 10 | Optimization + Edge Deployment |
| 7–8 | Mar 11 – Mar 24 | Testing & Bug Fixes |
| 9–10 | Mar 25 – Apr 7 | Documentation + Final Demo |

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/image/upload` | POST | Upload crop leaf image |
| `/api/detect` | POST | Detect crop disease |
| `/api/recommendation` | GET | Fetch treatment recommendation |
| `/api/history` | GET | View detection history |

---

## 📄 Documentation

- [Full Project Synopsis](docs/synopsis.md)
- [System Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Setup Guide](docs/setup_guide.md)
- [Testing Guide](docs/testing_guide.md)

---

## 📜 License

Academic project — GLA University, 2026. All rights reserved.

---

> **Mentor**: Mrs. Chavi Bajpai | **Branch**: `ashwani` (Frontend, Integration & QA)
