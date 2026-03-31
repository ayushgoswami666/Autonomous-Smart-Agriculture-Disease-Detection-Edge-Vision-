# Project Synopsis
## Autonomous Smart Agriculture Disease Detection System using Edge AI and Computer Vision

**Team Winters (T-66) | GLA University | B.Tech CS (AIML & IIoT) | v1.0 | Jan 28, 2025**

---

## 0. Cover

| Detail | Info |
|---|---|
| Project Title | Autonomous Smart Agriculture Disease Detection System using Edge AI and Computer Vision |
| Team Name & ID | Team Winters (T-66) |
| Institute / Course | GLA University / B.Tech CS (AIML & IIoT) |
| Version | v1.0 |
| Date | 28 Jan 2026 |
| Mentor | Mrs. Chavi Bajpai |

---

## 1. Overview

### Problem Statement
Crop diseases often spread rapidly and remain undetected in early stages due to reliance on manual inspection. Traditional methods are time-consuming, require expert knowledge, and are prone to human error. Late detection leads to excessive pesticide usage, reduced yield, increased costs, and economic losses — especially for rural farmers.

### Goal
Build an AI-powered smart agriculture disease detection system using computer vision and edge AI to identify crop diseases at an early stage. The system analyzes plant leaf images in real time and provides accurate disease classification with actionable treatment recommendations.

### Non-Goals (v1)
- Automated pesticide spraying or robotic intervention
- Advanced soil nutrient analysis and weather-based prediction models
- Large-scale commercial deployment and government integration

### Value Proposition
AI-driven disease detection on low-cost edge devices enables early identification directly in the field — helping farmers reduce pesticide use, improve yield, lower costs, and make informed decisions without expert-level agricultural knowledge.

---

## 2. Scope and Control

### 2.1 In Scope
- Image-based crop disease detection using computer vision
- Dataset preparation and preprocessing of crop leaf images
- Training and deployment of a CNN-based disease classification model
- Edge device implementation (Raspberry Pi) for on-field detection
- Web-based dashboard to display results and confidence scores
- Disease-wise treatment and prevention recommendations
- Basic system analytics (detection accuracy, inference time)

### 2.2 Out of Scope
- Automated pesticide spraying or robotic intervention
- Advanced soil nutrient analysis and weather prediction
- Mobile application development in v1 (web dashboard only)
- Large-scale commercial or government integration

### 2.3 Assumptions
- Farmers/operators can capture clear images of crop leaves
- Dataset contains representative disease samples
- Edge devices have sufficient computational capability
- Internet connectivity available for dashboard and model updates where required

### 2.4 Constraints
- Development timeline: 8–10 weeks
- Limited hardware resources on edge devices
- Accuracy depends on image quality and dataset diversity
- Team skill level: Beginner to intermediate in Python, ML, and CV

### 2.5 Acceptance Criteria
- GIVEN a crop leaf image, WHEN the system processes it, THEN disease is detected with >85% accuracy
- GIVEN the model runs on edge device, WHEN inference is performed, THEN results display within 3–5 seconds

---

## 3. Stakeholders and RACI

| Activity | Responsible (R) | Accountable (A) | Consulted (C) | Informed (I) |
|---|---|---|---|---|
| Requirements | Kush | Kush | Mentor | Team |
| Design | Ayush | Kush | Mentor | Team |
| Implementation | Ayush, Ashwani | Kush | — | Team |
| Testing | Ashwani | Kush | Mentor | Team |
| Release | Ashwani | Kush | Mentor | Dept |

---

## 4. Team Composition

| Member | Role | Responsibilities | Key Skills | Availability |
|---|---|---|---|---|
| Ayush | Product Lead | Scope, requirements, mentor coordination | Product planning, APIs, docs | 10 hrs/wk |
| Kush | Tech Lead & ML Backend | System architecture, ML model, backend | Python, TensorFlow, CNN, Flask | 10 hrs/wk |
| Ashwani | Frontend & Integration | Dashboard UI, integration, testing, docs | Streamlit, React, Python | 10 hrs/wk |

---

## 5. Week-wise Plan

| Week | Dates | Milestone | Deliverables |
|---|---|---|---|
| 1 | Jan 28 – Feb 3, 2026 | Requirements Freeze | Problem definition, dataset plan |
| 2 | Feb 4 – Feb 10 | Architecture & Setup | System architecture, tech stack |
| 3 | Feb 11 – Feb 17 | Data Preparation | Cleaned dataset |
| 4 | Feb 18 – Feb 24 | Model Development | Trained model v1 |
| 5 | Feb 25 – Mar 3 | Model Optimization | Optimized model |
| 6 | Mar 4 – Mar 10 | Edge Deployment | Edge inference demo |
| 7 | Mar 11 – Mar 17 | Testing & Validation | Test report |
| 8 | Mar 18 – Mar 24 | Hardening & Fixes | Stable build |
| 9 | Mar 25 – Mar 31 | Documentation | Final documentation |
| 10 | Apr 1 – Apr 7 | Final Demo & Submission | Final submission & demo |

---

## 6. Users and UX

### Personas
- **Farmer Ravi**: Small/medium-scale farmer wanting early disease detection via a simple, affordable system
- **Agri Technician Neha**: Agricultural technician monitoring multiple crops, seeking accurate classification and history

### Key User Journeys
- Farmer: Home → Upload Image → Disease Detection → View Result → Read Treatment (≤4 steps, ≥85% success)
- Technician: Home → Upload Images → View Classification → Confidence Score → Review History (p95 ≤3s)

---

## 7. Market and Competitors

| Competitor | Product | Weakness | Our Differentiator |
|---|---|---|---|
| Plantix | Mobile Crop Advisory | Requires internet, mobile-only | Edge-based, offline-capable |
| AgroStar | Digital Agriculture Platform | No real-time disease detection | AI-powered vision detection |
| Kisan Suvidha | Govt. Agri App | No AI-based disease detection | Focused, accurate CV detection |
| Krushi Doctor | Agri Advisory App | Manual diagnosis, slower | Instant field-level AI diagnosis |

**Unique Angle**: Affordable, AI-powered, edge-deployed — early diagnosis in the field without continuous internet.

---

## 8. Objectives and Success Metrics

| Objective | KPI | Target |
|---|---|---|
| O1 — Model Readiness | Validation accuracy | ≥ 85% by Feb 15, 2026 |
| O2 — Inference Performance | p95 edge detection time | ≤ 3,000 ms by Mar 10, 2026 |
| O3 — Detection Reliability | Correct classification rate | ≥ 80% by Mar 20, 2026 |
| O4 — Usability | Critical UI defects | 0 by final release |

---

## 9. Key Features

| Feature | Priority | Acceptance Criteria |
|---|---|---|
| Image Capture & Upload | Must | Image accepted and processed successfully |
| Disease Detection Model | Must | ≥85% accuracy on leaf images |
| Edge AI Inference | Must | Result displayed within ≤3 seconds |
| Treatment Recommendation | Must | Relevant treatment shown per detected disease |
| Result Visualization | Should | Confidence score and disease details visible |
| Detection History | Could | Past records accessible |

---

## 10. Architecture

**Clients**: Web dashboard (Streamlit/React) for image upload, results, recommendations

**Services**:
- Image Processing Service (preprocessing & augmentation)
- Disease Detection Service (CNN-based AI model)
- Recommendation Service (treatment & prevention advice)
- Edge Inference Service (on-device prediction)

**Data Stores**: Local/cloud storage for images and results; model storage

**Edge Integration**: Camera module → Raspberry Pi → Dashboard

---

## 11. Quality: NFRs

| Metric | Target |
|---|---|
| Availability | ≥ 98.0% uptime |
| Latency (p95 detection) | ≤ 3,000 ms |
| Latency (p95 recommendation) | ≤ 2,000 ms |
| Prediction Error Rate | ≤ 2% |
| Open Critical Vulnerabilities | 0 |

---

## 12. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Low model accuracy | Medium | High | Dataset augmentation, hyperparameter tuning |
| Poor image quality | Medium | Medium | User guidance, preprocessing filters |
| Edge device limitation | Low | Medium | Model optimization (TFLite) |
| Schedule delay | Low | Medium | Weekly milestones, Ayush tracking |

---

## 13. Glossary

- **CNN** — Convolutional Neural Network
- **Edge AI** — AI inference running on-device without cloud dependency
- **DFD** — Data Flow Diagram
- **ERD** — Entity Relationship Diagram
- **TFLite** — TensorFlow Lite (optimized for edge devices)
- **RAG** — Retrieval-Augmented Generation

---

*Document prepared by: Ashwani Chauhan | Team Winters (T-66) | GLA University*
