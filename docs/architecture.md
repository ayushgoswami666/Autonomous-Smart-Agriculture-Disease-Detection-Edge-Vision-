# System Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────────┐
│                     FIELD LAYER                          │
│                                                          │
│  [Camera Module] ──→ [Raspberry Pi 4 (Edge Device)]      │
│                           │                              │
│                    [TFLite Inference]                     │
│                    ≤ 3 seconds / image                   │
└───────────────────────────┬──────────────────────────────┘
                            │ (local network / USB)
┌───────────────────────────▼──────────────────────────────┐
│                   APPLICATION LAYER                       │
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────┐     │
│  │  Flask REST API │    │  Streamlit/React         │     │
│  │  (Backend)      │◄──►│  Web Dashboard           │     │
│  │  port :5000     │    │  port :8501              │     │
│  └────────┬────────┘    └──────────────────────────┘     │
│           │                                              │
│  ┌────────▼────────┐    ┌──────────────────────────┐     │
│  │  Image          │    │  Recommendation          │     │
│  │  Processing     │    │  Engine                  │     │
│  │  Service        │    │  (Treatment DB)          │     │
│  └────────┬────────┘    └──────────────────────────┘     │
└───────────┼──────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│                      ML LAYER                            │
│                                                          │
│  [CNN Model — MobileNetV2]                               │
│  Input:  224×224 RGB image                               │
│  Output: 38-class softmax (PlantVillage classes)         │
│  Format: .h5 (server) / .tflite (edge)                   │
└───────────┬──────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│                     DATA LAYER                           │
│                                                          │
│  data/uploads/   — uploaded leaf images                  │
│  data/history.json — detection records                   │
│  models/         — trained model files                   │
└──────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. Camera Module
- USB camera or Raspberry Pi Camera Module v2
- Captures leaf images at the field
- Triggers inference on capture

### 2. Edge Device (Raspberry Pi 4)
- Runs `src/edge/inference.py` with TFLite model
- Inference target: ≤ 3 seconds per image
- Operates offline — no cloud required for detection

### 3. Flask REST API (`src/api/app.py`)
- Handles image upload, disease detection, recommendations, history
- Loads trained Keras model once at startup
- Persists detection records to `data/history.json`

### 4. CNN Disease Detection Model (`src/model/`)
- Base: MobileNetV2 (ImageNet pre-trained, fine-tuned on PlantVillage)
- Input: 224×224 RGB
- Output: 38 disease class probabilities
- Training accuracy target: ≥ 85%

### 5. Streamlit Dashboard (`src/frontend/app.py`)
- Image upload UI
- Displays disease name, confidence score, severity
- Shows treatment and prevention recommendations
- Detection history view

### 6. Recommendation Engine (`src/api/recommendations.py`)
- Static knowledge base of disease → treatment mappings
- Covers tomato, potato, corn diseases
- Returns cause, symptoms, treatment steps, prevention tips

---

## Data Flow

```
User uploads image
      │
      ▼
POST /api/image/upload  →  Save to data/uploads/{uuid}.jpg
      │
      ▼
POST /api/detect        →  Load image → Preprocess → CNN Inference
      │
      ▼
Return disease, confidence, severity, top5
      │
      ▼
GET /api/recommendation?disease=X  →  Lookup treatment DB
      │
      ▼
Dashboard renders result + treatment to user
```

---

## Model Architecture

```
Input (224×224×3)
      │
MobileNetV2 (frozen base — ImageNet weights)
      │
GlobalAveragePooling2D
      │
BatchNormalization
      │
Dense(256, relu)
      │
Dropout(0.4)
      │
Dense(38, softmax)  →  38 disease class probabilities
```

### Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Image Size | 224×224 |
| Val Split | 20% |
| Early Stopping | patience=5 |
| LR Reduction | factor=0.5, patience=3 |

---

## Edge Optimization

For Raspberry Pi deployment, the Keras `.h5` model is converted to TFLite:

```
MobileNetV2 .h5  →  TFLite Converter (DEFAULT optimization)  →  .tflite
~14 MB                                                          ~4–6 MB
```

TFLite uses post-training dynamic range quantization, reducing model size and improving inference speed on ARM hardware.
