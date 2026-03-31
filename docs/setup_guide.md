# Setup Guide

## Prerequisites

- Python 3.9+
- pip
- (For edge deployment) Raspberry Pi 4 with Raspberry Pi OS

---

## 1. Clone Repository

```bash
git clone https://github.com/ayushgoswami666/Autonomous-Smart-Agriculture-Disease-Detection-Edge-Vision-.git
cd Autonomous-Smart-Agriculture-Disease-Detection-Edge-Vision-
git checkout ashwani
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For Raspberry Pi (edge only):
```bash
pip install tflite-runtime
```

---

## 4. Download Dataset

Download the **PlantVillage** dataset:
- Source: https://www.kaggle.com/datasets/emmarex/plantdisease
- Place extracted dataset in `data/raw/`

Expected structure:
```
data/raw/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___healthy/
├── Potato___Early_blight/
...
```

---

## 5. Preprocess Data

```bash
python scripts/preprocess.py --input data/raw/ --output data/processed/ --augment
```

---

## 6. Train the Model

```bash
python src/model/train.py --dataset data/processed/ --epochs 20 --model mobilenetv2
```

This saves:
- `models/disease_model.h5` — Keras model
- `models/class_names.npy` — class label array
- `models/training_plot.png` — accuracy/loss curves

---

## 7. Start API Server

```bash
python src/api/app.py
```

API runs at `http://localhost:5000`

Test health:
```bash
curl http://localhost:5000/health
```

---

## 8. Start Dashboard

In a new terminal:
```bash
streamlit run src/frontend/app.py
```

Dashboard opens at `http://localhost:8501`

---

## 9. Edge Deployment (Raspberry Pi)

### Convert model to TFLite
```bash
python src/edge/inference.py --convert models/disease_model.h5 --model models/disease_model.tflite
```

### Copy files to Raspberry Pi
```bash
scp models/disease_model.tflite pi@<PI_IP>:/home/pi/agri/models/
scp models/class_names.npy pi@<PI_IP>:/home/pi/agri/models/
scp src/edge/inference.py pi@<PI_IP>:/home/pi/agri/
```

### Run on Raspberry Pi
```bash
# Single image test
python inference.py --model models/disease_model.tflite --image test_leaf.jpg

# Live camera loop
python inference.py --model models/disease_model.tflite --camera
```

---

## 10. Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## Environment Variables

Create a `.env` file in the project root:
```
FLASK_ENV=development
MODEL_PATH=models/disease_model.h5
CLASSES_PATH=models/class_names.npy
API_PORT=5000
```
