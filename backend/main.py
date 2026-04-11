import sys
import os

# Add the backend directory to sys.path so utils, model, data can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from utils.predict import predict
from utils.agri_cure import AGRI_CURE
from auth import router as auth_router, get_current_user

app = FastAPI(title="Agri-Vision Edge API")

# Setup CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

@app.get("/")
def read_root():
    return {"status": "Edge Node Online"}

from fastapi import Depends

@app.post("/api/predict")
async def run_prediction(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    
    results = predict(image)
    if results is None:
        raise HTTPException(status_code=500, detail="Model core plant_disease_vgg16.pth not found locally.")
    
    top_prediction = results[0]
    top_class = top_prediction['class']
    top_conf = top_prediction['confidence']
    
    cure_info = AGRI_CURE.get(top_class, None)
    
    return {
        "predictions": results,
        "top_class": top_class,
        "top_confidence": top_conf,
        "treatment": cure_info
    }
