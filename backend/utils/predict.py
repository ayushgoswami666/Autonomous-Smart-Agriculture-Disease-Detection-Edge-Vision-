import torch
import os
from utils.preprocess import transform
from data.class_name import CLASSES
from model.model_loader import load_model

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "model", "plant_disease_vgg16.pth")
    if not os.path.exists(model_path):
        return None
    try:
        _cached_model = load_model()
        return _cached_model
    except Exception:
        return None

def predict(image):
    model = get_model()
    if model is None:
        return None

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        top_k = min(3, len(CLASSES))
        top_prob, top_catid = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_prob.size(0)):
            results.append({
                "class": CLASSES[top_catid[i].item()],
                "confidence": top_prob[i].item()
            })

    return results