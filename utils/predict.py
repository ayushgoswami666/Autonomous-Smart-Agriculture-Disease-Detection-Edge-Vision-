import torch
import streamlit as st
from utils.preprocess import transform
from data.class_name import CLASSES
from model.model_loader import load_model

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

def predict(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        conf, pred = torch.max(probabilities,1)

    return CLASSES[pred.item()], conf.item()