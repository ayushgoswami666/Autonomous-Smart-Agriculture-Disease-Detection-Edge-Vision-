import streamlit as st
from PIL import Image
from utils.predict import predict

st.title("🌿 Plant Disease Detection Dashboard")

st.write("Upload a plant leaf image to detect disease")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    disease, confidence = predict(image)

    st.success(f"Prediction: {disease}")

    st.info(f"Confidence: {confidence*100:.2f}%")