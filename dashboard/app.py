import streamlit as st

st.set_page_config(page_title="Crop Disease Detector")
st.title("Smart Crop Disease Detection")
st.write("Upload a photo of your crop leaf and we will check for disease.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png"])

if uploaded_file:
    st.image(uploaded_file, caption="Your uploaded leaf", use_column_width=True)
    if st.button("Check for Disease"):
        st.info("Analyzing your leaf... please wait")
        st.success("Result will appear here after model is connected")
