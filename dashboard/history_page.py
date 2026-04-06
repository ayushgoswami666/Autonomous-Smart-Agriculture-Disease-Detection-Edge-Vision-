import streamlit as st

# This page shows all previous disease detections
# Farmers can look back at past results and track their crop health

st.set_page_config(page_title="Detection History")
st.title("Detection History")
st.write("Here you can see all your previous crop disease checks.")

# Sample history data to show how it will look
sample_history = [
    {"date": "April 05 2026", "crop": "Tomato", "disease": "Early Blight", "result": "High"},
    {"date": "April 03 2026", "crop": "Potato", "disease": "Late Blight", "result": "Medium"},
    {"date": "April 01 2026", "crop": "Tomato", "disease": "Healthy", "result": "Low"},
]

# Show each record in a simple table
st.subheader("Your Past Detections")
for record in sample_history:
    st.write("Date: " + record["date"])
    st.write("Crop: " + record["crop"])
    st.write("Disease: " + record["disease"])
    st.write("Severity: " + record["result"])
    st.write("---")

st.info("New detections will be saved here automatically after each check.")
