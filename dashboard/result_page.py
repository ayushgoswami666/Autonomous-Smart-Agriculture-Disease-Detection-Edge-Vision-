import streamlit as st

def show_result(disease, confidence, severity, treatment):
    st.subheader("Detection Result")
    st.write("Disease Found: " + str(disease))
    st.write("AI Confidence: " + str(confidence) + "%")
    st.progress(confidence / 100)

    if severity == "high":
        st.error("Severity: HIGH - Act immediately")
    elif severity == "medium":
        st.warning("Severity: MEDIUM - Monitor closely")
    else:
        st.success("Severity: LOW - Keep watching")

    st.subheader("What to do")
    for step in treatment:
        st.write("- " + step)
