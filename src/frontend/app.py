"""
app.py — Streamlit Web Dashboard
Team Winters (T-66) | Autonomous Smart Agriculture Disease Detection

Run with:
    streamlit run src/frontend/app.py
"""

import streamlit as st
import requests
import json
from PIL import Image
import io

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:5000"
PAGE_ICON = "🌿"

st.set_page_config(
    page_title="AgriDisease Detector",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #2e7d32; }
    .result-card { background: #f1f8e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #66bb6a; }
    .warning-card { background: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffa726; }
    .danger-card  { background: #ffebee; padding: 1rem; border-radius: 8px; border-left: 4px solid #ef5350; }
    .metric-box { text-align: center; padding: 1rem; border-radius: 8px; background: #e8f5e9; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-rain.png", width=80)
    st.title("🌿 AgriDisease Detector")
    st.markdown("**Team Winters (T-66)**")
    st.markdown("GLA University | B.Tech CS (AIML & IIoT)")
    st.divider()

    page = st.radio("Navigate", [
        "🔍 Detect Disease",
        "📊 Detection History",
        "ℹ️ About",
    ])


# ── Page: Detect Disease ──────────────────────────────────────────────────────
if page == "🔍 Detect Disease":
    st.markdown('<p class="main-header">🌿 Crop Disease Detection</p>', unsafe_allow_html=True)
    st.markdown("Upload a clear image of a crop leaf to detect diseases and get treatment recommendations.")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Upload Leaf Image")
        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Best results with clear, well-lit images of single leaves",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🧠 Detection Result")

        if uploaded and st.button("🔍 Detect Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf image..."):
                try:
                    # Step 1: Upload image
                    files    = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp     = requests.post(f"{API_BASE}/api/image/upload", files=files, timeout=10)
                    resp.raise_for_status()
                    image_id = resp.json()["imageId"]

                    # Step 2: Detect disease
                    detect_resp = requests.post(
                        f"{API_BASE}/api/detect",
                        json={"imageId": image_id},
                        timeout=15,
                    )
                    detect_resp.raise_for_status()
                    result = detect_resp.json()

                    disease    = result["disease"]
                    confidence = result["confidence"]
                    severity   = result["severity"]

                    # Display result
                    severity_color = {"low": "result-card", "medium": "warning-card", "high": "danger-card"}
                    card_class     = severity_color.get(severity, "result-card")

                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>🦠 {disease.replace("___", " — ").replace("_", " ")}</h3>
                        <p><strong>Confidence:</strong> {confidence * 100:.1f}%</p>
                        <p><strong>Severity:</strong> {severity.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence bar
                    st.progress(confidence)

                    # Top 5
                    if "top5" in result:
                        with st.expander("📈 Top 5 Predictions"):
                            for name, prob in result["top5"]:
                                label = name.replace("___", " — ").replace("_", " ")
                                st.write(f"**{label}** — {prob * 100:.1f}%")
                                st.progress(prob)

                    # Fetch recommendation
                    rec_resp = requests.get(
                        f"{API_BASE}/api/recommendation",
                        params={"disease": disease},
                        timeout=5,
                    )
                    if rec_resp.status_code == 200:
                        rec = rec_resp.json()
                        st.divider()
                        st.subheader("💊 Treatment Recommendations")
                        st.write(f"**Cause:** {rec.get('cause', 'N/A')}")
                        st.write(f"**Symptoms:** {rec.get('symptoms', 'N/A')}")

                        st.write("**Treatment Steps:**")
                        for step in rec.get("treatment", []):
                            st.write(f"• {step}")

                        st.write("**Prevention:**")
                        for tip in rec.get("prevention", []):
                            st.write(f"✅ {tip}")

                except requests.ConnectionError:
                    st.error("❌ Cannot connect to API server. Make sure `src/api/app.py` is running.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        elif not uploaded:
            st.info("👆 Upload a leaf image to get started")


# ── Page: History ─────────────────────────────────────────────────────────────
elif page == "📊 Detection History":
    st.markdown('<p class="main-header">📊 Detection History</p>', unsafe_allow_html=True)
    st.divider()

    try:
        resp = requests.get(f"{API_BASE}/api/history", timeout=5)
        resp.raise_for_status()
        records = resp.json().get("records", [])

        if not records:
            st.info("No detection records yet. Run a detection first!")
        else:
            st.metric("Total Detections", len(records))
            for rec in reversed(records[-20:]):   # Latest 20
                disease    = rec.get("disease", "").replace("___", " — ").replace("_", " ")
                confidence = rec.get("confidence", 0)
                severity   = rec.get("severity", "low")
                timestamp  = rec.get("timestamp", "")

                st.write(f"**{disease}** — {confidence * 100:.1f}% confidence | Severity: {severity} | {timestamp[:16]}")
            st.caption("Showing latest 20 records")

    except requests.ConnectionError:
        st.error("❌ Cannot connect to API server.")


# ── Page: About ───────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    ### 🌿 Autonomous Smart Agriculture Disease Detection System

    A low-cost, AI-powered system for early crop disease detection using **Edge Vision AI + Minimal Sensors**.

    | Detail | Info |
    |---|---|
    | **Team** | Team Winters (T-66) |
    | **Institute** | GLA University |
    | **Course** | B.Tech CS (AIML & IIoT) |
    | **Mentor** | Mrs. Chavi Bajpai |
    | **Version** | v1.0 — Jan 2026 |

    ### 🛠️ Tech Stack
    - **ML Model**: CNN (MobileNetV2) — TensorFlow/Keras
    - **Edge Device**: Raspberry Pi 4
    - **Backend**: Flask API
    - **Frontend**: Streamlit Dashboard
    - **Dataset**: PlantVillage (38 disease classes)

    ### 👥 Team
    - **Ayush** — Product Lead
    - **Kush** — Tech Lead & ML Backend
    - **Ashwani** — Frontend & Integration
    """)
