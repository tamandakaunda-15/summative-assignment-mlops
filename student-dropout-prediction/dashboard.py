# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

API_URL = "https://summative-assignment-mlops.onrender.com/metrics"

st.set_page_config(page_title="Student Engagement Predictor", layout="wide")
st.title("🎓 Student Engagement Predictor Dashboard")

# Sidebar: Upload image for prediction
st.sidebar.header("Upload Image for Prediction")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.sidebar.button("Predict"):
        # Send to /predict endpoint
        with st.spinner("Sending image to API for prediction..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(f"{API_URL}/predict", files={"file": uploaded_file})
                result = response.json()
                if response.status_code == 200:
                    st.subheader("Prediction Results")
                    st.write(result["message"])

                    for pred in result["predictions"]:
                        st.write(f"🧠 **Status**: {pred['engagement_status']} | Score: {pred['prediction_score']:.2f}")

                    st.image(
                        Image.open(io.BytesIO(base64.b64decode(result["image_base64"]))),
                        caption="Analyzed Image",
                        use_column_width=True
                    )
                else:
                    st.error(f"Error: {result.get('error')}")
            except Exception as e:
                st.error(f"API Error: {e}")

# Dashboard tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Metrics", "📈 Model Performance", "⚙️ API Status"])

with tab1:
    st.subheader("Live Engagement Metrics")
    try:
        response = requests.get(f"{API_URL}/metrics")
        data = response.json()
        st.metric("Total Predictions", data["total_predictions"])
        st.metric("Engaged Count", data["engaged_count"])
        st.metric("Not Engaged Count", data["not_engaged_count"])

        st.write("Recent Predictions:")
        for item in data["recent_predictions"]:
            st.write(f"{item['timestamp']}: {item['status']}")

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(
            [data["engaged_count"], data["not_engaged_count"]],
            labels=["Engaged", "Not Engaged"],
            autopct="%1.1f%%",
            colors=["green", "red"]
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")

with tab2:
    st.subheader("Model Evaluation Metrics")
    try:
        response = requests.get(f"{API_URL}/model_metrics")
        result = response.json()

        st.write(f"**Accuracy**: {result['accuracy']:.2f}")
        st.write(f"**Loss**: {result['loss']:.3f}")

        history = result["training_history"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history["epochs"], history["accuracy"], label="Train Acc")
        ax[0].plot(history["epochs"], history["val_accuracy"], label="Val Acc")
        ax[0].set_title("Accuracy over Epochs")
        ax[0].legend()

        ax[1].plot(history["epochs"], history["loss"], label="Train Loss")
        ax[1].plot(history["epochs"], history["val_loss"], label="Val Loss")
        ax[1].set_title("Loss over Epochs")
        ax[1].legend()

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to load model metrics: {e}")

with tab3:
    st.subheader("API & Model Status")
    try:
        response = requests.get(f"{API_URL}/status")
        status = response.json()
        st.success(f"Uptime: {status['uptime']}")
        st.write(f"Model Status: **{status['model_status']}**")
        st.write(f"Face Cascade: **{status['face_cascade_status']}**")
    except Exception as e:
        st.error(f"Failed to load API status: {e}")
