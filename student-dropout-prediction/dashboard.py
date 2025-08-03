import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import requests
import io
import json
from PIL import Image
from datetime import timedelta

# --- API Configuration ---
# Corrected: Using the deployed API endpoint.
# The previous `http://localhost:5001` only works if the API is running locally.
API_ENDPOINT = "https://summative-assignment-mlops.onrender.com"

# --- Functions to interact with the API ---
@st.cache_data(ttl=5) # Cache the metrics to avoid spamming the API
def get_metrics_from_api():
    """
    Fetches real-time metrics (uptime, training history, dataset distribution)
    from the Flask API.
    """
    try:
        response = requests.get(f"{API_ENDPOINT}/metrics")
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        # Handle connection errors or JSON decoding errors
        st.error(f"Failed to get metrics from API. Please ensure the deployed API is running: {e}")
        return None

def make_prediction_request(image_data):
    """
    Sends an image to the API's prediction endpoint and returns the result.
    """
    try:
        files = {'file': ('image.jpg', image_data.getvalue(), 'image/jpeg')}
        response = requests.post(f"{API_ENDPOINT}/predict", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get prediction from API: {e}")
        return None

def trigger_training_request(training_data):
    """
    Sends new data (a zip file) to the /retrain endpoint to trigger
    the model retraining process on the API.
    """
    try:
        files = {'data': ('training_data.zip', training_data.getvalue(), 'application/zip')}
        response = requests.post(f"{API_ENDPOINT}/retrain", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to start training: {e}")
        return None

# --- Main App Layout ---
st.set_page_config(
    page_title="Student Engagement Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Engagement Model Dashboard")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ("Dashboard Overview", "Live Prediction", "Model Retraining")
)

if page == "Dashboard Overview":
    st.header("Dashboard Overview")
    
    # Fetch all metrics once and use the data throughout this section
    metrics_data = get_metrics_from_api()
    if not metrics_data:
        st.warning("Could not fetch metrics from the API. Please ensure the backend is running and reachable.")
        st.stop()

    # Display Model Up-time
    st.subheader("Model Up-time")
    uptime_seconds = metrics_data.get("uptime_seconds", 0)
    uptime_delta = timedelta(seconds=int(uptime_seconds))
    st.metric(label="API Uptime", value=str(uptime_delta))

    # Visualization section for model performance
    st.subheader("Model Training History")
    training_history = metrics_data.get("training_history", {})
    training_df = pd.DataFrame(training_history)
    
    col1, col2 = st.columns(2)
    with col1:
        if not training_df.empty:
            fig_loss = px.line(
                training_df, x="Epoch", y="Loss", title="Training Loss over Epochs"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("No training history data available.")
    with col2:
        if not training_df.empty:
            fig_accuracy = px.line(
                training_df, x="Epoch", y="Accuracy", title="Training Accuracy over Epochs"
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
        else:
            st.info("No training history data available.")

    # Visualization for Dataset Distribution
    st.subheader("Dataset Class Distribution")
    class_distribution = metrics_data.get("dataset_distribution", {})
    distribution_df = pd.DataFrame(list(class_distribution.items()), columns=['Class', 'Count'])
    if not distribution_df.empty:
        fig_dist = px.bar(
            distribution_df,
            x="Class",
            y="Count",
            title="Distribution of Classes in Dataset",
            color="Class"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No dataset distribution data available.")

    # Interpretation of features (as requested in the original prompt)
    st.subheader("Interpretation of Features")
    st.markdown(
        """
        -   **Uptime:** The uptime metric tells us how long the API has been running continuously. A stable, long uptime indicates a reliable and healthy deployment environment.
        -   **Training Loss & Accuracy:** These plots tell a story of the model's learning process. As the number of epochs increases, a healthy model's loss should decrease, and its accuracy should increase, showing that it's learning to make better predictions over time.
        -   **Dataset Distribution:** This bar chart shows the balance of our training data. If one class (e.g., 'Not engaged') has significantly fewer examples, it might tell us that the model could be biased and may perform poorly on that class.
        """
    )


elif page == "Live Prediction":
    st.header("Live Prediction")
    st.markdown("Upload an image to get a prediction from the current model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("Predicting...")
            image_bytes = io.BytesIO(uploaded_file.getvalue())

            # This now calls the real API endpoint
            prediction_response = make_prediction_request(image_bytes)

            if prediction_response:
                prediction = prediction_response.get("prediction")
                confidence = prediction_response.get("confidence")
                st.success(f"Prediction: **{prediction}** with a confidence of **{confidence:.2f}**")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.info("Please upload an image file to get a prediction.")

elif page == "Model Retraining":
    st.header("Model Retraining")
    st.markdown(
        """
        Use this section to upload a new dataset (in a `.zip` file) and trigger model retraining.
        """
    )
    
    with st.expander("Instructions for Retraining"):
        st.write("1. Prepare your new training data in a folder structure and compress it into a `.zip` file.")
        st.write("2. Upload the `.zip` file using the button below.")
        st.write("3. Click 'Start Retraining' to trigger the process on the API.")

    uploaded_data = st.file_uploader(
        "Upload a new dataset (ZIP file)",
        type=["zip"]
    )

    if st.button("Start Retraining", type="primary", use_container_width=True):
        if uploaded_data is not None:
            with st.spinner('Sending data and waiting for retraining to complete...'):
                retrain_response = trigger_training_request(uploaded_data)
                if retrain_response and retrain_response.get("status") == "success":
                    st.success(f"Retraining successful! Message: {retrain_response.get('message')}")
                    st.balloons()
                else:
                    st.error(f"Retraining failed. Message: {retrain_response.get('error', 'Unknown error')}")
        else:
            st.warning("Please upload a `.zip` dataset before starting the retraining process.")

