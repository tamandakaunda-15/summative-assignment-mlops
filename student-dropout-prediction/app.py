# app.py

import os
import io
import time
import pandas as pd
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)

# --- Global Variables and Configuration ---
# Vercel deploys files from the root directory, so we'll adjust the paths.
# Ensure your 'models' directory is at the same level as app.py
MODEL_PATH = 'models/student_engagement_model.h5'
CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'

# --- Load Model and Face Cascade ---
model = None
face_cascade = None

try:
    if os.path.exists(MODEL_PATH):
        print("--- Attempting to load the TensorFlow model ---")
        model = load_model(MODEL_PATH)
        print("SUCCESS: Model loaded successfully from " + MODEL_PATH)
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")

try:
    if os.path.exists(CASCADE_PATH):
        print("--- Attempting to load the face cascade ---")
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            print(f"ERROR: Failed to load face cascade from {CASCADE_PATH}")
        else:
            print("SUCCESS: Face cascade loaded successfully.")
    else:
        print(f"ERROR: Face cascade file not found at {CASCADE_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load face cascade: {e}")


# --- API Endpoints ---

@app.route("/")
def home():
    """A simple health check endpoint."""
    return "Student Engagement API is running!"

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns mock model evaluation metrics for the engagement model."""
    # Since this model is more complex, we'll use mock data for the demo.
    mock_metrics = {
        "accuracy": 0.92,
        "precision": 0.88,
        "recall": 0.95,
        "f1_score": 0.91,
        "classes": ["Engaged", "Not Engaged"]
    }
    return jsonify(mock_metrics)

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts student engagement from an uploaded image."""
    if model is None or face_cascade is None:
        return jsonify({"error": "Model or face cascade not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image file and convert to a numpy array
        image_stream = io.BytesIO(file.read())
        pil_image = Image.open(image_stream).convert('L') # Convert to grayscale
        image_np = np.array(pil_image)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"prediction": "No face detected"}), 200

        # For this demo, we'll just process the first detected face
        (x, y, w, h) = faces[0]
        face_roi = image_np[y:y+h, x:x+w]
        
        # Preprocess the face image for the model
        resized_face = cv2.resize(face_roi, (48, 48))
        expanded_face = np.expand_dims(np.expand_dims(resized_face, -1), 0)
        
        # Make a prediction
        prediction = model.predict(expanded_face)
        predicted_class = np.argmax(prediction)

        # Map prediction to a human-readable label
        engagement_labels = {0: "Engaged", 1: "Not Engaged"}
        result = {
            "prediction": engagement_labels.get(predicted_class, "Unknown"),
            "probability": float(np.max(prediction))
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route("/retrain", methods=["POST"])
def retrain_model():
    """Placeholder for a retraining endpoint."""
    return jsonify({"message": "Retraining logic would be here."})
