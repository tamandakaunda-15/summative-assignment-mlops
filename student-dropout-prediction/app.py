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

# --- Global Variables ---
MODEL_PATH = 'models/student_engagement_model.h5'
# The correct path from your previous code
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Attempt to load the model
print("--- Attempting to load the TensorFlow model ---")
try:
    model = load_model(MODEL_PATH)
    print("SUCCESS: Model loaded successfully from " + MODEL_PATH)
except Exception as e:
    model = None
    print(f"ERROR: Failed to load model from {MODEL_PATH}: {e}")
    print("Prediction and retraining endpoints will not work without a model.")

# Attempt to load the face detector
print("--- Attempting to load the face cascade ---")
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Cascade file is empty or corrupted: {CASCADE_PATH}")
    print("SUCCESS: Face cascade loaded successfully.")
except Exception as e:
    face_cascade = None
    print(f"ERROR: Failed to load face cascade from {CASCADE_PATH}: {e}")


# --- API Endpoints ---
@app.route("/", methods=["GET"])
def hello_world():
    """A simple route to confirm the server is running."""
    return jsonify({"message": "API is running. Use /predict for predictions."})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image uploads, detects faces, and returns face coordinates.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        image_stream = io.BytesIO(file.read())
        try:
            image = Image.open(image_stream)
            img_array = np.array(image.convert('L'))  # Convert to grayscale
        except Exception as e:
            return jsonify({"error": f"Could not process image file: {e}"}), 400

        if face_cascade is None:
            return jsonify({"error": "Face detection model not loaded"}), 500

        faces = face_cascade.detectMultiScale(
            img_array,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            faces_list = [face.tolist() for face in faces]
            return jsonify({"faces": faces_list, "face_count": len(faces)})
        else:
            return jsonify({"message": "No faces detected."})

    return jsonify({"error": "Something went wrong"}), 500


# --- Run the Flask app ---
if __name__ == '__main__':
    # Starts the web server on localhost port 5001, as you specified.
    app.run(debug=True, port=5001)
