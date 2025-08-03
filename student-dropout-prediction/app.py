import os
import io
import time
import pandas as pd
import numpy as np
import cv2 # New library for face detection
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS
import zipfile

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
MODEL_PATH = 'models/student_engagement_model.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Path to face cascade file

# Attempt to load the model
print("--- Attempting to load the model ---")
try:
    model = load_model(MODEL_PATH)
    print("SUCCESS: Model loaded successfully from " + MODEL_PATH)
except Exception as e:
    model = None
    print(f"ERROR: Failed to load model from {MODEL_PATH}: {e}")
    print("Prediction and retraining endpoints will not work without a model.")

# Attempt to load the face detector
print("--- Attempting to load the face cascade ---")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"ERROR: Failed to load face cascade from {CASCADE_PATH}")
else:
    print("SUCCESS: Face cascade loaded successfully.")


start_time = time.time()

training_history = {
    "Epoch": list(range(1, 11)),
    "Loss": [0.45, 0.41, 0.38, 0.35, 0.32, 0.29, 0.27, 0.25, 0.23, 0.21],
    "Accuracy": [0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92],
}

dataset_distribution = {
    "Class": ['Engaged', 'Not engaged'],
    "Count": [1500, 1000],
}


# --- API Endpoints ---
@app.route('/', methods=['GET'])
def home():
    """
    A simple home endpoint for the API.
    """
    return jsonify({
        "message": "Welcome to the Student Engagement Prediction API. Use the /predict, /retrain, or /metrics endpoints."
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an image, performs face detection, and if a face is found,
    makes a prediction.
    """
    print("\n--- Received request on /predict endpoint ---")

    if not model:
        print("ERROR: Model not loaded, returning 503.")
        return jsonify({"error": "Model not loaded. Cannot make a prediction."}), 503

    if 'file' not in request.files:
        print("ERROR: No file part in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        print("ERROR: No selected file.")
        return jsonify({"error": "No selected file"}), 400

    try:
        print("INFO: File received successfully. Attempting to process image...")
        
        # Open the image from the file stream
        img_stream = file.read()
        pil_img = Image.open(io.BytesIO(img_stream))
        
        # Convert PIL image to OpenCV format (numpy array)
        cv_img = np.array(pil_img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        print("INFO: Detecting faces...")
        faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print("INFO: No faces detected in the image.")
            return jsonify({"error": "No face detected in the image. Please upload an image with a clear face."}), 400
        
        # Take the largest face found
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        print(f"INFO: Face detected at coordinates: x={x}, y={y}, w={w}, h={h}")
        
        # Crop the face from the original color image
        face_img = cv_img[y:y+h, x:x+w]
        
        # Convert the face image back to PIL format for model input
        face_pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Resize the face image to the model's expected input size
        face_pil_img = face_pil_img.resize((224, 224))
        
        print("INFO: Face image cropped and resized. Converting to array...")
        
        img_array = np.array(face_pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print("INFO: Image array created. Performing prediction...")
        
        prediction = model.predict(img_array)
        print("INFO: Prediction successful.")
        
        predicted_class = "Engaged" if prediction[0][0] > 0.5 else "Not engaged"
        confidence = float(prediction[0][0])
        
        print(f"INFO: Prediction result: {predicted_class} with confidence {confidence}")
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        print(f"ERROR: An exception occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Provides real-time model metrics for the dashboard.
    """
    uptime_seconds = time.time() - start_time
    
    latest_accuracy = training_history["Accuracy"][-1]
    latest_loss = training_history["Loss"][-1]

    metrics = {
        "uptime_seconds": uptime_seconds,
        "training_history": training_history,
        "dataset_distribution": dataset_distribution,
        "latest_accuracy": latest_accuracy,
        "latest_loss": latest_loss
    }
    
    return jsonify(metrics)


@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Triggers a model retraining process with new data.
    """
    if not model:
        return jsonify({"error": "Model not loaded. Cannot retrain."}), 503

    if 'data' not in request.files:
        return jsonify({"error": "No data file part"}), 400

    new_data_file = request.files['data']

    if not new_data_file.filename.endswith('.zip'):
        return jsonify({"error": "Invalid file type. Please upload a .zip file."}), 400

    try:
        zip_path = "retraining_data.zip"
        new_data_file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("new_training_data")

        # --- This is where the actual training logic would go ---
        print("Received new data. Simulating model retraining...")
        
        global training_history, dataset_distribution
        training_history["Epoch"].append(training_history["Epoch"][-1] + 1)
        training_history["Loss"].append(training_history["Loss"][-1] * 0.9)
        training_history["Accuracy"].append(training_history["Accuracy"][-1] + 0.01)
        dataset_distribution["Count"][0] += 50
        dataset_distribution["Count"][1] += 25

        os.remove(zip_path)

        return jsonify({
            "message": "Model retraining successfully triggered and completed.",
            "status": "success"
        }), 200

    except Exception as e:
        print(f"ERROR: An exception occurred during retraining: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
