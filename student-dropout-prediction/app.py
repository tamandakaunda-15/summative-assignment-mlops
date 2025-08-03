import os
import io
import cv2
import numpy as np
import base64
import time
import pandas as pd
from flask import Flask, request, jsonify, Response
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS
import zipfile

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
MODEL_PATH = 'models/student_engagement_model.h5'
# This list will store a history of all predictions for the dashboard
prediction_history = []
# Record the start time of the application for uptime calculation
START_TIME = time.time()

# Attempt to load the model
print("--- Attempting to load the model ---")
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("SUCCESS: Model loaded successfully from " + MODEL_PATH)
    else:
        model = None
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Prediction and retraining endpoints will not work without a model.")
except Exception as e:
    model = None
    print(f"ERROR: Failed to load model from {MODEL_PATH}: {e}")
    print("Prediction and retraining endpoints will not work without a model.")

# --- API Endpoints ---

@app.route('/')
def home():
    """Root endpoint to confirm the API is running."""
    return jsonify({
        'status': 'success',
        'message': 'Student Engagement API is running! Available endpoints: /predict, /metrics, /status, /model_metrics, /train, /retrain.'
    }), 200

@app.route('/status', methods=['GET'])
def get_status():
    """
    Returns the model's uptime and health status.
    This provides the data needed for a "Model up-time" display.
    """
    uptime = time.time() - START_TIME
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    uptime_string = f"{hours}h {minutes}m {seconds}s"
    
    status_info = {
        'uptime': uptime_string,
        'model_status': 'Loaded' if model else 'Not Loaded'
    }
    return jsonify(status_info), 200

@app.route('/model_metrics', methods=['GET'])
def get_model_metrics():
    """
    Returns mock data for model performance visualizations.
    This provides data for "graphs for the trained model."
    A real implementation would read this from a training history file.
    """
    model_metrics = {
        'accuracy': 0.925,
        'loss': 0.156,
        'training_history': {
            'epochs': list(range(1, 11)),
            'accuracy': [0.70, 0.75, 0.81, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92, 0.93],
            'val_accuracy': [0.65, 0.70, 0.78, 0.82, 0.86, 0.88, 0.89, 0.91, 0.92, 0.925],
            'loss': [0.50, 0.45, 0.38, 0.30, 0.25, 0.20, 0.18, 0.17, 0.16, 0.156],
            'val_loss': [0.60, 0.55, 0.45, 0.35, 0.28, 0.22, 0.19, 0.17, 0.16, 0.158]
        }
    }
    return jsonify(model_metrics), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Returns dynamic data for the dashboard based on the history of predictions.
    This endpoint is wrapped in try...except to prevent crashes.
    """
    try:
        engaged_count = sum(1 for p in prediction_history if p.get('status') == 'Engaged')
        not_engaged_count = len(prediction_history) - engaged_count
        
        recent_predictions = prediction_history[-10:]
        
        metrics_data = {
            'total_predictions': len(prediction_history),
            'engaged_count': engaged_count,
            'not_engaged_count': not_engaged_count,
            'recent_predictions': recent_predictions
        }
        
        print(f"Metrics endpoint sending data: {metrics_data}")
        return jsonify(metrics_data), 200
    except Exception as e:
        print(f"ERROR: An error occurred in the metrics endpoint: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image, preprocesses it entirely (no face detection),
    and returns a JSON response with prediction and base64 image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if model is None:
        return jsonify({'error': 'Server is not properly configured.'}), 500

    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if opencv_image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Preprocess entire image to grayscale 48x48 for model input
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (48, 48))
        normalized_image = resized_image / 255.0
        input_array = np.expand_dims(normalized_image, axis=(0, -1))  # shape (1,48,48,1)

        prediction_score = model.predict(input_array)[0][0]
        engagement_status = "Engaged" if prediction_score > 0.5 else "Not Engaged"

        # Log prediction
        prediction_history.append({
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'status': engagement_status
        })
        print(f"Logged prediction: {engagement_status}")

        # Encode original image (no bounding boxes)
        _, buffer = cv2.imencode('.jpg', opencv_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'message': 'Image analyzed.',
            'prediction_score': float(prediction_score),
            'engagement_status': engagement_status,
            'image_base64': image_base64
        }), 200

    except Exception as e:
        print(f"ERROR: An error occurred in the predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """
    Placeholder endpoint to handle model training.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded, cannot train.'}), 500
    
    # Placeholder training logic would go here
    return jsonify({
        'status': 'success',
        'message': 'Training initiated. This is a placeholder.'
    }), 200

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Placeholder endpoint to handle model retraining.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded, cannot retrain.'}), 500

    # Placeholder retraining logic would go here
    return jsonify({
        'status': 'success',
        'message': 'Retraining initiated. This is a placeholder.'
    }), 200

if __name__ == '__main__':
    if model is None:
        print("API will run, but with reduced functionality due to missing model.")
    app.run(host='0.0.0.0', port=5002, debug=True)

