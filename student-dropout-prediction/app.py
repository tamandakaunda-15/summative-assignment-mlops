# flask_api.py
import os
import io
import cv2
import numpy as np
import base64
import time
import pandas as pd # Re-added for potential future use, though not strictly needed here
from flask import Flask, request, jsonify, Response
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS
import zipfile # Re-added for potential future use

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
MODEL_PATH = 'models/student_engagement_model.h5'
CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'
# This list will store a history of all predictions for the dashboard
prediction_history = [] 

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

# Attempt to load the face detector
print("--- Attempting to load the face cascade ---")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"ERROR: Failed to load face cascade from {CASCADE_PATH}")
else:
    print("SUCCESS: Face cascade loaded successfully.")

# --- API Endpoints ---
@app.route('/')
def home():
    """Root endpoint to confirm the API is running."""
    return jsonify({
        'status': 'success',
        'message': 'Student Engagement API is running! The /predict and /metrics endpoints are ready.'
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Returns dynamic data for the dashboard based on the history of predictions.
    """
    engaged_count = sum(1 for p in prediction_history if p.get('status') == 'Engaged')
    not_engaged_count = len(prediction_history) - engaged_count
    
    # Get the last 10 predictions for the recent activity feed
    recent_predictions = prediction_history[-10:]
    
    metrics_data = {
        'total_predictions': len(prediction_history),
        'engaged_count': engaged_count,
        'not_engaged_count': not_engaged_count,
        'recent_predictions': recent_predictions
    }
    return jsonify(metrics_data), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image, detects faces, and returns a JSON response
    that includes a base64-encoded image with bounding boxes
    and prediction details. It also logs the prediction to history.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if face_cascade.empty() or model is None:
        return jsonify({'error': 'Server is not properly configured.'}), 500

    try:
        # Read image bytes and convert to an OpenCV image format
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if opencv_image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        annotated_image = opencv_image.copy()
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        predictions = []
        if len(faces) == 0:
            status = "no_face_detected"
            message = 'No face found in the image.'
            # Log the no-face event to history for the dashboard
            prediction_history.append({
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'status': 'No Face'
            })
        else:
            status = "success"
            message = 'Face(s) detected and analyzed.'
            
            for (x, y, w, h) in faces:
                face_roi = gray_image[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi, (48, 48))
                face_roi_normalized = face_roi_resized / 255.0
                face_roi_expanded = np.expand_dims(face_roi_normalized, axis=(0, -1))

                prediction_score = model.predict(face_roi_expanded)[0][0]
                engagement_status = "Engaged" if prediction_score > 0.5 else "Not Engaged"

                color = (0, 255, 0) if engagement_status == "Engaged" else (0, 0, 255)
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    annotated_image,
                    f"{engagement_status}: {prediction_score:.2f}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                
                predictions.append({
                    'bounding_box': [int(x), int(y), int(w), int(h)],
                    'prediction_score': float(prediction_score),
                    'engagement_status': engagement_status
                })

                # Log the prediction to the global history
                prediction_history.append({
                    'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    'status': engagement_status
                })


        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': status,
            'message': message,
            'predictions': predictions,
            'image_base64': image_base64
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream_video')
def stream_video():
    """
    Generates a live video feed from the webcam with face detection.
    """
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Placeholder endpoint to retrain the model.
    """
    return jsonify({
        'status': 'success',
        'message': 'Retraining endpoint is a placeholder. Logic would be implemented here.'
    }), 200

if __name__ == '__main__':
    if not os.path.exists(CASCADE_PATH) or model is None:
        print("API will run, but with reduced functionality due to missing files.")
    app.run(host='0.0.0.0', port=5002, debug=True)
