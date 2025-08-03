import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
# Enable CORS for all domains to prevent cross-origin issues with the frontend
CORS(app)

# --- Global Variables ---
MODEL_PATH = 'models/student_engagement_model.h5'
CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'

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
    """
    Root endpoint to confirm the API is running.
    """
    return jsonify({
        'status': 'success',
        'message': 'Student Engagement API is running!'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image file via POST request, detects faces, and
    returns a prediction of engagement.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if face_cascade.empty() or model is None:
        return jsonify({'error': 'Server is not properly configured. Check logs for model/cascade errors.'}), 500

    try:
        # Read image bytes and convert to an OpenCV image format
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if opencv_image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return jsonify({'status': 'no_face_detected', 'message': 'No face found in the image.'}), 200

        # For this example, process the first detected face.
        (x, y, w, h) = faces[0]
        face_roi = gray_image[y:y+h, x:x+w]
        
        # Resize and preprocess the face for the model
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        face_roi_normalized = face_roi_resized / 255.0
        face_roi_expanded = np.expand_dims(face_roi_normalized, axis=(0, -1))

        prediction = model.predict(face_roi_expanded)
        # Placeholder logic for engagement status based on the prediction
        engagement_status = "Engaged" if prediction[0][0] > 0.5 else "Not Engaged"
        
        return jsonify({
            'status': 'success',
            'prediction': float(prediction[0][0]),
            'engagement_status': engagement_status
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream_video')
def stream_video():
    """
    Generates a live video feed from the webcam with face detection.
    This endpoint is not directly compatible with a simple Streamlit `st.image` call.
    It is provided for a standard web client.
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

# --- Main entry point to run the app ---
if __name__ == '__main__':
    # Ensure all necessary files are in place before starting the server
    if not os.path.exists(CASCADE_PATH):
        print(f"FATAL ERROR: The file {CASCADE_PATH} was not found. Please download it.")
    elif model is None:
        print("WARNING: Model failed to load. Only face detection endpoints will function.")
    else:
        # The host='0.0.0.0' allows the server to be accessed from other machines on the network
        # host='127.0.0.1' restricts it to the local machine. Use 0.0.0.0 for broader access.
        app.run(host='0.0.0.0', port=5002, debug=True)
