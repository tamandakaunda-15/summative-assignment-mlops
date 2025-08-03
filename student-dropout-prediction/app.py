import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Flask Application Setup ---
# Initialize the Flask app. This will be the main entry point for the API.
app = Flask(__name__)

# --- Model Configuration and Loading ---
# Define the path to your trained model file.
MODEL_PATH = 'models/student_engagement_model.h5'
# Define the required image dimensions for the model.
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Define the class labels, ensuring the order matches the model's output.
class_labels = {0: 'Engaged', 1: 'Not engaged'}

# Load the trained Keras model into memory once when the application starts.
# This is a crucial optimization to avoid reloading the model for every request.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Prediction Endpoint ---
# Define a route for the prediction endpoint. This will handle POST requests.
# The endpoint expects an image file to be uploaded.
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads and returns a prediction from the model.
    """
    # Check if a file was uploaded in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the model was loaded successfully.
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Read the image file from the request.
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess the image to fit the model's input requirements.
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        
        # Scale the pixel values to the range [0, 1]. This is critical as the
        # model was trained on scaled data.
        img_array = img_array / 255.0
        
        # Expand the dimensions to create a batch of a single image.
        # The model expects a shape of (batch_size, height, width, channels).
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction using the loaded model.
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_label = class_labels[predicted_class_index]

        # Return the prediction result as a JSON response.
        response = {
            'filename': file.filename,
            'prediction': predicted_label,
            'confidence': confidence
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Main Entry Point ---
# Run the application. Use `0.0.0.0` to make it accessible externally.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
