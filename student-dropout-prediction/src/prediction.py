
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Assuming constants are defined in preprocessing or passed in if needed
# from . import preprocessing

def predict_single_image(model, img_path, target_size=(224, 224)):
    print(f"Making prediction for image: {img_path}")

    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}.")
        return None, None, None

    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        img_array /= 255.0 # Rescale the image

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Simulate class labels - this should ideally come from the training data generator's class_indices
        # For now, hardcode based on the notebook's usage
        class_labels = ['Engaged', 'Not Engaged']
        predicted_label = class_labels[predicted_class_index]
        confidence = np.max(predictions)

        print(f"Prediction result: Label={predicted_label}, Confidence={confidence:.4f}")

        return predicted_label, confidence, None # Return None for any additional info not needed

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None
