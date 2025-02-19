import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model (you can use a model trained for skin disease classification)
model = tf.keras.models.load_model('skin_disease_model.h5')  # Assuming you have a model saved

# Map of disease names to image paths
disease_images = {
    "Actinic keratosis": "C:\Users\hp\Downloads\Actinic Keratosis.jpeg",
    "Atopic Dermatitis": "C:\Users\hp\Downloads\Atopic Dermatitis.jpeg",
    "Benign keratosis": "C:\Users\hp\Downloads\Benign Keratosis.jpeg",
    "Dermatofibroma": "C:\Users\hp\Downloads\Dermatofibroma.jpeg",
    "Melanocytic nevus": "C:\Users\hp\Downloads\Melanocytic Nevus.jpeg",
    "Melanoma": "C:\Users\hp\Downloads\Melanoma.jpeg",
    "Squamous cell carcinoma": "C:\Users\hp\Downloads\Squamous Cell Carcinoma.jpeg",
    "Tinea Ringworm Candidiasis": "Tinea Ringworm Candidiasis",
    "Vascular lesion":"C:\Users\hp\Desktop\Vascular lesion.png",
    
}

def prepare_image(image_path):
    """Preprocess image for model prediction."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route('/')
def home():
    return render_template('imageupload.html')  # Assuming you have an HTML template for frontend

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save the file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess the uploaded image and make prediction
    img_array = prepare_image(file_path)
    prediction = model.predict(img_array)
    
    # Map the prediction to a disease name (assuming model gives a class index)
    disease_name = list(disease_images.keys())[np.argmax(prediction)]
    
    # Return the result
    return jsonify({"disease": disease_name, "image_path": disease_images[disease_name]})

if __name__ == '__main__':
    app.run(debug=True)
