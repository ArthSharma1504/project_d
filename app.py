from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('model.h5')  # Replace with your .h5 file's name

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and text extraction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input

        # Use the model to predict
        prediction = model.predict(img_array)
        predicted_text = np.argmax(prediction, axis=1)[0]

        # Display the result
        return f"Extracted Text: {predicted_text}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
