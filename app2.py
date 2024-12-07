from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, ImageOps
import pytesseract
import os
import cv2
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Preprocess image for OCR
def preprocess_image(image_path):
    # Read image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply binarization (thresholding)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Save preprocessed image for debugging
    debug_path = os.path.join(UPLOAD_FOLDER, "debug_processed_image.png")
    cv2.imwrite(debug_path, binary_image)
    return Image.fromarray(binary_image)

# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part in the request"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Preprocess image
            processed_image = preprocess_image(file_path)
            
            # Perform OCR
            extracted_text = pytesseract.image_to_string(processed_image, config="--psm 6")
            os.remove(file_path)  # Optional: Remove uploaded file
            
            # Handle no text detected
            if not extracted_text.strip():
                return "<h1>No text detected in the uploaded image. Try another image.</h1>"

            return f"<h1>Extracted Text:</h1><pre>{extracted_text}</pre>"
        except Exception as e:
            return f"<h1>Error processing the image: {e}</h1>"

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
