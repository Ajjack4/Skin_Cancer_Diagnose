from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load model once when app starts
model = load_model('skin_cancer_model.h5')
IMG_SIZE = 224

def preprocess_image(file):
    # Read image from the uploaded file
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = preprocess_image(file)
    pred = float(model.predict(img)[0][0])

    if pred >= 0.5:
        prediction = 'Benign'
        confidence = pred
    else:
        prediction = 'Malignant'
        confidence = 1 - pred  # confidence for Malignant

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
