import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import io

app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")

# Function to predict the image
def img_pred(img_data):
    try:
        img = Image.open(io.BytesIO(img_data))
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(opencvImage, (150, 150))
        img_reshaped = img_resized.reshape(1, 150, 150, 3)
        p = model.predict(img_reshaped)
        p = np.argmax(p, axis=1)[0]

        if p == 0:
            result = 'Glioma Tumor'
        elif p == 1:
            result = 'No Tumor'
        elif p == 2:
            result = 'Meningioma Tumor'
        else:
            result = 'Pituitary Tumor'

        return result
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        img_data = file.read()
        prediction = img_pred(img_data)
        return jsonify({"result": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
