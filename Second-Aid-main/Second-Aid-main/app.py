from flask import render_template, jsonify, Flask, redirect, url_for, request, make_response
import os
import io
import numpy as np
from PIL import Image
import keras.utils as image
from keras.models import model_from_json

app = Flask(__name__)

SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'

}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signin')
def signin():
    return render_template('signin.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/price')
def price():
    return render_template('price.html')




@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

def findMedicine(pred):
    if pred == 0:
        return "Diclofenac"
    elif pred == 1:
        return "Erivedge (Vismodegib)"
    elif pred == 2:
        return "Prescription Hydrogen Peroxide"
    elif pred == 3:
        return "Mild Topical Steroid"
    elif pred == 4:
        return "Aldesleukin"
    elif pred == 5:
        return "Melanocytic nevi can be surgically removed"
    elif pred == 6:
        return "Laser Therapy"


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    json_response = {}
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'Invalid File!'
            }), 400)

        imagePil = Image.open(io.BytesIO(file.read()))
        # Convert the image to an array and normalize it
        img = np.array(imagePil.resize((50, 50))) / 255.0
        img = img.reshape((1, 50, 50, 3))

        # Load the model architecture from JSON
        with open('model.json', 'r') as j_file:
            loaded_json_model = j_file.read()
        model = model_from_json(loaded_json_model)

        # Load the weights
        model.load_weights('model.h5')

        # Perform prediction
        prediction = model.predict(img)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        accuracy = round(accuracy * 100, 2)
        medicine = findMedicine(pred)

        json_response = {
            "detected": False if pred == 2 else True,
            "disease": disease,
            "accuracy": accuracy,
            "medicine": medicine,
            "img_path": file.filename,
        }

        return make_response(jsonify(json_response), 200)

    else:
        return render_template('detect.html')


if __name__ == "__main__":
    app.run(debug=True, port=3000)
