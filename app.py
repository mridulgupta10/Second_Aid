from flask import render_template, jsonify, Flask, redirect, url_for, request, make_response
import json
import os
import io
import numpy as np
from pathlib import Path
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_NEW_PATH = BASE_DIR / 'model_new.h5'
MODEL_JSON_PATH = BASE_DIR / 'model.json'
MODEL_WEIGHTS_PATH = BASE_DIR / 'model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Disable caching for development
@app.after_request
def disable_cache(response):
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

CLASS_INDEX_PATH = BASE_DIR / 'class_indices.json'
MIN_PREDICTION_CONFIDENCE = 0.60
UNKNOWN_IMAGE_RESPONSE = {
    'success': True,
    'detected': False,
    'disease': 'Unable to identify a valid skin lesion',
    'accuracy': 0.0,
    'medicine': 'Please upload a clear skin image showing the affected area.',
    'img_path': None,
}

# Default fallback labels in case the saved class mapping is missing.
DEFAULT_SKIN_CLASSES = {
    0: 'Actinic Keratosis',
    1: 'Basal Cell Carcinoma',
    2: 'Melanocytic Nevi',
    3: 'Melanoma',
    4: 'Seborrheic Keratosis',
    5: 'Squamous Cell Carcinoma',
}

# Load training class labels if available.
CLASS_LABELS = DEFAULT_SKIN_CLASSES.copy()
if CLASS_INDEX_PATH.exists():
    try:
        with open(CLASS_INDEX_PATH, 'r') as f:
            loaded_labels = json.load(f)
        CLASS_LABELS = {int(k): v for k, v in loaded_labels.items()}
        print(f"Loaded class labels from '{CLASS_INDEX_PATH}'")
    except Exception as e:
        print(f"Unable to load class label mapping: {e}")
        CLASS_LABELS = DEFAULT_SKIN_CLASSES.copy()


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
    user_name = None
    if request.method == 'POST':
        user_name = request.form.get('name') or request.form.get('User') or request.form.get('email') or 'Guest'
    return render_template('dashboard.html', user_name=user_name)

def findMedicine(pred):
    medicines = {
        0: "Diclofenac / Imiquimod cream (consult dermatologist)",
        1: "Erivedge (Vismodegib) or surgical excision",
        2: "Monitoring recommended; surgical removal if changing",
        3: "Aldesleukin / Immunotherapy (urgent specialist referral)",
        4: "Prescription Hydrogen Peroxide or cryotherapy",
        5: "Surgical excision or radiation therapy",
    }
    return medicines.get(pred, "Consult a dermatologist")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_likely_skin_image(image, min_skin_ratio=0.20):
    try:
        small = image.resize((224, 224))
        hsv = np.array(small.convert('HSV'))
        ycbcr = np.array(small.convert('YCbCr'))
    except Exception:
        return False

    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    cb = ycbcr[..., 1].astype(np.int16)
    cr = ycbcr[..., 2].astype(np.int16)

    hsv_skin = (
        (h <= 40) &
        (s >= 35) &
        (s <= 220) &
        (v >= 50)
    )

    ycbcr_skin = (
        (cr >= 135) &
        (cr <= 180) &
        (cb >= 80) &
        (cb <= 135)
    )

    skin_mask = hsv_skin | ycbcr_skin
    skin_ratio = np.count_nonzero(skin_mask) / float(skin_mask.size)
    print(f"[DETECT] skin ratio: {skin_ratio:.3f}")
    return skin_ratio >= min_skin_ratio

try:
    import tensorflow as tf  # type: ignore[import]
    keras = tf.keras
    preprocess_input = keras.applications.efficientnet.preprocess_input
    load_model = keras.models.load_model
    model_from_json = keras.models.model_from_json
    print('Using TensorFlow Keras backend for model loading.')
except Exception as tf_err:
    try:
        import keras  # type: ignore[import]
        preprocess_input = keras.applications.efficientnet.preprocess_input
        load_model = keras.models.load_model
        model_from_json = keras.models.model_from_json
        print('TensorFlow import failed, using standalone Keras backend for model loading.')
    except Exception as keras_err:
        print('Failed to import TensorFlow Keras and standalone Keras:')
        print('TensorFlow error:', tf_err)
        print('Keras error:', keras_err)
        preprocess_input = None
        load_model = None
        model_from_json = None

# Load model here outside the request to avoid loading it on every hit.
# It will load once on server startup.
try:
    if load_model is None:
        raise ImportError('No Keras backend available to load the model.')

    if MODEL_NEW_PATH.exists():
        model = load_model(MODEL_NEW_PATH)
        target_size = (224, 224)
        print("Loaded customized model_new.h5!")
    elif MODEL_JSON_PATH.exists() and MODEL_WEIGHTS_PATH.exists():
        with open(MODEL_JSON_PATH, 'r') as j_file:
            model = model_from_json(j_file.read())
        model.load_weights(MODEL_WEIGHTS_PATH)
        target_size = (50, 50)
        print("Loaded legacy model.h5")
    else:
        raise FileNotFoundError('No valid model file found.')
except Exception as e:
    print("Error loading model:", e)
    model = None
    target_size = (224, 224)

# Warmup: run one dummy prediction so TF traces the graph now, not on first request
if model is not None:
    try:
        _dummy = np.zeros((1,) + target_size + (3,), dtype=np.float32)
        model.predict(_dummy, verbose=0)
        print("Model warmup complete.")
    except Exception as e:
        print("Warmup failed (non-critical):", e)


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        print(f"[DETECT] POST request received. Content-Type: {request.content_type}")
        if 'file' not in request.files:
            print("[DETECT] No file in request.files")
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'Please upload an image file.'
            }), 400)

        file = request.files['file']

        if file.filename == '':
            print("[DETECT] Empty filename")
            return make_response(jsonify({
                'error': 'No file selected',
                'code': 'NO_FILE',
                'message': 'Please choose a file to upload.'
            }), 400)

        if not allowed_file(file.filename):
            return make_response(jsonify({
                'error': 'Unsupported file type',
                'code': 'TYPE',
                'message': 'Allowed image types are: png, jpg, jpeg, bmp, gif.'
            }), 400)

        try:
            imagePil = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            print(f"[DETECT] Image open error: {e}")
            return make_response(jsonify({
                'error': 'Invalid image file',
                'code': 'IMAGE',
                'message': 'Unable to read the uploaded image. Please upload a valid image file.'
            }), 400)

        if not is_likely_skin_image(imagePil):
            print('[DETECT] Rejected image because it does not appear to contain skin-like regions')
            return make_response(jsonify({
                'success': True,
                'detected': False,
                'disease': 'Unable to identify a valid skin lesion',
                'accuracy': 0.0,
                'medicine': 'Please upload a clear photo of the affected skin area.',
                'img_path': file.filename,
            }), 200)

        img = np.array(imagePil.resize(target_size, Image.BILINEAR), dtype=np.float32)

        if MODEL_NEW_PATH.exists():
            img = preprocess_input(img)
        else:
            img = img / 255.0

        img = img.reshape((1,) + target_size + (3,))

        if model is None:
            print("[DETECT] Model is None")
            return make_response(jsonify({
                'error': 'Model could not be loaded.',
                'code': 'MODEL',
                'message': 'Model is unavailable on the server.'
            }), 500)

        print(f"[DETECT] Starting prediction for {file.filename}")
        prediction = model.predict(img, verbose=0)
        print(f"[DETECT] Prediction complete. Result: {prediction}")
        pred = int(np.argmax(prediction))
        accuracy = float(prediction[0][pred])
        accuracy = round(accuracy * 100, 2)

        if accuracy < MIN_PREDICTION_CONFIDENCE * 100:
            print(f"[DETECT] Low confidence prediction: {accuracy}%")
            unknown_response = UNKNOWN_IMAGE_RESPONSE.copy()
            unknown_response['img_path'] = file.filename
            return make_response(jsonify(unknown_response), 200)

        disease = CLASS_LABELS.get(pred, DEFAULT_SKIN_CLASSES.get(pred, 'Unknown'))
        medicine = findMedicine(pred)

        json_response = {
            'success': True,
            'detected': False if pred == 2 else True,
            'disease': disease,
            'accuracy': accuracy,
            'medicine': medicine,
            'img_path': file.filename,
        }

        return make_response(jsonify(json_response), 200)

    return render_template('detect.html')


@app.errorhandler(413)
def request_entity_too_large(error):
    return make_response(jsonify({
        'error': 'File too large',
        'code': 'LIMIT',
        'message': 'Image must be smaller than 16MB.'
    }), 413)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=3000)
