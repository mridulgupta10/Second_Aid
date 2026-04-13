import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model_new.h5'

def test_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    print("Model summary:")
    model.summary()

    # Test with dummy input
    dummy_input = np.random.rand(1, 224, 224, 3)
    prediction = model.predict(dummy_input)
    print(f"Dummy prediction shape: {prediction.shape}")
    print(f"Predicted class: {np.argmax(prediction[0])}")
    print(f"Confidence: {np.max(prediction[0]):.4f}")

if __name__ == "__main__":
    test_model()