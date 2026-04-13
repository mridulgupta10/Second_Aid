import numpy as np
import json
from pathlib import Path

try:
    import tensorflow as tf
    keras = tf.keras
    load_model = keras.models.load_model
    print('Using TensorFlow Keras backend.')
except Exception as tf_err:
    try:
        import keras
        load_model = keras.models.load_model
        print('Using standalone Keras backend.')
    except Exception as keras_err:
        print('Failed to import Keras:', tf_err, keras_err)
        exit(1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model_new.h5'
CLASS_INDEX_PATH = BASE_DIR / 'class_indices.json'
DATASET_DIR = BASE_DIR / 'dataset_organized'

def load_class_indices():
    with open(CLASS_INDEX_PATH, 'r') as f:
        reverse_mapping = json.load(f)
    class_indices = {label: int(index) for index, label in reverse_mapping.items()}
    return class_indices

def evaluate_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    class_indices = load_class_indices()
    class_names = list(class_indices.keys())
    print(f"Classes: {class_names}")

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    test_generator = test_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    print("Evaluating model on validation set...")
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Simple accuracy per class
    class_names = list(class_indices.keys())
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = (true_classes == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == i)
            print(f"{class_name}: {class_acc:.4f}")

if __name__ == "__main__":
    evaluate_model()