import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

from pathlib import Path
import argparse

# --- Setup Parameters ---
# Define the path to your extracted dataset.
# The folder should contain subdirectories for each class.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / 'dataset_organized'
CLASS_INDEX_PATH = BASE_DIR / 'class_indices.json'

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT = 100


def build_model(num_classes):
    print("Building simple CNN model...")
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def fine_tune_model(model, num_unfrozen=50):
    print(f"Fine-tuning the top {num_unfrozen} layers of EfficientNetB0...")
    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:-num_unfrozen]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def save_class_indices(class_indices):
    reverse_mapping = {str(index): label for label, index in class_indices.items()}
    with open(CLASS_INDEX_PATH, 'w') as f:
        json.dump(reverse_mapping, f, indent=2)
    print(f"Saved class index mapping to '{CLASS_INDEX_PATH}'")


def main():
    parser = argparse.ArgumentParser(description='Train skin disease classifier')
    parser.add_argument(
        '--dataset',
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help='Path to the dataset root directory containing class subfolders.'
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        print("Please place your organized dataset in the project and pass --dataset <path> if needed.")
        return

    print("Setting up Data Generators with Augmentation...")
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    classes = list(train_generator.class_indices.keys())
    print(f"Found {num_classes} classes: {classes}")

    save_class_indices(train_generator.class_indices)

    labels = train_generator.classes
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights_arr)}
    print("Computed Class Weights to handle imbalance:")
    print(class_weight_dict)

    model = build_model(num_classes)
    model.summary()

    checkpoint = ModelCheckpoint("model_new.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stop]
    )

    print("\nTraining Complete! The best weights have been saved as 'model_new.h5'"
          )
    print("The class label mapping has been saved to 'class_indices.json'.")
    print("Update app.py or use the loaded class map so predictions match your dataset.")


if __name__ == "__main__":
    main()
