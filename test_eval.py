import numpy as np
from tensorflow.keras.models import model_from_json

print("Reading JSON...")
with open('model.json', 'r') as j_file:
    loaded_json_model = j_file.read()

print("JSON read. Loading model...")
try:
    model = model_from_json(loaded_json_model)
    print("Model architecture loaded. Loading weights...")
    model.load_weights('model.h5')
    print("Weights loaded successfully.")
    
    # test dummy input
    img = np.random.rand(1, 50, 50, 3)
    pred = model.predict(img)
    print("Prediction test:", pred)
except Exception as e:
    import traceback
    traceback.print_exc()
