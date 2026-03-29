import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/bpm_model.h5"

model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

def predict_bpm_from_signal(signal_window):
    global model

    if model is None:
        return 0

    signal_window = np.array(signal_window, dtype=np.float32)

    if len(signal_window) < 150:
        return 0

    signal_window = signal_window[-150:]
    signal_window = signal_window.reshape(1, 150, 1)

    prediction = model.predict(signal_window, verbose=0)
    return float(prediction[0][0])
