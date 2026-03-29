import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

os.makedirs("model", exist_ok=True)

# Placeholder training data
# Replace with real extracted signal windows later
X = np.random.rand(200, 150, 1).astype(np.float32)
y = np.random.randint(60, 100, size=(200, 1)).astype(np.float32)

model = Sequential()
model.add(Conv1D(32, 3, activation="relu", input_shape=(150, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 3, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

model.save("model/bpm_model.h5")
print("Model saved as model/bpm_model.h5")
