
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.load(r"D:\Skin diease detection\preprocessed_128.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

print("✅ Loaded dataset:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(7, activation="softmax")  # 7 classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)


save_path = os.path.join(os.getcwd(), "skin_disease_model.h5")
model.save(save_path)
print("✅ Model saved at:", save_path)
