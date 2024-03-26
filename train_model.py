import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from helpers.tflite_c_converter import convert_tflite_to_c
from helpers.function_generator import get_function_samples

X, y = get_function_samples(samples=2000)
X = X.reshape(-1, 1)

# TODO: Split the data into training and testing sets below
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_test, X_vali, y_test, y_vali = train_test_split(X_test, y_test, test_size=0.15)

# Plotting training, testing, and validation data points in blue, yellow, and red respectively,
# with corresponding labels 'Train', 'Test', 'Valid'
# plt.plot(X_train, y_train, 'b.', label="Train")
# plt.plot(X_test, y_test, 'y.', label="Test")
# plt.legend()
# plt.show()

# TODO: Add the TensorFlow Keras Sequential model below
model = keras.Sequential(
    [
        keras.layers.Dense(1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ]
)
opt = 'RMSprop'
model.compile(
    optimizer=opt,
    loss='mean_squared_error',
    metrics=['MAE']
)
model.summary()

# TODO: Add the model training (fitting) below
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=80,
    validation_data=(X_vali, y_vali)
)

# TODO: Optional > Uncomment the following lines to plot the training and validation history
# Extracting training and validation loss from training history
# Plotting training loss in green and validation loss in blue over epochs
loss = history.history['loss']
val_loss = history.history['val_loss']
x_range = range(1, len(history.epoch) + 1)
plt.plot(x_range, loss, 'g.', label='Training loss')
plt.plot(x_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# TODO: Uncomment the following lines to plot the actual data versus the predicted function
# Make predictions on the test set
predictions = model.predict(X_test)

# Plot actual vs. predicted values
plt.clf()
plt.plot(X_test, y_test, 'b.', label='Actual')
plt.plot(X_test, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()

# TODO: Uncomment the following lines to save the TensorFlow model
# Define a path where models are saved
model_path = "models"

# Export the trained model
model.export(model_path)

# TODO: Convert the saved TensorFlow model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# TODO: Optional > Implement a TensorFlow Lite Interpreter and check if the accuracy of the model stays the same


# TODO: convert TensorFlow Lite model to a c-array for use on microcontrollers
array_path = "model.tflite"
convert_tflite_to_c(array_path, "model1")
