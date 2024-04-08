"""
Train and save CNN model

for both dumb and smart models

"""
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tflite_c_converter import convert_tflite_to_c

model_is_smart = False

def prepare_spectrogram_data():
    X, y = get_spectrogram_data()

    # do as little data prep as possible, tiny ml has to do this too

    # split data

    # maybe add validation data
    return X_train, X_test, y_train, y_test

def get_spectrogram_data():
    # get images from correct file paths

    # loop through all folders "spectrogram_y"
    # save label form folder name

    return X, y

def construct_smart_model():
    return model

def construct_dumb_model():
    return model

def train_model(model, X_train, y_train, validation_split):
    # do validation split here
    # train_test_split(split=validation_split)

    model.fit(X_train, y_train)

    return model

def save_model(model, model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        path = "models/cnn/smart"
    else:
        path = "models/cnn/dumb"
    model.export(path)

    # convert to tf lite model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    # convert to c-array
    array_path = "model.tflite"
    convert_tflite_to_c(array_path, "model1")

if __name__ == "__main__":
    # prepare data
    X_train, X_test, y_train, y_test = prepare_spectrogram_data()

    if model_is_smart:
        # make smart neural network
        model = construct_smart_model()
    else:
        # make dumb model
        model = construct_dumb_model()

    # train model, with validation split
    train_model(model, X_train, y_train, 0.1)

    # evaluate the model

    # save the model
    save_model(model)