"""
Train and save LSTM model

for both dumb and smart models




!!!!!!!!!! THIS HAS TO BE CHANGED TO LSTM model

"""
import math
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tflite_c_converter import convert_tflite_to_c

model_is_smart = False
n_of_keywords = 2

def prepare_spectrogram_data(data_split):
    # get, prepare and split the data
    X, y = get_spectrogram_data("data")

    # do as little data prep as possible, tiny ml has to do this too

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_split)

    # maybe add validation data
    return X_train, X_test, y_train, y_test

def get_spectrogram_data(data_path):
    # get all images from the different subfolders in the data folder

    X = []
    y = []

    # Iterate through the subfolders
    for subfolder in os.listdir(data_path):
        subfolder_path = os.path.join(data_path, subfolder)

        # Check if the item is a directory
        if os.path.isdir(subfolder_path):
            # Get the label. this is the number at end of name "spectrogram_1"
            split_name = subfolder.split("_")

            if len(split_name) > 1:
                if split_name[0] == "spectrogram":
                    # store number after _ as the label
                    label = split_name[1]
                else:
                    continue
            else:
                # skip folder if does not exist
                print("Skipped a folder, has no index number")
                continue

            # Load the images from the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".png"):
                    print(filename)
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path)
                    X.append(image)
                    y.append(label)
    return X, y

def construct_smart_model(width, heigth, n_of_classes):
    # construct cnn model here
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, heigth, 1)),
            keras.layers.MaxPooling2D((3, 3)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            # flatten the output and add the fully connected layers
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),

            # set number of outputs equal to the number of classes
            keras.layers.Dense(n_of_classes)
        ]
    )
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['MAE']
    )
    model.summary()
    return model

def construct_dumb_model(width, heigth, n_of_classes):
    # construct cnn model here
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, heigth, 1)),
            keras.layers.MaxPooling2D((3, 3)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            # flatten the output and add the fully connected layers
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),

            # set number of outputs equal to the number of classes
            keras.layers.Dense(n_of_classes)
        ]
    )
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['MAE']
    )
    model.summary()
    return model

def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    # do validation split here
    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=validation_split)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_vali, y_vali))

    return model, history

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
    X_train, X_test, y_train, y_test = prepare_spectrogram_data(0.2)

    print(y_train)

    if model_is_smart:
        # make smart neural network
        # 128x128
        model = construct_smart_model(128, 128, n_of_keywords)
    else:
        # make dumb model
        # 128x128
        model = construct_dumb_model(128, 128, n_of_keywords)

    # train model
    # epochs, batch_size, validation split
    model, history = train_model(model, X_train, y_train, 50, 80, 0.5)

    # evaluate the model

    # predictions
    predictions = model.predict(X_test)

    # plot some results?


    # save the model
    save_model(model, model_is_smart)