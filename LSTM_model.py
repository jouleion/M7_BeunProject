"""
Train and save LSTM model

for both dumb and smart models


!! check input dimentions to model
!! handle audio array input properly
(output classes of the model stay the same)

!! input shape of the model for the fit function should be checked

"""
import math
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tflite_c_converter import convert_tflite_to_c


def prepare_spectrogram_data(data_split):
    # get, prepare and split the data
    X, y = get_audio_data("data")

    # Convert 2D grayscale images to 3D tensors with single channel
    # add a channel dimention to each image (128, 128) -> (128, 128, 1)
    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    print("X shape")
    print(X.shape)
    print("Y shape")
    print(y.shape)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_split)

    # maybe add validation data
    return X_train, X_test, y_train, y_test


def get_audio_data(data_path):
    # get all images from the different subfolders in the data folder
    # only those in the format "audio_x"

    # init x and y lists
    X = []
    y = []

    # Iterate through the subfolders
    for subfolder in os.listdir(data_path):
        subfolder_path = os.path.join(data_path, subfolder)

        # Check if the item is a directory
        if os.path.isdir(subfolder_path):
            # Get the label. this is the number at end of name "audio_1"
            split_name = subfolder.split("_")

            # determine label (only if folder format is correct)
            if len(split_name) > 1:
                if split_name[0] == "audio":
                    # store number in folder name as label
                    label = int(split_name[1])
                else:
                    # different folder name
                    continue
            else:
                # skip folder if in wrong format
                print("Skipped a folder, has wrong format")
                continue

            # Load all audio samples from the subfolder
            for filename in os.listdir(subfolder_path):
                # get all csv files
                if filename.endswith(".csv"):
                    print(filename)

                    audio_path = os.path.join(subfolder_path, filename)

                    # load csv as array and store in X
                    audio_sample = np.genfromtxt(audio_path, delimiter=',')
                    print(len(audio_sample))
                    X.append(audio_sample)
                    y.append(label)
    return X, y


def construct_smart_lstm_model(timesteps, features, n_of_classes):
    # construct lstm model here

    # timesteps are the number of datapoits in the audio_sample. there is only one feature, amplitude
    # timesteps = X.shape[1], features = X.shape[2]

    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(timesteps, features), stateful=True, return_sequences=False),  # I think these are supported by the arduino esp32 library
        keras.layers.LSTM(64, stateful=True, return_sequences=True),
        keras.layers.LSTM(64, stateful=True, return_sequences=True),
        keras.layers.LSTM(64, stateful=True, return_sequences=True),
        keras.layers.LSTM(64, stateful=True),
        keras.layers.Dense(40),
        keras.layers.Dense(n_of_classes),
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def construct_dumb_lstm_model(timesteps, features, n_of_classes):
    # construct lstm model here

    # timesteps are the number of datapoits in the audio_sample. there is only one feature, amplitude
    # timesteps = X.shape[1], features = X.shape[2]

    model = keras.Sequential([
        keras.layers.LSTM(32, batch_input_shape=(1, timesteps, features), stateful=True,
                          return_sequences=False),
        #keras.layers.LSTM(32, input_shape=(timesteps, features), stateful=True, return_sequences=False),  # I think these are supported by the arduino esp32 library
        keras.layers.LSTM(32, stateful=True, return_sequences=True),
        keras.layers.Dense(40),
        keras.layers.Dense(n_of_classes),
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    # do validation split here
    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=validation_split)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_vali, y_vali))

    return model, history


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x_range = range(1, len(history.epoch) + 1)
    plt.plot(x_range, loss, 'g', label='training loss')
    plt.plot(x_range, val_loss, 'b', label='Validation loss')
    plt.title('training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_predictions(X_test, actual_values, predictions):
    # Reshape X_test to 2D
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    plt.clf()
    plt.plot(X_test_2d[:, 0], actual_values, 'b.', label='Actual')
    plt.plot(X_test_2d[:, 0], predictions, 'r.', label='Predicted')
    plt.legend()
    plt.show()

    cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def load_model(model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        path = "models/lstm/smart/model.keras"
    else:
        path = "models/lstm/dumb/model.keras"

    # load the model
    model = keras.models.load_model(path)
    return model


def save_model(model, model_is_smart):
    # this is different from exporting, export is needed to convert to tflite
    # Define a path where models are saved
    if model_is_smart:
        path = "models/lstm/smart/model.keras"
    else:
        path = "models/lstm/dumb/model.keras"
    model.save(path)


def export_model(model, model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        path = "models/lstm/smart"
    else:
        path = "models/lstm/dumb"
    model.export(path)


def convert_to_tf_lite(model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        path = "models/lstm/smart"
    else:
        path = "models/lstm/dumb"

    # convert to tf lite model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()

    # prepare file name, in correct folder
    file_name = os.path.join(path + "_lite", "model.tflite")

    with open(file_name, 'wb') as f:
        f.write(tflite_model)
        print("saved tflite model as:")
        print(file_name)


def convert_to_c_array(model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        model_path = "models/lstm/smart_lite/model.tflite"
        file_name = "smart_LSTM"
    else:
        model_path = "models/lstm/dumb_lite/model.tflite"
        file_name = "dumb_LSTM"

    # convert to c-array
    convert_tflite_to_c(model_path, file_name, "models/c_arrays")


if __name__ == "__main__":
    # setup variables
    #     save_new_model will overwrite previous model
    #     model_is_smart is used to use a dumb or smart neural network. smart is used for the later stages
    #     n_of_keywords determines the number of output neurons, dependend on the nummber of classes used
    export_new_model = True
    model_is_smart = False
    n_of_keywords = 4

    # prepare data
    X_train, X_test, y_train, y_test = prepare_spectrogram_data(0.2)

    # print number of training samples
    print("number of training samples:")
    print(y_train.shape[0])

    if model_is_smart:
        # make smart neural network
        # 128x128
        model = construct_smart_lstm_model(X_train.shape[1], X_train.shape[2], n_of_keywords)     # !! check input dimentions !!
    else:
        # make dumb model
        # 16512 x 1
        model = construct_dumb_lstm_model(X_train.shape[1], X_train.shape[2], n_of_keywords)

    # train model
    max_batch_size = X_train.shape[0]
    epochs = 70

    # epochs, batch_size, validation split
    model, history = train_model(model, X_train, y_train, epochs, max_batch_size, 0.1)

    # plot accuray over time
    plot_history(history)

    if export_new_model:
        # export model
        export_model(model, model_is_smart)

        # convert exported model to tflite model. (model is loaded in function)
        convert_to_tf_lite(model_is_smart)

        # convert tflite model to c-array
        convert_to_c_array(model_is_smart)

    # evaluate new model
    predictions = model.predict(X_test)
    plot_predictions(X_test, y_test, predictions)
