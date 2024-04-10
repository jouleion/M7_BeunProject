"""
Train and save CNN model
convert to tflite model and c-array

for both dumb and smart models

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
    X, y = get_spectrogram_data("data")

    # Convert 2D grayscale images to 3D tensors with single channel
    # add a channel dimention to each image (128, 128) -> (128, 128, 1)
    X = [np.expand_dims(img, axis=-1) for img in X]
    X = np.array(X)

    # Normalize pixel values to [0, 1] range
    X = X.astype('float32') / 255.0

    y = np.array(y)
    y = np.expand_dims(y, axis=-1)

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
                    label = int(split_name[1])
                else:
                    continue
            else:
                # skip folder if in wrong format
                print("Skipped a folder, has wrong format")
                continue

            # Load the images from the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".png"):
                    print(filename)
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path)
                    # Convert the image to grayscale
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    X.append(gray_image)
                    y.append(label)
    return X, y


def construct_smart_cnn_model(width, heigth, n_of_classes):
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


def construct_dumb_cnn_model(width, heigth, n_of_classes):
    # construct cnn model here
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(width, heigth, 1)),
            keras.layers.MaxPooling2D((4, 4)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

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


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x_range = range(1, len(history.epoch) + 1)
    plt.plot(x_range, loss, 'g.', label='training loss')
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
        path = "models/cnn/smart/model.keras"
    else:
        path = "models/cnn/dumb/model.keras"

    # load the model
    model = keras.models.load_model(path)
    return model


def save_model(model, model_is_smart):
    # this is different from exporting, export is needed to convert to tflite
    # Define a path where models are saved
    if model_is_smart:
        path = "models/cnn/smart/model.keras"
    else:
        path = "models/cnn/dumb/model.keras"
    model.save(path)

def export_model(model, model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        path = "models/cnn/smart"
    else:
        path = "models/cnn/dumb"
    model.export(path)


def convert_to_tf_lite(model_is_smart):
    # Define a path where models are saved
    if model_is_smart:
        path = "models/cnn/smart"
    else:
        path = "models/cnn/dumb"

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
        model_path = "models/cnn/smart_lite/model.tflite"
        file_name = "smart_CNN"
    else:
        model_path = "models/cnn/dumb_lite/model.tflite"
        file_name = "dumb_CNN"


    # convert to c-array
    convert_tflite_to_c(model_path, file_name, "models/c_arrays")


if __name__ == "__main__":
    # setup variables
    #     save_new_model will overwrite previous model
    #     model_is_smart is used to use a dumb or smart neural network. smart is used for the later stages
    #     n_of_keywords determines the number of output neurons, dependend on the nummber of classes used
    export_new_model = True
    model_is_smart = True
    n_of_keywords = 4

    # prepare data
    X_train, X_test, y_train, y_test = prepare_spectrogram_data(0.2)

    # print number of training samples
    print("number of training samples:")
    print(y_train.shape[0])

    if model_is_smart:
        # make smart neural network
        # 128x128
        model = construct_smart_cnn_model(128, 128, n_of_keywords)
    else:
        # make dumb model
        # 128x128
        model = construct_dumb_cnn_model(128, 128, n_of_keywords)

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
