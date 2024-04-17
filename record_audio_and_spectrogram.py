"""
This is used to record the microphone data from Wilson
either as a spectrogram or as an audio array

INSTRUCTIONS:
    Audio and spectrogram read:
    - upload AudioAndSpectrogram.ino to the tinyml

    Recording data
    - Set the keyword id
    - Press reset on the tiny ml board
    - Directly after press run on this python file
    - when the red led is fully lit the tiny ml board is recording, there is a short flash just before it starts.
    - the tiny ml send the audio and the spectrogram data, this is then saved
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
import os
import csv

# set true when tiny ml is conected
tiny_ml_connected = True

def read_serial_data(buffer_size, number_of_samples):
    # Initialize variables to track the state of the data
    audio_data = []

    spectrogram_data = np.zeros((number_of_samples, buffer_size))
    row_index = 0
    col_index = 0

    is_audio = False
    is_spectrogram = False
    audio_received = False
    spectrogram_received = False

    # Get the start time
    start_time = time.time()

    while True:
        # Read and decode the incoming data
        data = tiny_ml_serial.readline().decode().strip()

        if data:
            # Check if the data indicates the start of audio or spectrogram
            if data == "a":
                is_audio = True
                is_spectrogram = False
                print("Start of audio data")
            elif data == "s":
                is_audio = False
                is_spectrogram = True
                print("Start of spectrogram data")
            elif data == "e":
                # Check if the data indicates the end of audio or spectrogram
                if is_audio:
                    print("End of audio data")
                    is_audio = False
                    audio_received = True
                elif is_spectrogram:
                    print("End of spectrogram data")
                    is_spectrogram = False
                    spectrogram_received = True

                # Check if both audio and spectrogram have been received
                if audio_received and spectrogram_received:
                    print(f"Spectrogram data shape: {spectrogram_data.shape}")
                    return np.array(audio_data), spectrogram_data
            else:
                # Process the data based on the current state
                if is_audio:
                    try:
                        # Add the audio value to the audio_data list
                        audio_data.append(float(data))
                    except ValueError:
                        audio_data.append(0)
                        # report error
                        print(f"Error with audio input: {data}")
                elif is_spectrogram:
                    # Split the spectrogram data and store it in the spectrogram_data array
                    if "," in data:
                        if row_index < number_of_samples - 1:
                            try:
                                row, value = data.split(",")
                                row = int(re.sub("[^0-9]", "", row))
                                value = int(re.sub("[^0-9]", "", value))
                                spectrogram_data[row_index][col_index] = value

                                # increment col index
                                col_index += 1
                                if col_index >= buffer_size:
                                    col_index = 0
                                    row_index = row
                            except ValueError:
                                # value stays 0
                                # report error
                                print(f"Error with spectrogram input: {data}")

        # Check if the loop has been running for too long without receiving both audio and spectrogram
        if time.time() - start_time > 10:
            print("Error: Could not receive both audio and spectrogram data.")
            return None, None

def save_spectrogram(spectrogram, data_path, keyword_id):
    # save spectrogram in correct folder depending on the keyword id thats being used

    # put in right subfolder
    subfolder_path = os.path.join(data_path, str(keyword_id))

    # make subfolder if it did not exsist yet
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Find the highest existing index and generate a new file name
    existing_files = os.listdir(subfolder_path)
    max_index = 0
    for file in existing_files:
        if file.startswith("spectrogram_") and file.endswith(".png"):
            try:
                index = int(file.split("_")[1].split(".")[0])
                max_index = max(max_index, index)
            except ValueError:
                pass

    new_index = max_index + 1
    # make new name with 3 leading zeros
    new_filename = f"spectrogram_{new_index:03d}.png"
    new_file_path = os.path.join(subfolder_path, new_filename)

    # Save the spectrogram to the new file
    cv2.imwrite(new_file_path, spectrogram)
    print(f"Spectrogram saved as {new_filename}")

def save_audio(audio_array, data_path, keyword_id):
    # save audio sample in correct folder depending on the keyword id thats being used

    # put in right subfolder
    subfolder_path = os.path.join(data_path, str(keyword_id))

    # make subfolder if it did not exsist yet
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Find the highest existing index and generate a new file name
    existing_files = os.listdir(subfolder_path)
    max_index = 0
    for file in existing_files:
        if file.startswith("audio_") and file.endswith(".csv"):
            try:
                index = int(file.split("_")[1].split(".")[0])
                max_index = max(max_index, index)
            except ValueError:
                pass

    new_index = max_index + 1
    # make new name with 3 leading zeros
    new_filename = f"audio_{new_index:03d}.csv"
    new_file_path = os.path.join(subfolder_path, new_filename)

    # Save the audio to the new file
    with open(new_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(audio_array)
    print(f"Audio saved as {new_filename}")

def show_spectrogram(path):
    img = cv2.imread(path)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_spectrogram(spectrogram):
    # plot a spectrogram
    # flip 2d array
    spectrogram = np.transpose(spectrogram)
    plt.imshow(spectrogram, cmap='viridis')
    plt.colorbar()

    # Adding details to the plot
    plt.title("Spectrogram")
    plt.xlabel('time')
    plt.ylabel('frequency')

    # Displaying the plot
    plt.show()

def plot_audio(audio):
    sample_rate = 1000  # Assuming a sample rate of 1 kHz
    time = np.linspace(0, len(audio) / sample_rate, len(audio))

    # Plot the audio waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # set the id of the keyword thats being used
    keyword_id = 1

    if tiny_ml_connected:
        tiny_ml_serial = serial.Serial(port='COM5', baudrate=115200, timeout=0.1)
        print("connected to tiny ml")

        # use the same values as in AudioAndSpectrogramRead (sample_buffer_size, number_of_samples)
        audio, spectrogram = read_serial_data(128, 128)

        print("spectrogram shape")
        print(spectrogram.shape)

        # save data right keyword folder with correct naming
        # data/audio_1/audio_00x.csv
        # data/spectrogram_1/spectrogram_00x.png
        save_audio(audio, "data", "audio_" + str(keyword_id))
        save_spectrogram(spectrogram, "data", "spectrogram_" + str(keyword_id))

        # plot the audio and spectrogram
        plot_audio(audio)
        plot_spectrogram(spectrogram)
    exit()