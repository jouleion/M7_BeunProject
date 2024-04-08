"""
This is used to record the microphone data from Wilson
either as a spectrogram or as an audio array



following this tutorial to fetch serial data:
https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756

"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
import os


# set true when tiny ml is conected
tiny_ml_connected = False

if tiny_ml_connected:
    tiny_ml_serial = serial.Serial(port='COM5',  baudrate=115200, timeout=0.1)


def read_serial():
    times = []
    fft_values = []

    if tiny_ml_connected:
        ended = False
    else:
        ended = True
        print("Please connect Tiny Ml board")

    while not ended:
        # Read and decode the incoming data
        data = tiny_ml_serial.readline().decode().strip()

        # Check if data is not empty
        if data:
            # check if end signifier is read
            if data == "End,":
                print("End of serial read")
                break

            # Split the data by comma and convert to integers and convert to int
            time_val, fft_val = data.split(',')

            #temp
            print(time_val)
            print(fft_val)

            # store in array
            times.append(time_val)
            fft_values.append(fft_val)

    return times, fft_values

def remove_letters(input_string):
    return re.sub("[^0-9]", "", input_string)

def read_spectrogram():
    # Initialize a numpy array filled with zeros
    # N_O_SAMPLES, SAMPLE_BUFFER_SIZE
    spectrogram = np.zeros((128, 128))
    #spectrogram[100, 20] = 1

    if tiny_ml_serial.is_open:
        end_of_image = False
        start_of_image = True
        column_index = 0
        last_row = 0

        while not end_of_image:
            # Read and decode the incoming data
            pixel = tiny_ml_serial.readline().decode().strip()

            if pixel:
                print(pixel)
                if pixel == "s":
                    start_of_image = True
                    print("start of image")
                    end_of_image = False
                    continue

                if start_of_image:
                    if pixel == "e":
                        print("end of image")
                        end_of_image = True
                        continue

                    if "," in pixel:
                        # split image at ","
                        row, frequency_strength = pixel.split(",")
                        row = remove_letters(row)
                        frequency_strength = remove_letters(frequency_strength)

                        if row != last_row:
                            # Reset index for column when a new row is send
                            column_index = 0

                        #print(frequency_strength)

                        # Store pixel at correct coordinate in np.array
                        spectrogram[int(row)][column_index] = frequency_strength
                        column_index += 1
                        last_row = row

                else:
                    end_of_image = True

    # normalize spectrogram using min max scaling
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - 50 - min_val + 0.01)

    # return the spectrogram
    return spectrogram

def save_spectrogram_blind(spectrogram):
    cv2.imwrite("spectrogram_001.png", spectrogram)

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
        if file.startswith("audio_") and file.endswith(".png"):             # !!!!!!!!!! This is saves as .npy file, change this!
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
    np.save(new_file_path, audio_array)
    print(f"Spectrogram saved as {new_filename}")

def show_spectrogram(path):
    img = cv2.imread(path)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        if tiny_ml_connected:
            spectrogram = read_spectrogram()
        else:
            spectrogram = np.random.randint(255, size = (128,128,1))


        keyword_id = 1
        save_spectrogram(spectrogram, "data", "spectrogram_" + str(keyword_id))

        audio = [1, 2, 3, 1, 3, 1, 3, 2,4 ,4, 5, 3, 5, 3 ,7 ,4 ,8 ,4 ,7 ,4 ,6 ,4,3 ,4 ,5, 7 ,6 ,5 ,5 ,5 ,5]
        save_audio(audio, "data", "audio_" + str(keyword_id))



        #print(spectrogram)
        # # if arduino_serial.is_open:
        # times, fft_values = read_serial()
        # print(times)
        # print(fft_values)
        #
        # #plot values
        # x = np.array(times)
        # y = np.array(fft_values)
        #
        # # Convert values to float
        # x = list(map(float, x))
        # y = list(map(float, y))
        #

        # Plot the data
        # Function to show the heat map

        # plt.imshow(np.transpose(spectrogram), cmap='viridis')
        # plt.colorbar()
        #
        # # Adding details to the plot
        # plt.title("Spectrogram")
        # plt.xlabel('x-axis')
        # plt.ylabel('y-axis')
        #
        # # Displaying the plot
        # plt.show()
        # exit()


# save to json