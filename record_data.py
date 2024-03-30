"""
This is used to record the microphone data from Wilson

following this tutorial to fetch serial data:
https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756

"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import re

arduino_serial = serial.Serial(port='COM5',  baudrate=115200, timeout=0.1)


def read_serial():
    times = []
    fft_values = []

    ended = False

    while not ended:
        # Read and decode the incoming data
        data = arduino_serial.readline().decode().strip()

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
    spectrogram = np.zeros((128, 128))
    #spectrogram[100, 20] = 1

    if arduino_serial.is_open:
        end_of_image = False
        start_of_image = True
        column_index = 0
        last_row = 0

        while not end_of_image:
            # Read and decode the incoming data
            pixel = arduino_serial.readline().decode().strip()

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
    spectrogram = (spectrogram - min_val) / (max_val - min_val + 0.01)

    # return the spectrogram
    return spectrogram


if __name__ == "__main__":
    while True:
        spectrogram = read_spectrogram()
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

        plt.imshow(np.transpose(spectrogram), cmap='viridis')
        plt.colorbar()

        # Adding details to the plot
        plt.title("Spectrogram")
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')

        # Displaying the plot
        plt.show()
        exit()


# save to json