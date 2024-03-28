"""
This is used to record the microphone data from Wilson

following this tutorial to fetch serial data:
https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756

"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    while True:
        # if arduino_serial.is_open:
        times, fft_values = read_serial()
        print(times)
        print(fft_values)

        #plot values
        x = np.array(times)
        y = np.array(fft_values)

        # Convert values to float
        x = list(map(float, x))
        y = list(map(float, y))

        # Plot the data
        plt.plot(x, y)

        # Add labels and title
        plt.xlabel('frequency')
        plt.ylabel('magnitude')
        plt.title('Wilson FFT')

        # Display the plot
        plt.show()


# save to json