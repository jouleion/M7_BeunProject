import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.fft import fft, fft2, fftshift

def get_audio_data(data_path):
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
                if split_name[0] == "augmentedAudio":
                    label = int(split_name[1])
                    for filename in os.listdir(subfolder_path):
                        if filename.endswith(".csv"):
                            print(filename)
                            audio_path = os.path.join(subfolder_path, filename)
                            audio_sample = np.genfromtxt(audio_path, delimiter=',')
                            X.append(audio_sample)
                            y.append(label)
                else:
                    continue
            else:
                # skip folder if in wrong format
                print("Skipped a folder, has wrong format")
                continue

    return X, y

def save_spectrogram(spectrogram, data_path, keyword_id):
    # save spectrogram in correct folder depending on the keyword id thats being used

    # put in right subfolder
    subfolder_path = os.path.join(data_path, f"augmentedSpectrogram_{keyword_id}")

    # make subfolder if it did not exist yet
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


import numpy as np
from scipy.fft import fft, fftshift

def generate_spectrogram(audio_data):
    # Ensure the input audio_data has the correct length
    expected_length = 128 * 128
    if len(audio_data) != expected_length:
        raise ValueError(f"Input audio_data must have length {expected_length}")

    # Reshape audio_data into a 1D array of shape (16384,)
    audio_data = np.reshape(audio_data, (16384,))

    # Initialize the spectrogram array
    spectrogram = np.zeros((128, 128), dtype=np.float32)

    # Compute the spectrogram row by row
    for i in range(128):
        start = i * 128
        end = start + 128
        row_data = audio_data[start:end]

        # Apply a Hanning window to the row data
        window = np.hanning(128)
        row_data_windowed = row_data * window

        # Compute the FFT and get the magnitude
        row_fft = fft(row_data_windowed)
        row_magnitude = np.abs(row_fft)

        # Store the row in the spectrogram
        spectrogram[i] = row_magnitude

    # Normalize the spectrogram
    min_value = np.min(spectrogram)
    max_value = np.max(spectrogram)
    spectrogram = (spectrogram - min_value) / (max_value - min_value) * 255
    spectrogram = np.uint8(spectrogram)

    return spectrogram


if __name__ == "__main__":
    # Define your data path
    audio_path = "data"
    spectrogram_path = "data"

    # Get audio data
    X, y = get_audio_data(audio_path)

    # Process each audio sample and generate spectrogram
    for i, audio_sample in enumerate(X):
        # Assuming audio_sample is normalized from -1 to 1, and has length 128x128

        # Generate spectrogram
        spectrogram = generate_spectrogram(audio_sample)

        # Save spectrogram
        save_spectrogram(spectrogram, spectrogram_path, y[i])