import pyaudio
import wave
import numpy as np
import os

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

            # limit to one subfolder for testing
            if subfolder != "audio_1":
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


def get_single_audio_data(file_path):
    """
    Read a single audio data file from the specified path.

    Args:
        file_path (str): The full path to the CSV file containing the audio data.

    Returns:
        numpy.ndarray: The audio data read from the CSV file, or None if the file is not in the expected format.
        int: The label extracted from the filename, or None if the file is not in the expected format.
    """
    # Check if the file exists and is a CSV file
    if os.path.isfile(file_path) and file_path.endswith(".csv"):
        # Extract the label from the filename
        filename = os.path.basename(file_path)
        label = int(filename.split("audio_")[1].split(".")[0])
        audio_sample = np.genfromtxt(file_path, delimiter=',')
        return audio_sample, label
    else:
        return None, None


if __name__ == "__main__":

    input_path = "data/augmentedAudio_1/audio_019.csv"
    one_audio_sample, label = get_single_audio_data(input_path)

    if one_audio_sample is not None:
        output_path = 'data/wav/output.wav'

        # Create a WAV file
        wav_file = wave.open(output_path, 'w')
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(2000)  # 9khz
        wav_file.writeframes(one_audio_sample.tobytes())
        wav_file.close()

        # Play the audio using PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=9000,
                        output=True)

        stream.write(one_audio_sample.astype(np.float32).tobytes())

        stream.stop_stream()
        stream.close()
        p.terminate()
    else:
        print("Failed to read audio data.")