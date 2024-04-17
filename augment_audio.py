import os
import numpy as np
import librosa
import sounddevice as sd

def augment_audio(audio_data, sample_rate, pitch_shift_factors, time_shift_factors, noise_levels):
    """Augment the audio data using various techniques."""

    print("Adding effects to a sample")

    augmented_audio = []
    for pitch_shift_factor in pitch_shift_factors:
        pitch_shifted_audio = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=pitch_shift_factor)
        pitch_shifted_audio = np.clip(pitch_shifted_audio, -1, 1)
        augmented_audio.append(pitch_shifted_audio)

    for time_shift_factor in time_shift_factors:
        time_shifted_audio = np.roll(audio_data, int(sample_rate * time_shift_factor))
        time_shifted_audio = np.clip(time_shifted_audio, -1, 1)
        augmented_audio.append(time_shifted_audio)

    for noise_level in noise_levels:
        noise = np.random.randn(len(audio_data))
        noisy_audio = audio_data + noise_level * noise
        noisy_audio = np.clip(noisy_audio, -1, 1)
        augmented_audio.append(noisy_audio)

    return augmented_audio


def save_augmented_audio(data_path, audio_sample, label):
    """
    Save the augmented audio files to the appropriate folder.
    """
    # Create the output folder if it doesn't exist
    output_folder = os.path.join(data_path, f"augmentedAudio_{label}")
    os.makedirs(output_folder, exist_ok=True)

    # Augment the audio sample
    pitch_shift_factors = np.linspace(-2, 8, 10)
    time_shift_factors = np.linspace(-0.2, 0.2, 5)
    noise_levels = np.linspace(0.001, 0.03, 5)
    augmented_audio = augment_audio(audio_sample, sample_rate=2000, pitch_shift_factors=pitch_shift_factors, time_shift_factors=time_shift_factors, noise_levels=noise_levels)


    # Find the highest existing index and generate a new file name
    existing_files = os.listdir(output_folder)
    max_index = 0
    for file in existing_files:
        if file.startswith("audio_") and file.endswith(".csv"):
            try:
                index = int(file.split("_")[1].split(".")[0])
                max_index = max(max_index, index)
            except ValueError:
                pass


    # Save the augmented audio files
    for i, audio in enumerate(augmented_audio):
        new_index = max_index + i + 1
        new_filename = f"audio_{new_index:03d}.csv"
        file_path = os.path.join(output_folder, new_filename)
        np.savetxt(file_path, audio, delimiter=",")
        print("saved audio as:")
        print(file_path)


def get_audio_data(data_path):
    # Existing code to load the audio data
    X = []
    y = []

    for subfolder in os.listdir(data_path):
        subfolder_path = os.path.join(data_path, subfolder)

        if os.path.isdir(subfolder_path):
            split_name = subfolder.split("_")

            if len(split_name) > 1:
                if split_name[0] == "audio":
                    label = int(split_name[1])
                else:
                    continue
            else:
                print("Skipped a folder, has wrong format")
                continue

            for filename in os.listdir(subfolder_path):
                if filename.endswith(".csv"):
                    print(filename)

                    audio_path = os.path.join(subfolder_path, filename)
                    audio_sample = np.genfromtxt(audio_path, delimiter=',')
                    audio_sample = np.clip(audio_sample, -1, 1)  # Clip the audio sample to the range [-1, 1]
                    X.append(audio_sample)
                    y.append(label)

    return X, y

# Example usage
data_path = "data"
X, y = get_audio_data(data_path)

# Augment all audio samples
for i, audio_sample in enumerate(X):
    save_augmented_audio(data_path, audio_sample, y[i])

exit()