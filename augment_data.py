"""

"""
# import serial
# import time
import numpy as np
import matplotlib.pyplot as plt
# import re
import cv2
# import os
# import csv
# from record_data import save_spectrogram, save_audio

def augment_spectrogram(spectrogram):
    noise_spectrogram = noisy(spectrogram, prob=np.random.uniform(0.01, 0.04))
    tilt_spectrogram = rotate_image(noise_spectrogram, np.random.randint(-7,7))
    return tilt_spectrogram


def noisy(img, prob, noise_type="sp"): #https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == "gauss":
        image = img.copy()
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image, gauss)
        return image

    elif noise_type == "sp":
        image = img.copy()
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image

def rotate_image(image, angle): #https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def plot_two_digits(digits):
    plt.figure(figsize=(13, 6))
    axislabels = ["Unprocessed", "Processed"]
    for i in range(2):
        img = cv2.resize(digits[i], (128, 128))
        digit_sub = plt.subplot(2, 2, i + 1)
        digit_sub.imshow(img, cmap="gray")
        digit_sub.set_xlabel(f"{axislabels[i]}")

    plt.show()



if __name__ == "__main__":
    original_spectrogram = cv2.imread('data/spectrogram_1/spectrogram_001.png', cv2.IMREAD_GRAYSCALE)
    augmented_spectrogram = augment_spectrogram(original_spectrogram)
    spectrograms = [original_spectrogram, augmented_spectrogram]
    plot_two_digits(spectrograms)