// MicrophoneHandler.cpp
#include "MicrophoneHandler.h"
#include "MicrophoneConfig.h"

MicrophoneHandler::MicrophoneHandler() {
    // start up the I2S peripheral
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &i2s_mic_pins);

    // create FFT object (sample buffer size must be a power of 2)
    FFT = new ArduinoFFT<float>(vReal, vImag, SAMPLE_BUFFER_SIZE, SAMPLE_RATE, true);
}

MicrophoneHandler::~MicrophoneHandler() {
    delete FFT;
}

void MicrophoneHandler::readAudio() {
    // read from the I2S device
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, raw_samples, sizeof(int32_t) * SAMPLE_BUFFER_SIZE, &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int32_t);

    // reduce amplitude of signal and store in audio buffer
    for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
        // store audio in audio buffer aswell
        audio[audioIndex++] = raw_samples[i] >> 8;

        // overwrite audio buffer
        if (audioIndex >= AUDIO_BUFFER_SIZE) {
            // reset audio buffer to zero when it does this
            audioIndex = 0;
        }
    }
}

void MicrophoneHandler::sendAudio() {
    // normalize audio
    normalizeAudio();
  
    // send audio buffer over serial, with leading 'a' and ending 'e'
    Serial.println("a");
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        Serial.println(normalized_audio[i], 7);
    }
    Serial.println("e");
}

bool MicrophoneHandler::isSpectrogramReady() {
    // when all rows of spectrogram have been filled, send spectrogram
    if (spectrogramIndex == N_O_SAMPLES - 1) {
        return true;
    }
    return false;
}

void MicrophoneHandler::normalizeAudio() {
    // Normalize the audio
    float mean = 0.0f;
    float max_val = -32768.0f; // Minimum possible value for int16_t
    float min_val = 32767.0f; // Maximum possible value for int16_t

    // Calculate the mean, max, and min of the audio buffer
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        mean += float(audio[i]);
        max_val = max(max_val, float(audio[i]));
        min_val = min(min_val, float(audio[i]));
    }
    mean /= AUDIO_BUFFER_SIZE;

    // Normalize the audio buffer
    for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
        normalized_audio[i] = float((audio[i] - mean) / (max_val - min_val) * 2);
    }
}

void MicrophoneHandler::computeSpectrogramRow() {
    // do FFT on the read data
    doFFT();

    // write FFT data to spectrogram
    writeToSpectrogram(vReal);
    incrementSpectrogramIndex();
}

void MicrophoneHandler::normalizeSpectrogram() {
    // Find the minimum and maximum values in the spectrogram
    float min_val = 32767.0f; // Maximum possible value for int16_t
    float max_val = -32768.0f; // Minimum possible value for int16_t
    for (int i = 0; i < N_O_SAMPLES; i++) {
        for (int j = 0; j < SAMPLE_BUFFER_SIZE; j++) {
            min_val = min(min_val, spectrogram[i][j]);
            max_val = max(max_val, spectrogram[i][j]);
        }
    }

    // Normalize the spectrogram values to the range [0, 255]
    for (int i = 0; i < N_O_SAMPLES; i++) {
        for (int j = 0; j < SAMPLE_BUFFER_SIZE; j++) {
            spectrogram[i][j] = (spectrogram[i][j] - min_val) / (max_val - min_val) * 255.0f;
            spectrogram[i][j] = static_cast<uint8_t>(spectrogram[i][j]);
        }
    }
}


void MicrophoneHandler::sendSpectrogram() {
    // normalize to 0 - 255
    normalizeSpectrogram();

    // sprint all cells
    Serial.println("s");
    for (int i = 0; i < N_O_SAMPLES; i++) {
        for (int j = 0; j < SAMPLE_BUFFER_SIZE; j++) {
            // send pixel data, from top left going through the lines
            Serial.print(i);
            Serial.print(",");
            Serial.println(int(spectrogram[i][j]));
        }
    }
    Serial.println("e");
}

void MicrophoneHandler::doFFT() {
    // load data for FFT
    for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
        vReal[i] = raw_samples[i];
        vImag[i] = 0;
    }

    // weigh data and compute FFT
    FFT->windowing(FFTWindow::Hamming, FFTDirection::Forward);
    FFT->compute(FFTDirection::Forward);
    FFT->complexToMagnitude();
}

void MicrophoneHandler::writeToSpectrogram(float* frequency_data) {
    // store each frequency_data point
    // directly write to the pointer of the spectrogram
    for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
        spectrogram[spectrogramIndex][i] = frequency_data[i];
    }
}

void MicrophoneHandler::incrementSpectrogramIndex() {
    // increment the spectrogram_index
    spectrogramIndex = (spectrogramIndex + 1) % N_O_SAMPLES;
}
