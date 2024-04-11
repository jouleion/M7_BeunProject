// MicrophoneHandler.h
#ifndef MICROPHONEHANDLER_H
#define MICROPHONEHANDLER_H

#include "MicrophoneConfig.h"
#include "arduinoFFT.h"

class MicrophoneHandler {
public:
    MicrophoneHandler();
    ~MicrophoneHandler();

    void readAudio();
    void sendAudio();
    bool isSpectrogramReady();
    void computeSpectrogramRow();
    void sendSpectrogram();
    

private:
    void doFFT();
    void writeToSpectrogram(float* frequency_data);
    void incrementSpectrogramIndex();

    int32_t raw_samples[SAMPLE_BUFFER_SIZE];
    int32_t audio[AUDIO_BUFFER_SIZE];
    float vReal[SAMPLE_BUFFER_SIZE];
    float vImag[SAMPLE_BUFFER_SIZE];
    float spectrogram[N_O_SAMPLES][SAMPLE_BUFFER_SIZE];
    int spectrogramIndex = 0;
    int audioIndex = 0;
    ArduinoFFT<float>* FFT;
};

#endif // MICROPHONEHANDLER_H
