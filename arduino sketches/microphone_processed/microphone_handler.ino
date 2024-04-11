#include <driver/i2s.h>
#include "arduinoFFT.h"

// some timing for fft
#define SCL_INDEX 0x00
#define SCL_TIME 0x01
#define SCL_FREQUENCY 0x02
#define SCL_PLOT 0x03

// buffer for fft
// this should be a power of 2 i believe
#define SAMPLE_BUFFER_SIZE 128
#define SAMPLE_RATE 9000

// this makes a 256x128 pixel image = 32 KB
// 128x128 = 16KB
#define N_O_SAMPLES 128

// cool mic initialisation! -> requires sample rate
#include "mic_header.h"

// Buffer should be read as follows:
// save at index i = 0
// increment index.
// read buffer starting from index to indicate starting position.

// do we use floats, doubles or ints?
float spectogram[N_O_SAMPLES][SAMPLE_BUFFER_SIZE];

// keep track of current spectogram index
int spectogram_index = 0;

// real and imaginary components for fft transform
float vReal[SAMPLE_BUFFER_SIZE];
float vImag[SAMPLE_BUFFER_SIZE];

// mic buffer
int32_t raw_samples[SAMPLE_BUFFER_SIZE];


class MicrophoneHandler {
private:
  int microphonePin;
  float audioData[1024];
  float spectrogramData[1024];

  // FFT-related variables
  float vReal[SAMPLE_BUFFER_SIZE];
  float vImag[SAMPLE_BUFFER_SIZE];
  ArduinoFFT<float> FFT;

public:
  MicrophoneHandler(int pin) : microphonePin(pin), FFT(vReal, vImag, SAMPLE_BUFFER_SIZE, SAMPLE_RATE, true) {
    // Initialize the FFT object in the constructor
  }

  void getAudio() {
    for (int i = 0; i < 1024; i++) {
      audioData[i] = analogRead(microphonePin);
    }
  }

  void getSpectrogram() {
    // Populate the vReal array with the audio data
    for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
      vReal[i] = static_cast<float>(audioData[i]);
      vImag[i] = 0; // Initialize the imaginary part to 0
    }

    // Perform the FFT
    FFT.Compute();

    // Update the spectrogramData array with the FFT results
    for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
      spectrogramData[i] = FFT.GetRealPart(i);
    }
  }

  float* getAudioData() {
    return audioData;
  }

  float* getSpectrogramData() {
    return spectrogramData;
  }
};
