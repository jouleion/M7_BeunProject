// main.ino
#include <driver/i2s.h>

// pin def
#define pin_button1 0
#define pin_button2 47
#define pin_led 17

// define buffer sizes
#define SAMPLE_BUFFER_SIZE 128
#define SAMPLE_RATE 2000
#define N_O_SAMPLES 128
#define AUDIO_BUFFER_SIZE (SAMPLE_BUFFER_SIZE * N_O_SAMPLES)

#include "MicrophoneHandler.h"

MicrophoneHandler micHandler;

void setup() {
    // start serial communication
    Serial.begin(115200);
    delay(1000);

    // blink LED to indicate start
    pinMode(pin_led, OUTPUT);
    digitalWrite(pin_led, HIGH);
    delay(200);
    digitalWrite(pin_led, LOW);
    delay(1000);
    digitalWrite(pin_led, HIGH);
    // human reaction time
    delay(200);

    // button setup
    pinMode(pin_button1, INPUT_PULLUP);
}

void loop() {
    // read microphone data
    micHandler.readAudio();

    // compute spectrogram for each audio sample that is recorded
    micHandler.computeSpectrogramRow();

    // check if spectrogram is ready (filled)
    if (micHandler.isSpectrogramReady()) {

        // signify that the tiny ml is done reading audio
        digitalWrite(pin_led, LOW);

        // send the normalized audio buffer
        micHandler.sendAudio();
        
        // send the normalized spectrogram
        micHandler.sendSpectrogram();
        
        while (1) {
            delay(100);
        }
    }
}
