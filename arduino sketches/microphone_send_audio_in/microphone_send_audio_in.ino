// This is used to send audio data over serial to python
// Data collection of keywords
// light = red means that the tiny ml is recording. otherwise it is sending data over serial
// tiny ml settings: CDC on, PSRAM: QSPI PSRAM, Flash size: 16mb

#include <driver/i2s.h>
#include "arduinoFFT.h"

// pin def
#define pin_button1 0
#define pin_button2 47
#define pin_led 17

// some timing for fft
#define SCL_INDEX 0x00
#define SCL_TIME 0x01
#define SCL_FREQUENCY 0x02
#define SCL_PLOT 0x03

// buffer for fft
#define SAMPLE_BUFFER_SIZE 128
#define SAMPLE_RATE 9000
#define N_O_SAMPLES 128

// cool mic initialisation!
#include "mic_header.h"

// real and imaginary components
double vReal[SAMPLE_BUFFER_SIZE];
double vImag[SAMPLE_BUFFER_SIZE];

// mic buffer
int32_t raw_samples[SAMPLE_BUFFER_SIZE];
//int32_t spectogram[SAMPLE_BUFFER_SIZE][N_O_SAMPLES];

// Create FFT object
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, SAMPLE_BUFFER_SIZE, SAMPLE_RATE);

void setup(){
  // start up the I2S peripheral
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &i2s_mic_pins);
  
  // we need serial output for the plotter
  Serial.begin(115200);

  delay(2000);

  // blink so user can prepare to speak
  pinMode(pin_led, OUTPUT);
  digitalWrite(pin_led, HIGH);
  delay(200);
  digitalWrite(pin_led, LOW);
  delay(1000);
  digitalWrite(pin_led, HIGH);

  //button
  pinMode(pin_button1, INPUT_PULLUP);

  Serial.println("s");
}

int iterations = 0;
void loop(){
    // read microphone
    read_microphone();

    // send audio sample
    for(int i = 0; i < SAMPLE_BUFFER_SIZE; i++){
        //raw_samples[i] = raw_samples[i] >> 16;
        
        // reduce amplitude of signal
        Serial.printf("%ld\n", raw_samples[i]>>8);
    }
    
    // count how many samples have been send
    iterations++;

    // after 128 send audio samples stop sending audio data
    if(iterations > N_O_SAMPLES){
        // print ending character
        Serial.println("e");

        // indicate end of audio send
        digitalWrite(pin_led, LOW);
        while (1){
            // loop forever
            delay(100);
        }
    }
}

void read_microphone(){
    // read from the I2S device
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, raw_samples, sizeof(int32_t) * SAMPLE_BUFFER_SIZE, &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int32_t);
}
