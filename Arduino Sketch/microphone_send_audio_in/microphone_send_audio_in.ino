// just shows the fft over time
// please work further in microphone processed

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
#define SAMPLE_BUFFER_SIZE 256
#define SAMPLE_RATE 8000
#define N_O_SAMPLES 10

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
  // we need serial output for the plotter
  Serial.begin(115200);

  pinMode(pin_led, OUTPUT);
  digitalWrite(pin_led, HIGH);

  //button
  pinMode(pin_button1, INPUT_PULLUP);

  // start up the I2S peripheral
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &i2s_mic_pins);
}

int iterations = 0;
void loop(){
    if(iterations > 500){
        digitalWrite(pin_led, LOW);
        while (1){
            delay(100);
        }
    }
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, raw_samples, sizeof(int32_t) * SAMPLE_BUFFER_SIZE, &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int32_t);

    for(int i = 0; i < SAMPLE_BUFFER_SIZE; i++){
        // reduce amplitude of signal
        //raw_samples[i] = raw_samples[i] >> 16;
        Serial.printf("%ld\n", raw_samples[i]>>16);
    }

}

// void send_microphone_data(int samples_read){
//   //print all samples of the mic reading. send constant heigth bars for arduino plotter to have better magnitude read
//   for (int i = 0; i < samples_read; i++){
// //     Serial.print(100000000);
// //     Serial.printf(",");
// //     Serial.print(20000000);
// //     Serial.printf(",");
// //     Serial.print(-20000000);
// //     Serial.printf(",");
// //     Serial.print(-100000000);
// //     Serial.printf(",");
//     Serial.printf("%ld\n", raw_samples[i]>>16);
//   }
// }

// void do_fft(int samples_read){
//   // prepare data
//   for (int i = 0; i < samples_read; i++){
//       vReal[i] = raw_samples[i];
//       vImag[i] = 0;
//   }
//
//   //weigh data
//   FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
//   // compute FFT
//   FFT.compute(FFTDirection::Forward);
//   // compute magnitudes
//   FFT.complexToMagnitude();
//
//   //PrintVector(vReal, (SAMPLE_BUFFER_SIZE >> 1), SCL_FREQUENCY);
// }
// bool read_button(int pin){
//   if (!digitalRead(pin)) {
//     int sum = 0;
//     for (int i = 0; i < 8; i++) {
//         sum += !digitalRead(pin);
//     }
//     if (sum == 8) {
//         return true;
//     }
//   }
//   return false;
// }
// void PrintVector(double *vData, uint16_t bufferSize, uint8_t scaleType)
// {
//   for (uint16_t i = 0; i < bufferSize; i++){
//     double abscissa;
//     /* Print abscissa value */
//     switch (scaleType){
//       case SCL_INDEX:
//         abscissa = (i * 1.0);
//         break;
//       case SCL_TIME:
//         abscissa = ((i * 1.0) / SAMPLE_RATE);
//         break;
//       case SCL_FREQUENCY:
//         abscissa = ((i * 1.0 * SAMPLE_RATE) / SAMPLE_BUFFER_SIZE);
//         break;
//     }
//     // print standard height line
//     Serial.print(10000000);
//     Serial.print(",");
//
//     Serial.print(abscissa, 6);
//     if(scaleType==SCL_FREQUENCY)
//       Serial.print("Hz");
//     Serial.print(" ");
//     Serial.println(vData[i], 4);
//   }
//   Serial.println();
// }

// this was used to send empty line after fft has been send, didnt work great?
//   for(int i = 0; i < 255; i++){
//     Serial.print("");
//
//     // print standard height line
//     Serial.print(10000000);
//     Serial.print((i * 1.0 * SAMPLE_RATE) / SAMPLE_BUFFER_SIZE);
//     Serial.print("Hz");
//     Serial.print(",");
//     Serial.println(0, 4);
//   }
