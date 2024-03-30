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
// this should be a power of 2 i believe
#define SAMPLE_BUFFER_SIZE 128
#define SAMPLE_RATE 8000

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

// real and imaginary components    -> could be turned into floats
float vReal[SAMPLE_BUFFER_SIZE];
float vImag[SAMPLE_BUFFER_SIZE];

// mic buffer
int32_t raw_samples[SAMPLE_BUFFER_SIZE];


// Create FFT object
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, SAMPLE_BUFFER_SIZE, SAMPLE_RATE);

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

void loop(){
    // read read_microphone data, store in raw_samples
    int n_read_samples = read_microphone();

    // do ftt for read buffer -> stores in vReal
    do_fft(n_read_samples);

    // save vReal into spectogram as correct index
    write_to_spectogram(vReal, spectogram_index);

    // send spectogram when buffer is filled
    if(spectogram_index == 0){
        send_spectrogram();
        //test_python();
    }

    // option to send the microphone data
    // send_microphone_data();

    // short delay to prevent esp from exploding, maybe?
    delay(1);
}

int read_microphone(){
    // read from the I2S device
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, raw_samples, sizeof(int32_t) * SAMPLE_BUFFER_SIZE, &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int32_t);

    // reduce amplitude of signal
    for(int i = 0; i < SAMPLE_BUFFER_SIZE; i++){
      raw_samples[i] = raw_samples[i];
    }
    return samples_read;
}

void do_fft(int n_read_samples){
  for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++){
      vReal[i] = 0;
  }
  // prepare data
  for (int i = 0; i < n_read_samples; i++){
      vReal[i] = raw_samples[i];
      vImag[i] = 0;
  }

  //weigh data
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  // compute FFT
  FFT.compute(FFTDirection::Forward);
  // compute magnitudes
  FFT.complexToMagnitude();
}

void write_to_spectogram(float frequency_data[], int index){
  // at starting index of the spectogram we write the data
  // then increment the index -> do this in dedicated function
  increment_spectogram_index();

  // store each frequency_data point
  for(int i = 0; i < sizeof(frequency_data); i++){
    spectogram[spectogram_index][i] = frequency_data[i];
  }
}

void increment_spectogram_index(){
    // increment the spectogram_index
    spectogram_index += 1;
    // spectogram_index should not be greater than the number of samples    -> check for fenchpost error
    if(spectogram_index > N_O_SAMPLES - 1){
        spectogram_index = 0;
    }
}

void send_spectrogram(){
    Serial.println("s");
    for(int i = 0; i < N_O_SAMPLES; i++){
        for(int j = 0; j < SAMPLE_BUFFER_SIZE; j++){
            // send pixel data, from top left going through the lines
            Serial.print(i);
            Serial.print(",");
            Serial.println(int(spectogram[i][j]));
        }
    }
    Serial.println("e");
}
void test_python(){
    Serial.println("s");
    for(int i = 0; i < N_O_SAMPLES; i++){
        for(int j = 0; j < SAMPLE_BUFFER_SIZE; j++){
            // send pixel data, from top left going through the lines
            Serial.print(i);
            Serial.print(",");
            Serial.println(i * j);
        }
    }
    Serial.println("e");  
}

void send_microphone_data(){
  //print all samples of the mic reading. send constant heigth bars for arduino plotter to have better magnitude read
  for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++){
    Serial.print(100000000);
    Serial.printf(",");
    Serial.print(20000000);
    Serial.printf(",");
    Serial.print(-20000000);
    Serial.printf(",");
    Serial.print(-100000000);
    Serial.printf(",");
    Serial.printf("%ld\n", raw_samples[i]);
  }
}

bool read_button(int pin){
  if (!digitalRead(pin)) {
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += !digitalRead(pin);
    }
    if (sum == 8) {
        return true;
    }
  }
  return false;
}

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
