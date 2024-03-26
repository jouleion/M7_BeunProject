#include <driver/i2s.h>
#include "arduinoFFT.h"

#define SAMPLE_BUFFER_SIZE 512
#define SAMPLE_RATE 8000

#include "mic_header.h"

#define pin_button1 0
#define pin_button2 47
#define pin_led 17

#define SCL_INDEX 0x00
#define SCL_TIME 0x01
#define SCL_FREQUENCY 0x02
#define SCL_PLOT 0x03

// real and imaginary components
double vReal[SAMPLE_BUFFER_SIZE];
double vImag[SAMPLE_BUFFER_SIZE];



/* Create FFT object */
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, SAMPLE_BUFFER_SIZE, SAMPLE_RATE);

void setup()
{
  // we need serial output for the plotter
  Serial.begin(115200);
  Serial.print("Hi");
  pinMode(pin_led, OUTPUT);
  digitalWrite(pin_led, HIGH);

  //button
  pinMode(pin_button1, INPUT_PULLUP);
  
  // start up the I2S peripheral
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &i2s_mic_pins);
}

int32_t raw_samples[SAMPLE_BUFFER_SIZE];

void loop(){
  // read from the I2S device
  size_t bytes_read = 0;
  i2s_read(I2S_NUM_0, raw_samples, sizeof(int32_t) * SAMPLE_BUFFER_SIZE, &bytes_read, portMAX_DELAY);
  
  int samples_read = bytes_read / sizeof(int32_t);
  
  // dump the samples out to the serial channel.
//  for (int i = 0; i < samples_read; i++){
//    Serial.print(100000000);
//    Serial.printf(",");
//    Serial.print(20000000);
//    Serial.printf(",");
//    Serial.print(-20000000);
//    Serial.printf(",");
//    Serial.print(-100000000);
//    Serial.printf(",");
//    Serial.printf("%ld\n", raw_samples[i]);
//    
//  }

  //on button press
  if(read_button(pin_button1)){
    fft_doemaar(samples_read);
  }
}

void fft_doemaar(int samples_read){
  // prepare data
  for (int i = 0; i < samples_read; i++){
      vReal[i] = raw_samples[i];
      vImag[i] = 0;
  }  
  
  /* Print the results of the simulated sampling according to time */
  Serial.println("Data:");
  PrintVector(vReal, SAMPLE_BUFFER_SIZE, SCL_TIME);
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);  /* Weigh data */
  
  Serial.println("Weighed data:");
  PrintVector(vReal, SAMPLE_BUFFER_SIZE, SCL_TIME);
  FFT.compute(FFTDirection::Forward); /* Compute FFT */
  
  Serial.println("Computed Real values:");
  PrintVector(vReal, SAMPLE_BUFFER_SIZE, SCL_INDEX);
  Serial.println("Computed Imaginary values:");
  PrintVector(vImag, SAMPLE_BUFFER_SIZE, SCL_INDEX);
  FFT.complexToMagnitude(); /* Compute magnitudes */
  
  Serial.println("Computed magnitudes:");
  PrintVector(vReal, (SAMPLE_BUFFER_SIZE >> 1), SCL_FREQUENCY);
  double x = FFT.majorPeak();
  Serial.println(x, 6);
  while(1); /* Run Once */  

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
void PrintVector(double *vData, uint16_t bufferSize, uint8_t scaleType)
{
  for (uint16_t i = 0; i < bufferSize; i++)
  {
    double abscissa;
    /* Print abscissa value */
    switch (scaleType)
    {
      case SCL_INDEX:
        abscissa = (i * 1.0);
  break;
      case SCL_TIME:
        abscissa = ((i * 1.0) / SAMPLE_RATE);
  break;
      case SCL_FREQUENCY:
        abscissa = ((i * 1.0 * SAMPLE_RATE) / SAMPLE_BUFFER_SIZE);
  break;
    }
    Serial.print(abscissa, 6);
    if(scaleType==SCL_FREQUENCY)
      Serial.print("Hz");
    Serial.print(" ");
    Serial.println(vData[i], 4);
  }
  Serial.println();
}
