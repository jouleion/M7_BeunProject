#include <driver/i2s.h>
#include <FastLED.h>

#include "microphoneConfig.h"
#include "microphoneHandler.h"
#include "apiHandler.h"
#include "cnnModel.h"

// include wifi passwords
#include "credentials/credentials.h"
//#include "credentials/credentials_uni.h"

// pin def
#define pin_button1 0
#define pin_button2 47
#define pin_led 17
#define pin_neopixel 18

// other def
#define NUM_LEDS 5
#define NUM_KEYWORDS 5


// setup cnn model
cnnModel CNN(128, 128);

// setup microphone
microphoneHandler micHandler;


const float enschedeLat = 52.2215;
const float enschedeLng = 6.8936;

// setup the api
apiHandler api(ssid, password, enschedeLat, enschedeLng);

// setup neopixels
CRGB leds[NUM_LEDS];

int keywordId = 0;


void setup() {
  // start serial communication
  Serial.begin(115200);
  delay(100);

  // init neopixels
  FastLED.addLeds<NEOPIXEL, pin_neopixel>(leds, NUM_LEDS);
  FastLED.setBrightness(50);
}

void loop() {
    // read microphone data for one row
    micHandler.readAudio();

    // compute spectrogram for each audio sample that is recorded
    micHandler.computeSpectrogramRow();

    // continue reading the audio, when spectrogram is not ready
    if (!micHandler.isSpectrogramReady()) {
        digitalWrite(pin_led, HIGH);
        return;  
    }

    // this only runs when spectrogram is ready

    // signify that the tiny ml is done reading audio
    digitalWrite(pin_led, LOW);

    // send the normalized audio buffer
    //micHandler.sendAudio();
    // send the normalized spectrogram
    //micHandler.sendSpectrogram();

    // if you only want to do 1 prediction
    //while (1) {
    //    delay(100);
    //}

    // get the prediction from cnn model
    //float* output = CNN.predict(micHandler.getSpectrogram());
  
    // Find the index of maximum value in the output array, convert to keyword id (+1)
    //keywordId = getMaxIndex(output) + 1;
  
    Serial.print("Keyword:");
    Serial.println(keywordId);
  
    switch(keywordId){
        case 1:
            Serial.print(api.getSunsetTime());
            // output to led matrix
            break;
        case 2:
            api.getSunsetTime();
            // output to led matrix
            break;
        case 3:
            api.getSunsetTime();
            // output to led matrix
            break;
  }  
  delay(25);
}

int getMaxIndex(float* values){
    float maxValue = values[0];
    int maxIndex = 0;
    for (int i = 1; i < NUM_KEYWORDS; i++) {
        if (values[i] > maxValue) {
            maxValue = values[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
