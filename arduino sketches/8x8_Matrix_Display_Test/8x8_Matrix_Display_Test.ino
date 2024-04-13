#include <SPI.h>

// Pins for the shift register
const int latchPin = 11;
const int clockPin = 12;
const int dataPin = 13;

// 8-bit binary counter
byte counter = 0;

void setup() {
  pinMode(latchPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  SPI.begin();
}

void loop() {
  // Increment the counter
  counter++;
  if (counter > 255) {
    counter = 0;
  }

  // Update the shift register
  updateShiftRegister();
  delay(500);
}

void updateShiftRegister() {
  // Latch the data to the shift register
  digitalWrite(latchPin, LOW);
  SPI.transfer(~counter);
  digitalWrite(latchPin, HIGH);
}
