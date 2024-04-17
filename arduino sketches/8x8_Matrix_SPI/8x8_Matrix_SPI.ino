#include <SPI.h>

//#define pin_spi_mosi 11
//#define pin_spi_sck 12
//#define pin_spi_miso 13
//#define pin_spi_cs 14

const byte LATCH = 12;  //ss
// mosi 11 -> DS
// sck 13  -> SH

void setup (){
    SPI.begin ();
}

void loop() {
  for (int j = 0; j <= 0xFF; j++) {
    for(int i = 0; i <= 0xFF; i++){
      digitalWrite(LATCH, LOW);
      SPI.transfer(~i);
      SPI.transfer(~j);
      digitalWrite(LATCH, HIGH);
    }
  }
}
