// get serial input from python, add one to it and return it.
// baseline serial connection
// It works but has a lot of delay for the ESP32 dev module S3

int x;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
}

void  loop() {
  while (!Serial.available());
  
  x = Serial.readString().toInt();
  Serial.print(x + 1);
}
