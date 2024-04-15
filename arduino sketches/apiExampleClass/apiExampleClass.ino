// Writen by Julian van der Sluis with help from perplexity AI

#include "apiHandler.h"
#include "credentials/credentials.h"
//#include "credentials/credentials_uni.h"

const float enschedeLat = 52.2215;
const float enschedeLng = 6.8936;

apiHandler api(ssid, password, enschedeLat, enschedeLng);



void setup() {
    Serial.begin(9600);
    api.setup();
    Serial.println(api.getSunsetTime());
}

void loop() {
    
}
