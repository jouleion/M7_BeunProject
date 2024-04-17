// Writen by Julian van der Sluis with help from perplexity AI

#ifndef APIHANDLER_H
#define APIHANDLER_H

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

class apiHandler {
public:
    apiHandler(const char* ssid, const char* password, float latitude, float longitude);
    void setup();
    String getSunsetTime();

private:
    const char* _ssid;
    const char* _password;
    const char* _serverName;
    float _latitude;
    float _longitude;
    unsigned long _lastTime;
    unsigned long _timerDelay;

    DynamicJsonDocument getJsonFromUrl(String url);
};

#endif // APIHANDLER_H
