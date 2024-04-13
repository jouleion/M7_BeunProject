// Writen by Julian van der Sluis with help from perplexity AI

#include "apiHandler.h"

apiHandler::apiHandler(const char* ssid, const char* password, float latitude, float longitude) {
    _ssid = ssid;
    _password = password;
    _latitude = latitude;
    _longitude = longitude;
}

void apiHandler::setup() {
    Serial.begin(115200);
    WiFi.begin(_ssid, _password);

    Serial.print("\nDefault ESP32 MAC Address: ");
    Serial.println(WiFi.macAddress());

    Serial.println("Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.print("Connected to WiFi network with IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.println("Timer set to 5 seconds (timerDelay variable), it will take 5 seconds before publishing the first reading.");
}

String apiHandler::getSunsetTime() {
    // Check WiFi connection status
    if (WiFi.status() == WL_CONNECTED) {
        // if connected, get json from api
        String api_url = "https://api.sunrisesunset.io/json?lat=" + String(_latitude) + "&lng=" + String(_longitude);
        DynamicJsonDocument json_data = getJsonFromUrl(api_url);

        // extract sunset time
        const char* sunsetTime = json_data["results"]["sunset"];

        // return sunset time
        return sunsetTime;
    } else {
        Serial.println("WiFi Disconnected");
        return "Error: WiFi Disconnected";
    }
}

DynamicJsonDocument apiHandler::getJsonFromUrl(String url) {
    // input a url, and return the json object!
    HTTPClient http;
    http.begin(url);
    int httpResponseCode = http.GET();

    DynamicJsonDocument doc(1024);
    if (httpResponseCode > 0) {
        String payload = http.getString();
        deserializeJson(doc, payload);
    } else {
        Serial.print("Error code: ");
        Serial.println(httpResponseCode);
    }

    http.end();
    return doc;
}
