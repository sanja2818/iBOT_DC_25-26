/*
Author: Sanjana Vani V
Date: 01-03-2026
Board: ESP32
Description: Connects ESP to WiFi Network
*/

#include <WiFi.h>

const char* ssid = "<Wifi Name>";
const char* password = "<WiFi Password>";

void setup() {
  Serial.begin(115200);
  delay(1000); 

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("Wi-Fi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {

}