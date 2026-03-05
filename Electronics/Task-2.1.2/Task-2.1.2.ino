/*
Author: Sanjana Vani V
Date: 01-03-2026
Board: ESP32
Description: Turns an LED On and Off using a Webserver
*/

#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "<Wifi Name>";
const char* password = "<WiFi Password>";

WebServer server(80);

String led_output = "off";
const int led = 26;

void handleRoot(){
  String html = "<html><body>";
  html += "<h1>ESP32 LED Control</h1>";
  html += "<p><a href=\"/on\"><button style='padding:20px; background:green; color:white;'>ON</button></a></p>";
  html += "<p><a href=\"/off\"><button style='padding:20px; background:red; color:white;'>OFF</button></a></p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleLEDON() {
  led_output = "on";
  digitalWrite(led, HIGH);
  handleRoot();
}

void handleLEDOFF() {
  led_output = "off";
  digitalWrite(led, LOW);
  handleRoot();
}

void setup(){
  Serial.begin(115200);
  pinMode(led, OUTPUT);
  digitalWrite(led, LOW);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/on", handleLEDON);
  server.on("/off", handleLEDOFF);

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}

/*
LED Connections:
Anode - Pin 26
Cathode - Ground via 330 Ohm Resistor
*/