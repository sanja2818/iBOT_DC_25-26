/*
Author: Sanjana Vani V
Date: 01-03-2026
Board: ESP32
Description: Breathing LED using LEDC
*/

const int led = 15;
unsigned long lastmillis = 0;
int brightness = 0;
int i = 1;

void setup() {
  ledcAttach(led, 1000, 8);
}

void loop() {
  unsigned long currentmillis = millis();

  if (currentmillis - lastmillis >= 10) {
    lastmillis = currentmillis;

    ledcWrite(led, brightness);
    brightness = brightness+i;

    if (brightness <= 0 || brightness >= 255) {
      i*=-1;
    }
  }
}

/*
LED Connections:
Anode - Pin 15
Cathode - Ground via 330 Ohm Resistor
*/