/*
Author: Sanjana Vani V
Date: 04-02-2026
Board: Arduino Uno R3
Description: Tests reading of PIR Sensor
*/

const int pir = 2;

void setup() {
  pinMode(pir, INPUT);
  Serial.begin(9600);
  delay(5000);
}

void loop() {
  int motion = digitalRead(pir);

  if (motion == HIGH) {
    Serial.println("motion detected");
  } else {
    Serial.println("no motion detected");
  }
  delay(500);
}

/*
PIR Connections:
VCC - 5V
GND - Ground
O/P - Digital Pin 2
*/