/*
Author: Sanjana Vani V
Date: 05-02-2026
Board: Arduino Uno R3
Description: Tests reading of Sound Sensor
*/


const int sound_sensor = 4;
int reading;

void setup() {
  pinMode(sound_sensor, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000);
  reading = digitalRead(sound_sensor);
  Serial.println(reading);
  if (reading == HIGH)
  {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(2000);
  }
}

/*
Sound Sensor Connections:
VCC - 5V
GND - Ground
D0 - Digital Pin 4
*/