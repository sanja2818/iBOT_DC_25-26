/*
Author: Sanjana Vani V
Date: 04-02-2026
Board: Arduino Uno R3
Description: Tests reading of Flame Sensor using IR Sensor
*/

const int flame = A1; 
int value = 0;

void setup() {
  Serial.begin(9600);
  pinMode(flame, INPUT);
}

void loop() {
  value = analogRead(flame);
  
  Serial.print("Reading: ");
  Serial.println(value);
  
  delay(100);
}

/*
Flame Sensor Connections:
VCC - 5V
GND - Ground
O/P - Analog Pin A1

IR Sensor Connections:
VCC - 5V
GND - Ground
*/