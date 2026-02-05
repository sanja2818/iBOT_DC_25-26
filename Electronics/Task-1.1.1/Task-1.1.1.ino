/*
Author: Sanjana Vani V
Date: 04-02-2026
Board: Arduino Uno R3
Description: Tests reading of IR Sensor
*/

int ir = A1;
int reading = 0;

void setup()
{
  pinMode(ir, INPUT);
  Serial.begin(9600);
}

void loop()
{
  reading = analogRead(ir);
  Serial.println(reading);
  delay(500);
}

/*
IR Sensor Connections:
VCC - 5V
GND - Ground
O/P - Analog pin A1
*/