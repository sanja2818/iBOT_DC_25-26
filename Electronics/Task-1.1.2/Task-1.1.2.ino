/*
Author: Sanjana Vani V
Date: 04-02-2026
Board: Arduino Uno R3
Description: Tests reading of LDR
*/

int ldr = A1;
int reading = 0;

void setup()
{
  pinMode(ldr, INPUT);
  Serial.begin(9600);
}

void loop()
{
  reading = analogRead(ldr);
  Serial.println(reading);
  delay(500);
}

/*
LDR Connections:
VCC - 5V
GND - Ground
O/P - Analog Pin A1
*/