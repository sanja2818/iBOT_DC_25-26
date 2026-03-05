/*
Author: Sanjana Vani V
Date: 01-03-2026
Board: ESP32
Description: Blinks Built in LED
*/

const int LED = 2;

void setup()
{
  pinMode(LED, OUTPUT);
}

void loop()
{
  digitalWrite(LED, HIGH);
  delay(1000);
  digitalWrite(LED, LOW);
  delay(1000); 
}