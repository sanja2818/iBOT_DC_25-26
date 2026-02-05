/*
Author: Sanjana Vani V
Date: 04-02-2025
Board: Arduino Uno R3
Description: Blinks internal LED of Arduino
*/

void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop()
{
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000); 
}