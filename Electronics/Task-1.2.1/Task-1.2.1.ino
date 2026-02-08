/*
Author: Sanjana Vani V
Date: 06-02-2026
Board: Arduino Uno R3
Description: Tests piezo buzzer module
*/

const int buzzer = 9;

void setup()
{
  pinMode(buzzer, OUTPUT);
}

void loop()
{
  tone(buzzer, 1000); 
  delay(1000);
  noTone(buzzer);
  delay(1000);
}