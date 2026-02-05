/*
Author: Sanjana Vani V
Date: 02-12-2025
Board: Arduino Uno R3
Description: Breathing LED with breath cycle 1s
*/

const int LED = 3;

void setup()
{
  pinMode(LED, OUTPUT);
}

void loop()
{
  //increasing brightness
  for (int i=0; i<256; i++) {
    analogWrite(LED, i);
    delayMicroseconds(1000000/256);
  }
  //decreasing brightness
  for (int i=0; i<256; i++) {
    analogWrite(LED, 255-i);
    delayMicroseconds(1000000/256);
  }
}

/*
LED Connections:
Anode - 330Ω Resistor from Digital Pin 3 (PWM)
Cathode - Ground
*/