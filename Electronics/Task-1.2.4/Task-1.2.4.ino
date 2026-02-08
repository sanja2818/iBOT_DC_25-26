/*
Author: Sanjana Vani V
Date: 05-02-2026
Board: Arduino Uno R3
Description: Rotates a Servo motor smoothly sweeping 180 degrees
*/

#include <Servo.h>

Servo servo;
int position = 0;

void setup() {
  servo.attach(9);
}

void loop() {
  for (position = 0; position <= 180; position += 1) {
    servo.write(position);
    delay(30);
  }
  for (position = 180; position >= 0; position -= 1) {
    servo.write(position);
    delay(30);
  }
}

/*
Servo Connections:
VCC - 5V
GND - Ground
Signal - Digital Pin 9 (PWM)
*/