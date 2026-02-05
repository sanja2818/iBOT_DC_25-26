/*
Author: Sanjana Vani V
Date: 04-02-2026
Board: Arduino Uno R3
Description: LED toggled by a push button
*/

const int led = 4;
int state = -1;
int current = 0;
int prev = 0;
const int button = 5;

void setup() {
  pinMode(led, OUTPUT);
  pinMode(button, INPUT);
}

void loop() {
  current = digitalRead(button);
  state = (digitalRead(button)==HIGH && prev == LOW)? -state:state;
  if (state == 1) {
    digitalWrite(led, HIGH);
  }
  else {
    digitalWrite(led, LOW);
  }
  delay(100);
  prev = current;
}

/*
LED Connections:
Anode - 5V via 330 Ohm Resistor
Cathode - Ground
Push button connections:
Terminal 1 - 5V
Terminal 2 - Digital Pin 5, with pulldown resistor
*/