/*
Author: Sanjana Vani V
Date: 05-02-2026
Board: Arduino Uno R3
Description: Calculates distance using Ultrasonic Sensor Module
*/


const int trig = 11;
const int echo = 12;
const float v_sound = 0.0343; //speed of sound in cm/microsecond
float time,distance;

void setup()
{
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
  Serial.begin(9600);
}

void loop()
{
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW); //pulse of length 10 microseconds

  time = pulseIn(echo, HIGH,25000); //time taken for pulse to hit obj and return
  distance = time*v_sound/2;

  Serial.print("Distance: ");
  Serial.println(distance);
  delay(100);
}

/*
Ultrasonic Sensor Connections:
VCC - 5V
GND - Ground
Trig - Digital Pin 11
Echo - Digital Pin 12
*/