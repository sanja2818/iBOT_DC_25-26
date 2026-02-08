/*
Author: Sanjana Vani V
Date: 07-02-2026
Board: Arduino Uno R3
Description: Tripwire Security System using Laser and LDR
*/

#include <LiquidCrystal.h>

const int rs = 10, en = 7, d4 = 6, d5 = 5, d6 = 4, d7 = 3;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);
const int laser = 2;
const int buzzer = 11;
const int ldr = A0;

void setup()
{
  pinMode(laser, OUTPUT);
  pinMode(buzzer, OUTPUT);
  pinMode(ldr, INPUT);
  digitalWrite(laser, HIGH);
  lcd.begin(16, 2);
}

void loop()
{
  lcd.setCursor(0,0);
  if (analogRead(ldr)>100)
  {
    tone(buzzer, 1000);
    lcd.print("Interference");
    lcd.setCursor(0,1);
    lcd.print("Detected");
  }
  else
  {
    noTone(buzzer);
    lcd.print("No Interference        ");
  }
  delay(100);
  lcd.clear();
}

/*
LCD Connections:
VCC - 5V
GND - Ground
V0 - Potentiometer
RS - Digital Pin 10
RW - Ground
EN - Digital Pin 7
DB4 - Digital Pin 6
DB5 - Digital Pin 5
DB6 - Digital Pin 4
DB7 - Digital Pin 3
A - 5V
K - Ground via 330 Ohm Resistor

Potentiometer Connections:
T1 - 5V
T2 - Ground
O/P - V0 of LCD

LDR Connections:
VCC - 5V
GND - Ground
O/P - Analog Pin A0

Laser Connections:
VCC - 5V
GND - Ground
O/P - Digital Pin 2
*/