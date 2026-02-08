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