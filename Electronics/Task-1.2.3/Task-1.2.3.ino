/*
Author: Sanjana Vani V
Date: 05-02-2026
Board: Arduino Uno R3
Description: Hello World on 128x64 OLED
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const int Width = 128;
const int Height = 64;

Adafruit_SSD1306 Display(Width, Height, &Wire, -1);

void setup() {
  Serial.begin(9600);

  if(!Display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
  }
  Display.setTextSize(1);
  Display.setTextColor(SSD1306_WHITE);
  Display.clearDisplay();
  Display.setCursor(0, 0);;
}

void loop() {
  Display.setCursor(0,0);
  Display.println("Hello World");
  Display.drawCircle(20, 40, 15, SSD1306_WHITE);
  Display.drawCircle(20, 40, 10, SSD1306_WHITE);
  Display.display();
  delay(1000);

}

/*
OLED Connections:
GND - Ground
VCC - 3.3V
SCL - Analog Pin A5
SDA - Analog Pin A4
*/