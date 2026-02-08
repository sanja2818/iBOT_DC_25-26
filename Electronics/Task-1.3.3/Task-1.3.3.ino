/*
Author: Sanjana Vani V
Date: 07-02-2026
Board: Arduino Uno R3
Description: Visualises Audio Input using a 128x64 OLED
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const int Width = 128;
const int Height = 64;
const int sound_sensor = A1;
int reading;
int y = 0;
int y_prev = 0;

Adafruit_SSD1306 Display(Width, Height, &Wire, -1);

void setup() 
{
  Serial.begin(9600);
  if(!Display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
  }
  Display.setTextSize(1);
  Display.setTextColor(SSD1306_WHITE);
  Display.clearDisplay();
  Display.setCursor(0, 0);
}

void loop() {
  Display.clearDisplay();
  reading = analogRead(sound_sensor);
  Serial.println(reading);
  y = map(reading, 500, 550, 0, 64);
  if (y>y_prev)
  {
    for (int i=y_prev; i<=y; i++)
    {
      Display.drawRect(60, 64-i, 8, i, SSD1306_WHITE);
      Display.fillRect(60, 64-i, 8, i, SSD1306_WHITE);
      delay(20);
    }
  }
  if (y<y_prev)
  {
    for (int i=y_prev; i>=y; i--)
    {
      Display.drawRect(64, 64-i, 8, i, SSD1306_WHITE);
      Display.fillRect(64, 64-i, 8, i, SSD1306_WHITE);
      delay(20);
    }
  }
  y=y_prev;
  Display.display();
}

/*
OLED Connections:
GND - Ground
VCC - 3.3V
SCL - Analog Pin A5
SDA - Analog Pin A4

Sound Sensor Connections:
VCC - 5V 
GND - Ground
A0 - Digital Pin A1
*/