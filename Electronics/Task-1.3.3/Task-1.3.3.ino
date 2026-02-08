#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const int Width = 128;
const int Height = 64;
const int sound_sensor = A0;
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
  y = map(reading, 0, 1023, 0, 64);
  if (y>y_prev)
  {
    for (int i=y_prev; i<=y; i++)
    {
      Display.drawRect(64, 0, 8, i, SSD1306_WHITE);
      Display.fillRect(64, 0, 8, i, SSD1306_WHITE);
      delay(100);
    }
  }
  if (y<y_prev)
  {
    for (int i=y_prev; i>=y; i--)
    {
      Display.drawRect(64, 0, 8, i, SSD1306_WHITE);
      Display.fillRect(64, 0, 8, i, SSD1306_WHITE);
      delay(100);
    }
  }
  y=y_prev;
  Display.display();
}