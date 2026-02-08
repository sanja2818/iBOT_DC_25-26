#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const int Width = 128;
const int Height = 64;
const int left_button = 2;
const int right_button = 3;
const int up_button = 4;
const int down_button = 5;
int x = 0;
int y = 0;

Adafruit_SSD1306 Display(Width, Height, &Wire, -1);

void setup() {
  Serial.begin(9600);
  pinMode(left_button, INPUT_PULLUP);
  pinMode(right_button, INPUT_PULLUP);
  pinMode(up_button, INPUT_PULLUP);
  pinMode(down_button, INPUT_PULLUP);
  if(!Display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
  }
  Display.setTextSize(1);
  Display.setTextColor(SSD1306_WHITE);
  Display.clearDisplay();
  Display.setCursor(0, 0);
  Display.drawRect(0, 0, 8, 8, SSD1306_WHITE);
  Display.fillRect(0, 0, 8, 8, SSD1306_WHITE);
}

void left(int* x)
{
  *x=(*x-8+128)%128;
}

void right(int* x)
{
  *x=(*x+8)%128;
}

void up(int* y)
{
  *y=(*y-8+64)%64;
}

void down(int* y)
{
  *y=(*y+8)%64;
}

void loop() {
  Display.clearDisplay();
  if (digitalRead(left_button) == LOW)
  {
    left(&x);
  }
  if (digitalRead(right_button) == LOW)
  {
    right(&x);
  }
  if (digitalRead(up_button) == LOW)
  {
    up(&y);
  }
  if (digitalRead(down_button) == LOW)
  {
    down(&y);
  }
  Display.drawRect(x, y, 8, 8, SSD1306_WHITE);
  Display.fillRect(x, y, 8, 8, SSD1306_WHITE);
  Display.display();
  delay(700);
}