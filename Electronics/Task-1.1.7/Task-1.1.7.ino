/*
Author: Sanjana Vani V
Date: 05-02-2026
Board: Arduino Uno R3
Description: Tests reading of DHT11 
*/


#include <DHT.h>
#define DHTTYPE DHT11

const int DHT_pin = 3;

DHT dht(DHT_pin, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
  delay(1000);
}

void loop() {
  float Humidity = dht.readHumidity();
  float Temperature = dht.readTemperature();

  if(isnan(Humidity) || isnan(Temperature))
  {
    Serial.println(F("Reading Failed"));
  }
  else
  {
    Serial.print("Humidity: ");
    Serial.print(Humidity);
    Serial.println("%");
    Serial.print("Temperature: ");
    Serial.print(Temperature);
    Serial.println("°C"); 
  }
  delay(1000);
}

/*
DH11 Connections:
VCC - 5V
GND - Ground
O/P - Digital Pin 3
*/

