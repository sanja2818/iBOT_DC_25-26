/*
Author: Sanjana Vani V
Date: 01-03-2026
Board: ESP32
Description: Blinks LED using Bluetooth
*/

#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;
const int led = 26;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_Sanjana"); 
  Serial.println("Device Started");
  pinMode(led, OUTPUT);
}

void loop() {
  if (SerialBT.available()) {
    Serial.print("available");
    char val = SerialBT.read();
    if (val == '1'){
      digitalWrite(led, HIGH);
    }
    if (val == '0'){
      digitalWrite(led, LOW);
    }
  }
  delay(20);
}

/*
LED Connections:
Anode - Pin 26
Cathode - Ground via 330 Ohm Resistor
*/