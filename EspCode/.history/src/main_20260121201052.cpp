#include <Arduino.h>
#include "esp_camera.h"

void setup() {
  Serial.begin(115200); 
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(10000);
  Serial.println("HELP IN GAIA");
}