#include <Arduino.h>
#include "esp_camera.h"

// check these pins
#define PWDN_PIN    -1
#define RESET_PIN   -1
#define XCLK_PIN    15
#define SIOD_PIN    4
#define SIOC_PIN    5
#define VSYNC_PIN   6
#define HREF_PIN    7
#define PCLK_PIN    13

// these pinouts are correct
#define Y9_PIN      16
#define Y8_PIN      17
#define Y7_PIN      18
#define Y6_PIN      12
#define Y5_PIN      10
#define Y4_PIN      8
#define Y3_PIN      9
#define Y2_PIN      11





void setup() {
  Serial.begin(115200); 
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(10000);
  Serial.println("HELP IN GAIA");
}