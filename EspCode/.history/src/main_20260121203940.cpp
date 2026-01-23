#include <Arduino.h>
#include "esp_camera.h"

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5

// these pinouts are correct
#define Y9_PIN      16
#define Y8_PIN      17
#define Y7_PIN      18
#define Y6_PIN      12
#define Y5_PIN      10
#define Y4_PIN      8
#define Y3_PIN      9
#define Y2_PIN      11


#define VSYNC_PIN   6
#define HREF_GPIO_NUM    7
#define PCLK_GPIO_NUM    13



void setup() {
  Serial.begin(115200); 
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(10000);
  Serial.println("HELP IN GAIA");
}