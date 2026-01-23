#include <Arduino.h>
#include "esp_camera.h"

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5

// these pinouts are correct
#define Y9_PIN      16
#define Y8_PIN_NUM      17
#define Y7_PIN_NUM      18
#define Y6_PIN_NUM      12
#define Y5_PIN_NUM      10
#define Y4_PIN_NUM      8
#define Y3_PIN_NUM      9
#define Y2_PIN_NUM      11


#define VSYNC_PIN_NUM   6
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