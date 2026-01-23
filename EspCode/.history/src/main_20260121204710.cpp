#include <Arduino.h>
#include "esp_camera.h"

// the pinout doesn't have these
#define PWDN_PIN    -1
#define RESET_PIN   -1

// these are correct
#define SIOD_PIN    4
#define SIOC_PIN    5
#define VSYNC_PIN   6
#define HREF_PIN    7
#define XCLK_PIN    15
#define PCLK_PIN    13

// these pinouts are correct
#define Y7_PIN      16
#define Y6_PIN      17
#define Y5_PIN      18
#define Y4_PIN      12
#define Y3_PIN      10
#define Y2_PIN      8
#define Y1_PIN      9
#define Y0_PIN      11

#define Clock_Freq = 20000000





void setup() {
  Serial.begin(115200); 

  camera_config_t camConfig = {
    .pin_d0 = Y0_PIN,
    .pin_d1 = Y1_PIN,
    .pin_d2 = Y2_PIN,
    .pin_d3 = Y3_PIN,
    .pin_d4 = Y4_PIN,
    .pin_d5 = Y5_PIN,
    .pin_d6 = Y6_PIN,
    .pin_d7 = Y7_PIN,

    .pin_xclk = XCLK_PIN,
    .pin_pclk = PCLK_PIN,
    .pin_vsync = VSYNC_PIN,
    .pin_href = HREF_PIN,

    .pin_sccb_sda = SIOD_PIN,
    .pin_sccb_scl = SIOC_PIN,

    .pin_pwdn = -1,
    .pin_reset = -1,

    .xclk_freq_hz = Clock_Freq
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(10000);
  Serial.println("HELP IN GAIA");
}