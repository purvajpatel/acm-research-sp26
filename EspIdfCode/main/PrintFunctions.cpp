#include "PrintFunctions.h"
#include <stdio.h>
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_system.h"
#include "driver/uart.h"

bool sendingImage;
uint8_t width = 0;
uint8_t height = 0;

void setSendingImage(bool setter)
{
    sendingImage = setter;
}

void CustomPrintln(const char* data)
{
  if(!sendingImage)
  {
    printf("%s\n", data);
  }
}

void CustomPrintf(const char* beginningString, int theNumber)
{
  if(!sendingImage)
  {
    printf(beginningString, theNumber);
  }
}

void CustomPrintf(const char* beginningString)
{
  if(!sendingImage)
  {
    printf(beginningString);
  }
}

bool CustomSerialReadForReadiness()
{
  size_t bufferLength = 0;
  uart_get_buffered_data_len(UART_NUM_0, &bufferLength);
  
  if(bufferLength == 1 && sendingImage)
  {
    uint8_t susVar;
    // check if we're able to read correctly
    int readResult = uart_read_bytes(UART_NUM_0, &susVar, 1, 100 / portTICK_PERIOD_MS);
    return readResult == 1;
  }
  return false;
}

void CustomWrite(size_t number)
{
    uint8_t convertedToOneByteNum = (uint8_t) number;
    uart_write_bytes(UART_NUM_0, (const char *)&convertedToOneByteNum, 1);
  
    // Serial.write((uint8_t) number);
    if(width == 0)
    {
      width = number;
    }
    else if (height == 0)
    {
      height = number;
    }
}

void CustomWrite(uint8_t* buf)
{
    uart_write_bytes(UART_NUM_0, (const char*) buf, width * height);
}

void CustomWrite(int8_t number)
{
    uart_write_bytes(UART_NUM_0, (const char *)&number, 1);
}