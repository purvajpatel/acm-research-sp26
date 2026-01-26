#include <Arduino.h>

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
    Serial.println(data);
  }
}

void CustomPrintf(const char* beginningString, int theNumber)
{
  if(!sendingImage)
  {
    Serial.printf(beginningString, theNumber);
  }
}

void CustomPrintf(const char* beginningString)
{
  if(!sendingImage)
  {
    Serial.printf(beginningString);
  }
}

bool CustomSerialReadForReadiness()
{
  if(Serial.available() == 1 && sendingImage)
  {
    uint8_t susVar;
    Serial.readBytes(&susVar, 1);
    return true;
  }
  return false;
}

void CustomWrite(size_t number)
{
    if(sendingImage)
    {
        Serial.write((uint8_t) number);
        if(width == 0)
        {
        width = number;
        }
        else if (height == 0)
        {
        height = number;
        }
    }
}

void CustomWrite(uint8_t* buf)
{
    if(sendingImage)
    {
        Serial.write(buf, width * height);
    }
}