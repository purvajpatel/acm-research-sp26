#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
void setSendingImage(bool setter);

void CustomPrintln(const char* data);

void CustomPrintf(const char* beginningString, int theNumber);

void CustomPrintf(const char* beginningString);

bool CustomSerialReadForReadiness();

void CustomWrite(size_t number);

void CustomWrite(int8_t number);

void CustomWrite(uint8_t* buf);