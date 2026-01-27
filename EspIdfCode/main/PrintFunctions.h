#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
void setSendingImage(bool setter);

void CustomPrint(const char* logType, const char* data, ...);

bool CustomSerialReadForReadiness();

void CustomWrite(size_t number);

void CustomWrite(int8_t number);

void CustomWrite(uint8_t* buf);