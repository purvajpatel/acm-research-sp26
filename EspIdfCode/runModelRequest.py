import time
import serial
from PIL import Image

ser = serial.Serial('COM36', 115200)  # change COM port
imgName = "frame"

width = 0
height = 0

ser.write(bytes([1]))

width = ser.read(1)
width = int.from_bytes(width, "little")

height = ser.read(1)
height = int.from_bytes(height, "little")

imageBytes = ser.read(height * width)

class0Logit = ser.read(1)
class0Logit = int.from_bytes(class0Logit, "little")

class1Logit = ser.read(1)
class1Logit = int.from_bytes(class1Logit, "little")


# l for grayscale, then pass in the resolution, and then the bytes
img = Image.frombytes('L', (width, height), imageBytes)  # 'L' = grayscale
# now save as image
img.save(f"{imgName}.png")

print(f"Data saved! Class 0 logit was {class0Logit}, class 1 logit was {class1Logit}")