const int sensorPin = A0;
const int red_led = 2;

void setup() {
  Serial.begin(9600);
  pinMode(red_led,OUTPUT);
}

void loop() {
  int rawValue = analogRead(sensorPin);   // 0–1023
  float voltage = rawValue * (5.0 / 1023.0); // convert to volts
  float temperatureC = (voltage - 0.5) * 100; // TMP36 formula

  Serial.print("Raw: ");
  Serial.print(rawValue);
  Serial.print(" | Voltage: ");
  Serial.print(voltage);
  Serial.print(" V | Temp: ");
  Serial.print(temperatureC);
  Serial.println(" °C");
  
  // in case temperature too hot
  if (temperatureC >= 80){
  	digitalWrite(red_led,HIGH);
  }else{
    digitalWrite(red_led,LOW);
  }

  delay(1000);
}
