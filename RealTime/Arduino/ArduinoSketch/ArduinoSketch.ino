#include "src/GenericSerial/GenericSerial.h"

#define BAUDRATE 115200

GenericSerial gs = GenericSerial();

int INPUTS[] = {4,5}; // define pins of digital inputs
int lastinput [sizeof(INPUTS) / sizeof(int)] = {1}; //prior input states
int currinput [sizeof(INPUTS) / sizeof(int)] = {1}; //current input states
int input_num = sizeof(INPUTS) / sizeof(int);

void setup()
{
  gs.begin(BAUDRATE);
}

void loop()
{
  gs.process();
  for (int i = 0; i < input_num; i++) {
    currinput[i] = digitalRead(INPUTS[i]); 
    if ((currinput[i]==1) && (currinput[i] != lastinput[i])) { 
      //digitalWrite(13, LOW); //this can be used for testing output state with LED
      Serial.write(INPUTS[i]);
      Serial.write(currinput[i]);
      Serial.write(255);
    }
    if ((currinput[i]==0) && (currinput[i] != lastinput[i])) {
      //digitalWrite(13, HIGH); //this can be used for testing output state with LED
      Serial.write(INPUTS[i]);
      Serial.write(currinput[i]);
      Serial.write(255);
    }
    lastinput[i] = currinput[i];
  }
}
