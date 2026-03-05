#include <RH_ASK.h>
#ifdef RH_HAVE_HARDWARE_SPI
#include <SPI.h> 
#endif
#include <string.h>

int position = 0;

RH_ASK driver;
int state = 1;
int a = 0;

const int trig = 10;
const int echo = 12;
const int buzzer = 6;
const int ldr = A0;
const int servo = 9;
int k=0;
int state2 = 1;
const float v_sound = 0.0343; 
float time,distance;
int currentmillis;
int previousmillis=0;
int max_num = 5 ;
float values[5] = {100,100,100,100,100};

void forward(){
  digitalWrite(servo, HIGH);
  delayMicroseconds(1900);
  digitalWrite(servo, LOW);
  delayMicroseconds(18100);
}

void reverse(){
  digitalWrite(servo, HIGH);
  delayMicroseconds(1100);
  digitalWrite(servo, LOW);
  delayMicroseconds(18900);
}

void setup()
{
#ifdef RH_HAVE_SERIAL
    Serial.begin(9600);	 
#endif
    if (!driver.init())
#ifdef RH_HAVE_SERIAL
         Serial.println("init failed");
#else
	;
#endif
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
  pinMode(ldr, INPUT);
  pinMode(buzzer, OUTPUT);
  pinMode(servo, OUTPUT);
  //digitalWrite(servo, LOW);
  if (k<10){ // 1 = forward and 0 = reverse
    forward();
    k++;
  }
  k=0;
}

void loop()
{
  currentmillis = millis();
  uint8_t buf[RH_ASK_MAX_MESSAGE_LEN];
  uint8_t buflen = sizeof(buf);
  if (driver.recv(buf, &buflen))
  {
    int i;
    buf[1] = '\0';
	  int j = strcmp("A", ((char*)buf));
    Serial.println((char*)buf);
    Serial.println(j);
    if (j==0){
      state = 0;
      a=0;
    }
    else {
      state = 1;
    }
  }
  if (state == 1){
    if (currentmillis-previousmillis > 100){
      if (k<10 && state2 == 1){ // 1 = forward and 0 = reverse
        reverse();
        k++;
        if(k==10){
          state2 = 0;
        }
      }
      if(k>=0 && state2==0){
        forward();
        k--;
        if(k==0){
          state2 = 1;
        }
      }
      previousmillis = currentmillis;

    }
    digitalWrite(trig, LOW);
    delayMicroseconds(2);
    digitalWrite(trig, HIGH);
    delayMicroseconds(10);
    digitalWrite(trig, LOW);
    time = pulseIn(echo, HIGH, 25000); 
    distance = time*v_sound/2;
    float avg = 0;
    for(int al=1;al<(max_num);al++){
      values[al-1]= values[al];
      avg += values[al];
    }
    values[max_num-1] = distance;
    avg += values[max_num-1];
    avg /= max_num;
    Serial.println(analogRead(ldr));
    Serial.println(avg);
    Serial.print(values[0]);
    Serial.print(",");
    Serial.print(values[1]);
    Serial.print(",");
    Serial.print(values[2]);
    Serial.print(",");
    Serial.print(values[3]);
    Serial.print(",");
    Serial.print(values[4]);
    Serial.print(",");
    if (analogRead(ldr)>100 || avg<10.0)
    {
      a=1;
    }
    if (a!=0){
      tone(buzzer, 5000);
    }
  }
  if (a==0){
    noTone(buzzer);
  }
}
