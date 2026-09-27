// Human activity recognition on ESP32 + MPU6050.
//
// Pipeline: 50 Hz periodic sampling over I2C -> ring buffer -> 128x6 window
// every 64 samples (50% hop) -> per-channel standardization -> int8
// quantization -> on-device int8 CNN inference -> Serial + MQTT telemetry.
//
// Serial commands: '1'..'6' replay a recorded test window of class 1..6
// (walking, upstairs, downstairs, sitting, standing, laying) through the
// same pipeline, for deterministic demos without moving the sensor.

#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>

extern "C" {
#include "infer.h"
}
#include "model.h"
#include "demo.h"

#define MPU_ADDR 0x68
#define PERIOD_US 20000UL // 50 Hz
#define HOP (WIN / 2)

#define WIFI_SSID "Wokwi-GUEST" // Wokwi simulated AP; change for real hardware
#define WIFI_PASS ""
#define MQTT_HOST "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_TOPIC "hy/har"

static float buf[WIN][IN_C];
static int head, filled, since;
static uint32_t next;
static int replay = -1, rt;

static WiFiClient wc;
static PubSubClient mqtt(wc);
static uint32_t mqtt_try;

static void reg(uint8_t r, uint8_t v) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(r);
  Wire.write(v);
  Wire.endTransmission();
}

static void mpuInit() {
  reg(0x6B, 0x00); // PWR_MGMT_1: wake, internal clock
  reg(0x1A, 0x03); // CONFIG: DLPF ~44 Hz accel / ~42 Hz gyro (anti-alias for 50 Hz)
  reg(0x1B, 0x08); // GYRO_CONFIG: +/-500 dps -> 65.5 LSB/(deg/s)
  reg(0x1C, 0x08); // ACCEL_CONFIG: +/-4 g -> 8192 LSB/g
}

static void mpuRead(float *s) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // ACCEL_XOUT_H, burst 14 bytes: acc, temp, gyro
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14);
  int16_t raw[7];
  for (int i = 0; i < 7 && Wire.available() >= 2; i++)
    raw[i] = (int16_t)((Wire.read() << 8) | Wire.read());
  for (int i = 0; i < 3; i++)
    s[i] = raw[i] / 8192.0f;                                // g
  for (int i = 0; i < 3; i++)
    s[3 + i] = raw[4 + i] / 65.5f * (float)(M_PI / 180.0);  // rad/s
}

static void runInfer() {
  int8_t x[IN_C * WIN];
  for (int c = 0; c < IN_C; c++)
    for (int t = 0; t < WIN; t++) {
      float z = (buf[(head + t) % WIN][c] - MU[c]) / SD[c] / S0;
      int q = (int)roundf(z);
      x[c * WIN + t] = (int8_t)(q > 127 ? 127 : q < -128 ? -128 : q);
    }
  uint32_t t0 = micros();
  int8_t lg[NCLASS];
  int cls = infer(x, lg);
  uint32_t us = micros() - t0;

  float m = lg[cls] * S3, sum = 0;
  for (int o = 0; o < NCLASS; o++)
    sum += expf(lg[o] * S3 - m);
  float conf = 1.0f / sum;

  Serial.printf("%-11s conf=%.2f infer=%lu us\n", LABELS[cls], conf,
                (unsigned long)us);
  if (mqtt.connected()) {
    char js[96];
    snprintf(js, sizeof(js), "{\"a\":\"%s\",\"c\":%.2f,\"us\":%lu}",
             LABELS[cls], conf, (unsigned long)us);
    mqtt.publish(MQTT_TOPIC, js);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  mpuInit();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 80 && WiFi.status() != WL_CONNECTED; i++)
    delay(100);
  Serial.printf("READY wifi=%s cmds='1'..'6' replay\n",
                WiFi.status() == WL_CONNECTED ? "up" : "off");
  next = micros();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqtt.connected() && (int32_t)(millis() - mqtt_try) >= 0) {
      mqtt_try = millis() + 5000;
      mqtt.connect("hy-har-esp32");
    }
    mqtt.loop();
  }
  if (Serial.available()) {
    char c = (char)Serial.read();
    if (c >= '1' && c <= '6')
      replay = c - '1', rt = 0;
  }
  if ((int32_t)(micros() - next) < 0)
    return;
  next += PERIOD_US; // fixed-epoch scheduling: no cumulative drift

  float s[IN_C];
  if (replay >= 0) {
    for (int c = 0; c < IN_C; c++)
      s[c] = DEMO[replay][rt][c] / 1000.0f;
    if (++rt >= WIN)
      replay = -1;
  } else {
    mpuRead(s);
  }
  memcpy(buf[head], s, sizeof(s));
  head = (head + 1) % WIN;
  if (++filled > WIN)
    filled = WIN;
  if (++since >= HOP) {
    if (filled >= WIN)
      runInfer();
    since = 0;
  }
}
