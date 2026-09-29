# HY-DevAIoT - Human Activity Recognition on ESP32 (Wokwi)

AIoT prototype: MPU6050 sensing pipeline at 50 Hz feeding an int8-quantized
1D-CNN running fully on-device on an ESP32, with MQTT telemetry.

## Layout

```
firmware/   Wokwi/Arduino project
  sketch.ino    sampling pipeline, windowing, MQTT, serial replay
  infer.c/.h    integer-only int8 CNN engine (host-compilable, bit-exact)
  model.h       generated: int8 weights, scales, calibration, labels
  demo.h        generated: one real test window per class (replay)
  diagram.json  ESP32 + MPU6050 wiring
  libraries.txt PubSubClient
  scenario.yaml Wokwi automation: rest -> bounce -> rotation demo
ml/
  train.py      download UCI HAR, train FP32, int8 PTQ, export headers
  data/         UCI HAR dataset (auto-downloaded)
  requirements.txt
test/
  main.c        host test: asserts bit-exact logits + benchmarks
  vectors.h     generated test vectors
```

## Run the demo (Wokwi)

1. Create a project on wokwi.com, upload the `firmware/` files.
2. Press play. Serial shows `READY` then a classification every 1.28 s.
3. Send `1`..`6` in the serial monitor to replay a real recorded window
   (walking, upstairs, downstairs, sitting, standing, laying).
4. Or run `scenario.yaml` to drive the simulated MPU6050 (rest, 2 Hz bounce,
   rotation burst) through the real sensor path.
5. MQTT: subscribe to `hy/har` on broker.hivemq.com.

Real hardware: same sketch, SDA→GPIO21, SCL→GPIO22; set `WIFI_SSID`/`WIFI_PASS`.

## Reproduce the model

```sh
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch cpu + numpy
python train.py                   # trains, quantizes, exports headers, prints metrics
```

## Verify the inference engine

```sh
gcc -O2 -Ifirmware -Itest test/main.c firmware/infer.c -o /tmp/tinfer && /tmp/tinfer
# PASS (4 vectors) + host latency
```

Or compile the firmware locally:

```sh
arduino-cli core install esp32:esp32 && arduino-cli lib install PubSubClient
arduino-cli compile -b esp32:esp32:esp32 firmware
```
