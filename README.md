# ESP32-S3 Bird Detector (ESPDet-Pico 160x288, INT8)

Camera (OV3660/OV2640) -> ESP32-S3 -> ESP-DL ESPDet-Pico (single class: bird) -> log / buzzer (TODO).

## Layout
- `firmware/` - ESP-IDF project + model component. Copy into a clone of https://github.com/espressif/esp-dl
  (tested with commit 4f4efd7): `firmware/examples/bird_detect` -> `esp-dl/examples/`, `firmware/models/bird_detect` -> `esp-dl/models/`.
- `trained_models/` - `best.pt`, `best.onnx`, quantized `espdet_pico_160_288_bird.espdl`, training curves.
- `training/` - training/quantization scripts, CUB-200 converter, 200 calibration images, cat pretrained weights.
- `docs/` - project notes, original write-ups. `samples/` - test images, YOLOv8n.

## Build and flash
```bash
. ~/esp/esp-idf/export.sh            # your ESP-IDF
cd esp-dl/examples/bird_detect
idf.py set-target esp32s3 && idf.py build
idf.py -p /dev/cu.usbmodemXXXX flash monitor
```

## Firmware behaviour
Each frame prints `Candidate (unverified) score=... box=[...]` for every raw detection. `BIRD DETECTED` is printed only when
the best score is >= 0.90 on 3 consecutive frames (`CONFIRM_SCORE` / `CONFIRM_FRAMES` in `app_main.cpp`).
Debug JPEGs (base64 between `JPEG_BEGIN`/`JPEG_END`) are sent at frames 1 and 25.
`BIRD_SELFTEST=1` runs the model on embedded images at boot (generate with `tools/make_selftest_header.py`).

## Fixed bugs
- `dl::image::cvt_pix` converts ONE pixel; the frame was never converted, so the model saw uninitialised memory.
  Now a full-frame RGB565(BE) -> RGB888 loop is used.

## Known limitation (important)
The model was trained only on CUB bird photos, so it has never seen negatives. It scores empty scenes, ceilings and people
at 0.3-0.9, overlapping real birds (~0.92), so a confidence threshold cannot separate them. Float and INT8 models agree
(pigeon 0.915 vs 0.924), so quantization is not the cause. Fix = retrain with background/negative images (e.g. your own
empty-balcony frames + COCO images without birds), then re-quantize (`training/test_bird_int8.py`, `deploy/quantize.py` in esp-detection).

## TODO
Buzzer GPIO + cooldown, negative-image retraining, real-balcony false-positive/hour measurement.
