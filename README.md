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

## RECOMMENDED: YOLO11n (COCO) firmware - `firmware/examples/bird_yolo`
Uses ESP-DL's stock COCO-trained YOLO11n (320 px, INT8) and only reacts to COCO class 14 ("bird"). It was trained with
plenty of negatives, so it does not fire on people/ceilings. Board self-test: pigeon photo -> bird 0.88, bird photo -> bird 0.92,
person photo -> person 0.92 (not a bird). Cost: ~4.7 s per frame on the ESP32-S3. `BIRD DETECTED` needs 2 consecutive frames >= 0.40.
Needs `espressif/coco_detect` from the esp-dl repo (`models/coco_detect`) and the compile flag
`-Wno-error=non-c-typedef-for-linkage` (already in its CMakeLists) for GCC 16.
Build: copy to `esp-dl/examples/bird_yolo`, then `idf.py set-target esp32s3 && idf.py build flash monitor`.
Testing via a laptop screen is unreliable: the model reports the laptop instead of the small picture on it.

## Firmware behaviour (older custom ESPDet-Pico firmware, `bird_detect`)
Each frame prints `Candidate (unverified) score=... box=[...]` for every raw detection. `BIRD DETECTED` is printed only when
the best score is >= 0.90 on 3 consecutive frames (`CONFIRM_SCORE` / `CONFIRM_FRAMES` in `app_main.cpp`).
Debug JPEGs (base64 between `JPEG_BEGIN`/`JPEG_END`) are sent at frames 1 and 25.
`BIRD_SELFTEST=1` runs the model on embedded images at boot (generate with `tools/make_selftest_header.py`).

## Fixed bugs
- `dl::image::cvt_pix` converts ONE pixel; the frame was never converted, so the model saw uninitialised memory.
  Now a full-frame RGB565(BE) -> RGB888 loop is used.

## Known limitation of the custom ESPDet-Pico model (important)
The model was trained only on CUB bird photos, so it has never seen negatives. It scores empty scenes, ceilings and people
at 0.3-0.9, overlapping real birds (~0.92), so a confidence threshold cannot separate them. Float and INT8 models agree
(pigeon 0.915 vs 0.924), so quantization is not the cause. Fix = retrain with background/negative images (e.g. your own
empty-balcony frames + COCO images without birds), then re-quantize (`training/test_bird_int8.py`, `deploy/quantize.py` in esp-detection).

## TODO
Buzzer GPIO + cooldown, negative-image retraining, real-balcony false-positive/hour measurement.
