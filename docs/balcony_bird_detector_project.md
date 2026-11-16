# ESP32-S3 Balcony Bird Detection Project

## 1. Project goal

Build a standalone balcony bird detector using an ESP32-S3 and OV2640
camera. The ESP32-S3 should capture images, run a custom lightweight
ESPDet-Pico bird detector locally, and trigger an active buzzer when a
bird is confidently detected.

Final intended pipeline:

``` text
OV2640 camera
    ↓
ESP32-S3
    ↓
image capture / preprocessing
    ↓
ESPDet-Pico, 160×288, INT8
    ↓
bird confidence threshold
    ↓
2–3 consecutive detections
    ↓
GPIO
    ↓
buzzer driver / active buzzer
```

Inference is intended to be fully on-device. Wi-Fi is optional and is
not required for detection.

------------------------------------------------------------------------

## 2. Hardware

### ESP32-S3 board

The board is an ESP32-S3 WROOM/N16R8-style camera development board
with:

-   ESP32-S3
-   16 MB flash
-   8 MB PSRAM
-   OV2640 camera connector
-   microSD support
-   USB-C USB-UART
-   USB-C USB-OTG

Arduino configuration that worked:

-   Board: `ESP32S3 Dev Module`
-   Flash: `16 MB`
-   PSRAM: `OPI PSRAM`

### OV2640 pin mapping

  Signal       GPIO
  ---------- ------
  XCLK           15
  SIOD/SDA        4
  SIOC/SCL        5
  D0/Y2          11
  D1/Y3           9
  D2/Y4           8
  D3/Y5          10
  D4/Y6          12
  D5/Y7          18
  D6/Y8          17
  D7/Y9          16
  VSYNC           6
  HREF            7
  PCLK           13
  PWDN           -1
  RESET          -1

Other known pins:

-   Native USB: GPIO19/20
-   USB-UART: GPIO43/44
-   BOOT: GPIO0
-   Common-board status LED: GPIO2

The buzzer GPIO is not finalized yet. A transistor/driver or suitable
active-buzzer module is preferred over directly driving a high-current
bare buzzer from an ESP32 GPIO.

------------------------------------------------------------------------

## 3. Hardware validation completed

The OV2640 was successfully tested from Arduino.

Successful test output included:

``` text
PSRAM: OK
Camera initialization: OK
Frame capture: OK
Image size: 320x240
JPEG size: ~4392 bytes
CAMERA TEST PASSED!
```

This proves:

-   camera wiring/pin mapping works;
-   FPC connection works;
-   PSRAM works;
-   camera initialization works;
-   frame capture works.

A Wi-Fi camera server was also tested. It exposed `/capture` and polled
roughly every 500 ms. This was a validation tool, not the intended final
ML architecture.

------------------------------------------------------------------------

## 4. ML approach

YOLOv8n was first tested on the Mac using Ultralytics.

Model:

``` text
yolov8n.pt
```

Approximate size:

``` text
6.2 MB
```

Tests included:

``` text
test.jpg  → person ~0.92
test1.jpg → bird ~0.60
```

The project then moved to Espressif's lightweight ESPDet-Pico ecosystem
because the final detector needs to run on the ESP32-S3.

The detector intentionally uses one generic class:

``` text
class 0 = bird
```

It does not attempt to distinguish pigeon, sparrow, crow, etc.

------------------------------------------------------------------------

## 5. Dataset

Training dataset:

``` text
CUB-200-2011
```

All 200 bird species were collapsed into a single object-detection
class:

``` text
0 = bird
```

Converted YOLO dataset:

``` text
datasets/bird/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── bird.yaml
```

Dataset counts:

-   Training images: 5994
-   Validation images: 5794

`bird.yaml` used:

``` yaml
path: /esp/esp-detection/datasets/bird
train: images/train
val: images/val

names:
  0: bird
```

A verified label example was:

``` text
0 0.619000 0.515000 0.374000 0.690000
```

------------------------------------------------------------------------

## 6. Vast.ai training

Training was performed on Vast.ai using:

-   GPU: NVIDIA RTX 3060 Ti, 8 GB
-   CPU: Xeon W-2133
-   RAM: 128 GB
-   Approximate cost: \$0.185/hour

The initial Python 3.12 environment caused a NumPy compatibility/build
issue involving `pkgutil.ImpImporter`.

A Python 3.10 environment was then used:

``` text
Python 3.10.21
PyTorch 2.2.0+cu121
CUDA available: True
GPU: RTX 3060 Ti
```

------------------------------------------------------------------------

## 7. ESPDet-Pico training

Pretrained starting model:

``` text
pretrained/espdet_pico_160_288_cat.pt
```

Approximate size:

``` text
~976 KB
```

Training script:

``` python
from train import Train

results = Train(
    pretrained_path="pretrained/espdet_pico_160_288_cat.pt",
    dataset="datasets/bird/bird.yaml",
    imgsz=[160, 288],
    epochs=30,
    rect=True,
    device=0
)

print("Training complete!")
print("Best model:", results.save_dir / "weights/best.pt")
```

A rectangular image-size problem was found in the installed Ultralytics
trainer. It contained:

``` python
self.args.imgsz = check_imgsz(
    self.args.imgsz,
    stride=gs,
    floor=gs,
    max_dim=1
)
```

It was patched to permit the intended 2D size:

``` bash
sed -i 's/max_dim=1/max_dim=2/' /venv/espdet/lib/python3.10/site-packages/ultralytics/engine/trainer.py
```

The intended model input is:

``` text
160 × 288
```

------------------------------------------------------------------------

## 8. Training result

Training completed for 30 epochs.

Reported validation metrics:

``` text
Precision: 0.982
Recall: 0.982
mAP50: 0.993
mAP50-95: 0.821
```

Fused model:

``` text
~358,303 parameters
~1.3 GFLOPs
```

GPU validation inference was approximately:

``` text
0.4 ms/image
```

Best checkpoint:

``` text
/esp/esp-detection/runs/detect/train/weights/best.pt
```

Important limitation: CUB validation consists of bird images and is not
representative of the full balcony background. Real-world false
positives still need testing against empty balcony scenes, plants,
railings, people, sky, wires, shadows, etc.

------------------------------------------------------------------------

## 9. ONNX export

Export:

``` python
from deploy.export import Export

Export(
    "runs/detect/train/weights/best.pt",
    [160, 288]
)
```

Output:

``` text
runs/detect/train/weights/best.onnx
```

Approximate size:

``` text
1.5 MB
```

Input:

``` text
(1, 3, 160, 288)
```

The ONNX file was successfully produced.

------------------------------------------------------------------------

## 10. INT8 ESP32-S3 quantization

Calibration directory:

``` text
deploy/bird_calib
```

It contained 200 training images.

Quantization call:

``` python
from deploy.quantize import quant_espdet

quant_espdet(
    onnx_path="runs/detect/train/weights/best.onnx",
    target="esp32s3",
    num_of_bits=8,
    device="cpu",
    batchsz=32,
    imgsz=[160, 288],
    calib_dir="deploy/bird_calib",
    espdl_model_path="runs/detect/train/weights/bird_160_288.espdl",
)
```

Output:

``` text
runs/detect/train/weights/bird_160_288.espdl
```

Approximate size:

``` text
489 KB
```

The quantizer printed numerous skip/not-quantable messages, but the
final `.espdl` model was successfully created.

Final model name:

``` text
espdet_pico_160_288_bird.espdl
```

------------------------------------------------------------------------

## 11. ESP-Detection vs ESP-DL

These are different parts of the stack.

### ESP-Detection

Used for:

-   training;
-   model export;
-   model quantization;
-   deployment preparation.

Local training project:

``` text
pigeon_model/esp/esp-detection/
```

### ESP-DL

Used for:

-   on-device neural-network inference;
-   ESP32 deployment;
-   ESP-IDF integration;
-   loading/running `.espdl` models.

Architecture:

``` text
ESP-Detection
    ↓
espdet_pico_160_288_bird.espdl
    ↓
ESP-DL
    ↓
ESP32-S3
```

------------------------------------------------------------------------

## 12. Current local directory layout

Project root:

``` text
/Users/piyushpilania/development/pigeon_model/
```

Important directories:

``` text
pigeon_model/
├── esp-idf/
├── esp-dl/
├── esp-dl-custom/
├── esp/
│   └── esp-detection/
├── runs/
├── test.jpg
├── test1.jpg
├── yolo.py
├── yolov8n.pt
└── bird_detect_esp32s3.tar.gz
```

The real ESP-DL repository is:

``` text
pigeon_model/esp-dl/
```

It contains:

``` text
esp-dl/
├── esp-dl/          ← actual ESP-DL component
├── examples/
├── models/
├── docs/
├── tools/
└── ...
```

The backup generated custom files are:

``` text
pigeon_model/esp-dl-custom/
├── examples/bird_detect/
└── models/bird_detect/
```

The custom files have been copied into the real repository.

------------------------------------------------------------------------

## 13. Custom deployment files

Custom model component:

``` text
esp-dl/models/bird_detect/
├── CMakeLists.txt
├── LICENSE
├── idf_component.yml
├── Kconfig
├── espdet_detect.hpp
├── espdet_detect.cpp
└── models/
    └── s3/
        └── espdet_pico_160_288_bird.espdl
```

Custom example:

``` text
esp-dl/examples/bird_detect/
├── CMakeLists.txt
├── main/
│   ├── CMakeLists.txt
│   ├── idf_component.yml
│   └── app_main.cpp
├── partitions.csv
├── partitions2.csv
└── sdkconfig.defaults*
```

The model was verified locally at:

``` text
esp-dl/models/bird_detect/models/s3/espdet_pico_160_288_bird.espdl
```

------------------------------------------------------------------------

## 14. ESP-IDF installation

ESP-IDF was installed at:

``` text
/Users/piyushpilania/development/pigeon_model/esp-idf
```

The initial failure was caused by a `uv`-managed Python:

``` text
/Users/piyushpilania/.local/bin/python3.11
```

Running `ensurepip` there failed with PEP 668's:

``` text
externally-managed-environment
```

The solution was to install Homebrew Python:

``` bash
brew install python@3.11
```

which installed:

``` text
/opt/homebrew/bin/python3.11
```

ESP-IDF installation then completed successfully.

Activation:

``` bash
cd /Users/piyushpilania/development/pigeon_model/esp-idf
. ./export.sh
```

The environment reports Python dependencies as OK.

`idf.py --version` currently reports:

``` text
ESP-IDF v6.1-dev-8199-g4d59230ddff
```

The ESP32-S3 compiler/toolchain is working.

------------------------------------------------------------------------

## 15. Current build project

Project:

``` text
/Users/piyushpilania/development/pigeon_model/esp-dl/examples/bird_detect
```

Target:

``` text
esp32s3
```

The command:

``` bash
idf.py set-target esp32s3
```

successfully reaches CMake and dependency resolution.

------------------------------------------------------------------------

## 16. Current blocker

The current build fails because the `override_path` in:

``` text
esp-dl/models/bird_detect/idf_component.yml
```

was pointing to:

``` text
../../esp-dl
```

which resolves to:

``` text
/Users/piyushpilania/development/pigeon_model/esp-dl
```

That directory is the repository root, not the ESP-DL component.

The real component is:

``` text
/Users/piyushpilania/development/pigeon_model/esp-dl/esp-dl
```

Therefore the intended final path from:

``` text
esp-dl/models/bird_detect/
```

is:

``` text
../../esp-dl/esp-dl
```

The manifest should therefore be:

``` yaml
dependencies:
  espressif/esp-dl:
    version: "^3.1.3"
    override_path: "../../esp-dl/esp-dl"
```

This is the immediate item to verify/fix.

Afterward:

``` bash
cd /Users/piyushpilania/development/pigeon_model/esp-dl/examples/bird_detect
rm -rf build
idf.py set-target esp32s3
```

If configuration succeeds:

``` bash
idf.py build
```

------------------------------------------------------------------------

## 17. Completed milestones

-   [x] ESP32-S3 identified.
-   [x] OV2640 connected.
-   [x] Camera pin mapping established.
-   [x] PSRAM verified.
-   [x] Camera initialization verified.
-   [x] 320×240 capture verified.
-   [x] JPEG capture verified.
-   [x] Wi-Fi camera test completed.
-   [x] YOLOv8n tested on Mac.
-   [x] Bird detection confirmed on a test image.
-   [x] Generic `bird` class selected.
-   [x] CUB-200-2011 converted to YOLO format.
-   [x] All 200 species collapsed to one `bird` class.
-   [x] Training/validation dataset prepared.
-   [x] RTX 3060 Ti Vast.ai environment prepared.
-   [x] Python 3.10/CUDA environment fixed.
-   [x] ESPDet-Pico pretrained model obtained.
-   [x] Rectangular 160×288 training issue fixed.
-   [x] 30-epoch custom bird model trained.
-   [x] Validation metrics obtained.
-   [x] ONNX export completed.
-   [x] INT8 quantization completed.
-   [x] 489 KB `.espdl` model generated.
-   [x] Real ESP-DL repository cloned.
-   [x] Custom bird model/example copied into the real repository.
-   [x] ESP-IDF installed.
-   [x] ESP-IDF activated.
-   [x] ESP32-S3 target selected.
-   [x] Initial CMake configuration reached dependency resolution.

------------------------------------------------------------------------

## 18. Remaining work

### Immediate

1.  Fix/verify `override_path`.
2.  Get the complete ESP-DL bird example to compile.
3.  Resolve any ESP-DL / ESP-IDF API compatibility issues.
4.  Flash the first successful firmware.

### Camera integration

5.  Integrate the known-good OV2640 configuration into the ESP-IDF
    application.
6.  Capture frames in the format expected by the ESPDet-Pico
    preprocessing code.
7.  Verify preprocessing and image dimensions.

### ML inference

8.  Load `espdet_pico_160_288_bird.espdl`.
9.  Run inference on the ESP32-S3.
10. Print detection class, confidence, and bounding box.
11. Measure actual ESP32-S3 inference latency and memory use.

### Buzzer

12. Select a safe GPIO.
13. Use an appropriate active buzzer module/transistor driver.
14. Implement confidence threshold.
15. Require 2--3 consecutive detections.
16. Add buzzer duration.
17. Add cooldown.

### Real-world validation

18. Collect balcony images/video.
19. Test empty balcony false positives.
20. Test birds at different distances and lighting.
21. Test partial/occluded birds.
22. Tune confidence threshold.
23. Measure false positives per hour.
24. Measure missed detections.

------------------------------------------------------------------------

## 19. Recommended final trigger algorithm

Initial implementation:

``` text
capture frame
    ↓
run detector
    ↓
bird confidence >= threshold?
    ├── no → reset/decay counter
    └── yes
          ↓
     consecutive++
          ↓
     consecutive >= 2 or 3?
          ├── no → continue
          └── yes
                ↓
          buzzer ON
                ↓
          1–2 seconds
                ↓
          cooldown 5–10 seconds
                ↓
          resume detection
```

Starting values can be tuned experimentally:

``` text
confidence threshold: 0.50
consecutive detections: 2–3
buzzer duration: 1–2 seconds
cooldown: 5–10 seconds
```

These are initial engineering values, not final validated settings.

------------------------------------------------------------------------

## 20. Important validation caveat

The reported training metrics are excellent:

``` text
Precision 0.982
Recall    0.982
mAP50     0.993
mAP50-95  0.821
```

but these numbers should not be interpreted as balcony performance.

The validation set is dominated by the same type of bird imagery used
for training and does not adequately measure false detections caused by
a real balcony environment.

The most useful final evaluation should therefore measure:

``` text
false positives/hour
true detections
missed birds
detection latency
buzzer reliability
```

------------------------------------------------------------------------

## 21. Final architecture

``` text
                     TRAINING

CUB-200-2011
      ↓
YOLO conversion
      ↓
ESP-Detection
      ↓
ESPDet-Pico training
      ↓
best.pt
      ↓
ONNX
      ↓
INT8 quantization
      ↓
espdet_pico_160_288_bird.espdl


                    DEPLOYMENT

OV2640
  ↓
ESP32-S3
  ↓
camera frame
  ↓
preprocessing
  ↓
ESP-DL
  ↓
ESPDet-Pico
  ↓
bird detection
  ↓
confidence + temporal filtering
  ↓
GPIO
  ↓
buzzer
```

------------------------------------------------------------------------

## 22. Current project state

The project has successfully completed the difficult ML preparation
stage:

``` text
Camera hardware          COMPLETE
Dataset                  COMPLETE
Training                 COMPLETE
Export                   COMPLETE
INT8 quantization        COMPLETE
Model transfer           COMPLETE
ESP-IDF installation     COMPLETE
ESP-DL repository        COMPLETE
Embedded build           CURRENT BLOCKER
Camera + inference       PENDING
Buzzer                   PENDING
Balcony validation       PENDING
```

The project is now at the **embedded deployment stage**. The next
concrete objective is to resolve the ESP-DL component path, obtain a
successful `idf.py build`, and then integrate the already-validated
OV2640 camera with the custom bird model.
