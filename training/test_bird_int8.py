#!/usr/bin/env python3

import os
import sys
import traceback
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/Users/piyushpilania/development/pigeon_model")
ESP_DETECTION = PROJECT_ROOT / "esp" / "esp-detection"

BEST_ONNX = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.onnx"
BEST_PT = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

CALIB_DIR = ESP_DETECTION / "datasets" / "bird" / "calib"
DATA_YAML = ESP_DETECTION / "datasets" / "bird" / "bird.yaml"

# IMPORTANT:
# We create a NEW temporary model.
# We do NOT overwrite the model currently flashed/used by ESP32.
OUTPUT_ESPDL = Path("/tmp/pigeon_test_160_288_s3.espdl")

# Exact model input used during training/export.
IMG_SIZE = [160, 288]

# Exact target hardware.
TARGET = "esp32s3"

NUM_BITS = 8
DEVICE = "cpu"
BATCH_SIZE = 32
CALIB_STEPS = 32


# ============================================================
# BASIC CHECKS
# ============================================================

def check_file(path, description):
    print(f"Checking {description}:")
    print(f"  {path}")

    if not path.exists():
        print(f"\nERROR: {description} does not exist:")
        print(path)
        sys.exit(1)

    print("  OK\n")


def check_calibration():
    if not CALIB_DIR.exists():
        print("ERROR: Calibration directory does not exist:")
        print(CALIB_DIR)
        sys.exit(1)

    images = [
        p for p in CALIB_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    print(f"Calibration images: {len(images)}")

    if len(images) == 0:
        print("ERROR: No calibration images found.")
        sys.exit(1)

    print()


# ============================================================
# IMPORT ESP-PPQ
# ============================================================

def import_ppq():
    print("Checking ESP-PPQ...")

    try:
        import esp_ppq
        print(f"ESP-PPQ imported successfully:")
        print(f"  {esp_ppq.__file__}\n")
        return esp_ppq

    except ModuleNotFoundError:
        print()
        print("=" * 70)
        print("ERROR: esp_ppq is not installed in this Python environment.")
        print("=" * 70)
        print()
        print("Current Python:")
        print(sys.executable)
        print()
        print("This script cannot perform INT8 validation until the ESP-PPQ")
        print("environment used by the ESP-DL quantization pipeline is available.")
        print()
        print("DO NOT install a random esp-ppq package yet.")
        print("The existing .espdl was already generated successfully, so")
        print("we need to use the compatible ESP-PPQ installation.")
        print()
        print("Run:")
        print()
        print("  find ~/.espressif -type d -name 'esp_ppq' -print")
        print()
        print("and:")
        print()
        print("  find /Users/piyushpilania/development/pigeon_model \\")
        print("    -type d -name 'esp_ppq' -print")
        print()
        sys.exit(2)


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_model():

    print("=" * 70)
    print("STEP 1: IMPORTING QUANTIZATION COMPONENTS")
    print("=" * 70)

    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx

    from torch.utils.data import DataLoader
    import torch

    from torchvision import transforms
    from PIL import Image

    from onnxsim import simplify
    import onnx

    print("Imports OK\n")

    # --------------------------------------------------------
    # Calibration dataset
    # --------------------------------------------------------

    class CaliDataset(torch.utils.data.Dataset):

        def __init__(self, path, img_shape):
            super().__init__()

            height, width = img_shape

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize(
                    mean=[0, 0, 0],
                    std=[1, 1, 1]
                ),
            ])

            self.imgs_path = []

            for img_name in sorted(os.listdir(path)):
                if not img_name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):
                    continue

                self.imgs_path.append(
                    os.path.join(path, img_name)
                )

        def __len__(self):
            return len(self.imgs_path)

        def __getitem__(self, idx):
            img = Image.open(self.imgs_path[idx]).convert("RGB")
            return self.transform(img)

    # --------------------------------------------------------
    # Load ONNX
    # --------------------------------------------------------

    print("=" * 70)
    print("STEP 2: LOADING ONNX")
    print("=" * 70)

    print(f"ONNX:")
    print(f"  {BEST_ONNX}")

    model = onnx.load(str(BEST_ONNX))

    print("ONNX loaded.")

    # --------------------------------------------------------
    # Simplify
    # --------------------------------------------------------

    print("\nSimplifying ONNX...")

    model, check = simplify(model)

    if not check:
        raise RuntimeError(
            "ONNX simplification validation failed."
        )

    # IMPORTANT:
    # Do NOT overwrite best.onnx.
    # Save the simplified copy to /tmp.
    simplified_onnx = Path("/tmp/pigeon_best_160_288_simplified.onnx")

    model = onnx.shape_inference.infer_shapes(model)

    onnx.save(
        model,
        str(simplified_onnx)
    )

    print(f"Simplified ONNX:")
    print(f"  {simplified_onnx}")

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3: CALIBRATION DATASET")
    print("=" * 70)

    print(f"Calibration directory:")
    print(f"  {CALIB_DIR}")

    calibration_dataset = CaliDataset(
        str(CALIB_DIR),
        img_shape=IMG_SIZE
    )

    print(f"Calibration images: {len(calibration_dataset)}")

    dataloader = DataLoader(
        dataset=calibration_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    def collate_fn(batch):
        return torch.stack(batch).to(DEVICE)

    # --------------------------------------------------------
    # Quantization settings
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4: ESP-PPQ SETTINGS")
    print("=" * 70)

    quant_setting = QuantizationSettingFactory.espdl_setting()

    quant_setting.equalization = True
    quant_setting.equalization_setting.iterations = 4
    quant_setting.equalization_setting.value_threshold = 0.4
    quant_setting.equalization_setting.opt_level = 2
    quant_setting.equalization_setting.interested_layers = None

    print("Equalization : True")
    print("Iterations   : 4")
    print("Threshold    : 0.4")
    print("Opt level    : 2")
    print(f"Input shape  : [1, 3, {IMG_SIZE[0]}, {IMG_SIZE[1]}]")
    print(f"Target       : {TARGET}")
    print(f"Bits         : {NUM_BITS}")
    print(f"Calib steps  : {CALIB_STEPS}")
    print()

    # --------------------------------------------------------
    # Quantize
    # --------------------------------------------------------

    print("=" * 70)
    print("STEP 5: INT8 QUANTIZATION")
    print("=" * 70)

    if OUTPUT_ESPDL.exists():
        print(f"Removing previous temporary model:")
        print(f"  {OUTPUT_ESPDL}")
        OUTPUT_ESPDL.unlink()

    print("\nStarting ESP-PPQ quantization...\n")

    graph = espdl_quantize_onnx(
        onnx_import_file=str(simplified_onnx),

        espdl_export_file=str(OUTPUT_ESPDL),

        calib_dataloader=dataloader,

        calib_steps=CALIB_STEPS,

        input_shape=[
            1,
            3,
            IMG_SIZE[0],
            IMG_SIZE[1],
        ],

        target=TARGET,

        num_of_bits=NUM_BITS,

        collate_fn=collate_fn,

        setting=quant_setting,

        device=DEVICE,

        error_report=True,

        skip_export=False,

        export_test_values=False,

        verbose=0,

        inputs=None,
    )

    print("\nINT8 quantization finished.")

    if not OUTPUT_ESPDL.exists():
        raise RuntimeError(
            f"ESP-PPQ finished but did not create:\n{OUTPUT_ESPDL}"
        )

    print()
    print("Generated ESPDL:")
    print(f"  {OUTPUT_ESPDL}")
    print(f"Size: {OUTPUT_ESPDL.stat().st_size / 1024:.1f} KB")

    return graph


# ============================================================
# QUANTIZED GRAPH INFERENCE
# ============================================================

def validate_quantized_graph(graph):

    print("\n" + "=" * 70)
    print("STEP 6: QUANTIZED GRAPH VALIDATION")
    print("=" * 70)

    import torch

    from esp_ppq.executor import TorchExecutor

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.torch_utils import smart_inference_mode
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import unwrap_model
    from ultralytics.utils.ops import Profile

    from ultralytics.nn.modules.head import Detect

    executor = TorchExecutor(
        graph=graph,
        device=DEVICE
    )

    print("TorchExecutor created.")

    # --------------------------------------------------------
    # PPQ -> ESPDet output conversion
    # --------------------------------------------------------

    def ppq_graph_inference(inputs):

        NC = 1

        print(
            "Running quantized inference with input:",
            tuple(inputs.shape)
        )

        graph_outputs = executor(inputs)

        print(
            "Graph outputs:",
            len(graph_outputs)
        )

        for i, output in enumerate(graph_outputs):
            print(
                f"  output[{i}] shape={tuple(output.shape)} "
                f"dtype={output.dtype}"
            )

        # ESPDet produces:
        #
        # box0, score0
        # box1, score1
        # box2, score2
        #
        boxes_ls = [
            graph_outputs[i]
            for i in range(0, 6, 2)
        ]

        bs = inputs.shape[0]

        boxes = torch.cat(
            [
                graph_outputs[2 * i].view(
                    bs,
                    4,
                    -1
                )
                for i in range(3)
            ],
            dim=-1,
        )

        scores = torch.cat(
            [
                graph_outputs[2 * i + 1].view(
                    bs,
                    NC,
                    -1
                )
                for i in range(3)
            ],
            dim=-1,
        )

        preds = {
            "boxes": boxes,
            "scores": scores,
            "feats": boxes_ls,
        }

        detect_model = Detect(
            nc=NC,
            reg_max=1,
            end2end=False,
            ch=[32, 64, 128],
        )

        detect_model.stride = [
            8.0,
            16.0,
            32.0,
        ]

        detect_model.to(DEVICE)

        return detect_model._inference(preds)

    # --------------------------------------------------------
    # Direct test on calibration images
    # --------------------------------------------------------

    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (IMG_SIZE[0], IMG_SIZE[1])
        ),
        transforms.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1]
        ),
    ])

    images = [
        p for p in CALIB_DIR.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        }
    ]

    print()
    print(f"Testing {len(images)} calibration images...")

    maximum_score = 0.0
    maximum_image = None

    for index, image_path in enumerate(images):

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = transform(image).unsqueeze(0)

            predictions = ppq_graph_inference(
                tensor
            )

            # Predictions returned by Ultralytics
            # Detect._inference have shape:
            #
            # [batch, 4 + nc, candidates]
            #
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            pred = predictions[0]

            if pred.numel() == 0:
                score = 0.0
            else:
                # class confidence is the last row
                score = float(
                    pred[4:].max().item()
                )

            if score > maximum_score:
                maximum_score = score
                maximum_image = image_path

            print(
                f"[{index + 1:03d}/{len(images):03d}] "
                f"{image_path.name}: "
                f"max_score={score:.6f}"
            )

        except Exception as exc:
            print(
                f"[{index + 1:03d}] "
                f"{image_path.name}: ERROR: {exc}"
            )

    print("\n" + "=" * 70)
    print("QUANTIZED GRAPH RESULT")
    print("=" * 70)

    print(
        f"Maximum confidence: {maximum_score:.6f}"
    )

    if maximum_image:
        print(
            f"Image with maximum confidence:"
            f"\n  {maximum_image}"
        )

    if maximum_score >= 0.25:
        print()
        print("RESULT: INT8 GRAPH IS PRODUCING DETECTIONS.")
        print()
        print(
            "This means the INT8 quantization path is probably "
            "working and we should investigate the ESP32 "
            "preprocessing/runtime path."
        )
    else:
        print()
        print("RESULT: INT8 GRAPH IS NOT PRODUCING A STRONG DETECTION.")
        print()
        print(
            "This points toward the INT8 quantization/calibration "
            "pipeline rather than the ESP32 camera."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PIGEON ESPDET INT8 DIAGNOSTIC")
    print("=" * 70)
    print()

    print("Python:")
    print(f"  {sys.executable}")
    print()

    print("Configuration:")
    print(f"  ONNX       = {BEST_ONNX}")
    print(f"  PT         = {BEST_PT}")
    print(f"  Calibration= {CALIB_DIR}")
    print(f"  Data YAML  = {DATA_YAML}")
    print(f"  Image size = {IMG_SIZE}")
    print(f"  Target     = {TARGET}")
    print(f"  Bits       = {NUM_BITS}")
    print(f"  Device     = {DEVICE}")
    print()

    check_file(BEST_ONNX, "best.onnx")
    check_file(BEST_PT, "best.pt")
    check_file(DATA_YAML, "bird.yaml")
    check_calibration()

    # This exits with a useful error if esp_ppq is absent.
    import_ppq()

    try:

        graph = quantize_model()

        validate_quantized_graph(graph)

    except Exception:

        print()
        print("=" * 70)
        print("DIAGNOSTIC FAILED")
        print("=" * 70)
        print()

        traceback.print_exc()

        sys.exit(1)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
