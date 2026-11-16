import os
import shutil

CUB_DIR = os.path.expanduser("/Users/piyushpilania/development/pigeon_model/esp/CUB_200_2011")
OUTPUT_DIR = os.path.expanduser("/Users/piyushpilania/development/pigeon_model/esp/esp-detection/datasets/bird")

# Create directories
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# Read image paths
images = {}

with open(os.path.join(CUB_DIR, "images.txt")) as f:
    for line in f:
        image_id, image_path = line.strip().split(" ", 1)
        images[int(image_id)] = image_path

# Read bounding boxes
boxes = {}

with open(os.path.join(CUB_DIR, "bounding_boxes.txt")) as f:
    for line in f:
        image_id, x, y, width, height = line.strip().split()
        boxes[int(image_id)] = (
            float(x),
            float(y),
            float(width),
            float(height)
        )

# Read official train/test split
splits = {}

with open(os.path.join(CUB_DIR, "train_test_split.txt")) as f:
    for line in f:
        image_id, is_train = line.strip().split()
        splits[int(image_id)] = int(is_train)

count_train = 0
count_val = 0

for image_id, image_path in images.items():

    if image_id not in boxes:
        continue

    source_image = os.path.join(CUB_DIR, "images", image_path)

    # Determine split
    if splits[image_id] == 1:
        split = "train"
        count_train += 1
    else:
        split = "val"
        count_val += 1

    filename = os.path.basename(image_path)

    destination_image = os.path.join(
        OUTPUT_DIR,
        "images",
        split,
        filename
    )

    destination_label = os.path.join(
        OUTPUT_DIR,
        "labels",
        split,
        os.path.splitext(filename)[0] + ".txt"
    )

    # Copy image
    shutil.copy2(source_image, destination_image)

    # Get image dimensions
    from PIL import Image

    with Image.open(source_image) as img:
        img_width, img_height = img.size

    x, y, width, height = boxes[image_id]

    # Convert CUB bbox to YOLO format
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    # Every bird = class 0
    with open(destination_label, "w") as f:
        f.write(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{norm_width:.6f} {norm_height:.6f}\n"
        )

print("Conversion complete!")
print(f"Training images: {count_train}")
print(f"Validation images: {count_val}")
print(f"Output: {OUTPUT_DIR}")
