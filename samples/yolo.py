from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("test1.jpg")

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        name = model.names[cls]

        print(f"{name}: {conf:.2f}")
