from train import Train

results = Train(
    pretrained_path="pretrained/espdet_pico_160_288_cat.pt",
    dataset="datasets/bird/bird.yaml",
    imgsz=[160, 288],
    epochs=30,
    rect=True,
    device="mps",
)

print("Training complete!")
print("Best model:", results.save_dir / "weights/best.pt")
