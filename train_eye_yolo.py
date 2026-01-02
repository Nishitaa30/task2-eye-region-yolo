from ultralytics import YOLO

def main():
    # Load tiny pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Very small training for weak CPU
    model.train(
        data="datasets/eyes-yolo/data.yaml",
        epochs=2,          # only 2 epochs
        imgsz=256,         # smaller images -> faster
        batch=2,           # small batch
        device="cpu",      # you are on CPU
        workers=0,         # no extra dataloader workers [web:58][web:59]
        project="runs/eye_yolo",
        name="exp_eye_tiny",
        patience=2,
    )

if __name__ == "__main__":
    main()
