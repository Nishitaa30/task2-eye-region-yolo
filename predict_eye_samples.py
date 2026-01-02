from ultralytics import YOLO

def main():
    model = YOLO(r"runs/eye_yolo/exp_eye_tiny2/weights/best.pt")

    model.predict(
        source="datasets/eyes-yolo/valid/images",  # validation images
        imgsz=256,
        conf=0.25,
        save=True,                                  # save annotated images
        project="runs/eye_yolo",
        name="pred_samples_tiny",                   # output folder name
    )

if __name__ == "__main__":
    main()
