import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def detect_eyes_and_extract_features(
    model_path: str,
    images_dir: str,
    output_csv: str,
    conf_thres: float = 0.25,
):
    model = YOLO(model_path)
    images_dir = Path(images_dir)
    rows = []

    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Run detection
        results = model(img, imgsz=256, conf=conf_thres)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N,4)

        eyes_info = []
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            aspect_ratio = cw / (ch + 1e-6)   # shape feature

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # brightness: mean intensity
            brightness = float(gray.mean())

            # openness: eye box height relative to full image height
            openness = ch / h

            eyes_info.append({
                "x_center": (x1 + x2) / 2 / w,
                "y_center": (y1 + y2) / 2 / h,
                "width_norm": cw / w,
                "height_norm": ch / h,
                "aspect_ratio": aspect_ratio,
                "brightness": brightness,
                "openness": openness,
            })

        num_eyes = len(eyes_info)

        if num_eyes > 0:
            avg_aspect = float(np.mean([e["aspect_ratio"] for e in eyes_info]))
            avg_brightness = float(np.mean([e["brightness"] for e in eyes_info]))
            avg_openness = float(np.mean([e["openness"] for e in eyes_info]))
        else:
            avg_aspect = avg_brightness = avg_openness = None

        # symmetry: if exactly 2 eyes, compare left/right
        symmetry_diff_aspect = None
        symmetry_diff_brightness = None
        if num_eyes == 2:
            eyes_sorted = sorted(eyes_info, key=lambda e: e["x_center"])
            left, right = eyes_sorted
            symmetry_diff_aspect = abs(left["aspect_ratio"] - right["aspect_ratio"])
            symmetry_diff_brightness = abs(left["brightness"] - right["brightness"])

        rows.append([
            img_path.name,
            num_eyes,
            avg_aspect,
            avg_brightness,
            avg_openness,
            symmetry_diff_aspect,
            symmetry_diff_brightness,
        ])

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "num_eyes",
            "avg_aspect_ratio",
            "avg_brightness",
            "avg_openness",
            "symmetry_diff_aspect",
            "symmetry_diff_brightness",
        ])
        writer.writerows(rows)

    print(f"Saved features to {output_csv}")


if __name__ == "__main__":
    detect_eyes_and_extract_features(
        model_path=r"runs/eye_yolo/exp_eye_tiny2/weights/best.pt",
        images_dir=r"datasets/eyes-yolo/valid/images",
        output_csv="eye_features.csv",
    )
