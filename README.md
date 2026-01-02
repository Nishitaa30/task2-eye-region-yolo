# Eye Region Detection & Feature Analysis (YOLO, Python)

This project implements a basic end‑to‑end pipeline to detect eye regions in face images using YOLO and to extract simple eye features (shape, symmetry, openness, brightness).

## Dataset

- **Source**: YOLOv7‑Eye Detection dataset from Roboflow Universe (single class: `eye`, YOLOv5 format). 
- **Structure**:

  datasets/eyes-yolo/
    train/images
    train/labels
    valid/images
    valid/labels
    test/images
    test/labels
    data.yaml

data.yaml:
    train: ../train/images
    val: ../valid/images
    test: ../test/images
nc: 1
names: ['eye']

## Model & Training
Framework: Ultralytics YOLOv8 (Python package). [web:24][web:30]

Base model: yolov8n.pt (pre‑trained nano model).

Training script: train_eye_yolo.py.

Settings:

Task: object detection (1 class: eye)

Epochs: 2

Image size: 256

Batch size: 2

Device: CPU (workers=0)

Best weights are saved at:

runs/eye_yolo/exp_eye_tiny2/weights/best.pt
Evaluation
Validation is run automatically after training using YOLO’s built‑in evaluation. [web:27]

Images: 357

Instances: 714

Precision: ≈ 0.84

Recall: ≈ 0.82

mAP@0.5: ≈ 0.88

mAP@0.5:0.95: ≈ 0.43

These values show that even a very short CPU‑only fine‑tuning run produces reasonable eye localisation performance for demonstration and feature analysis.

## Sample Predictions
Script: predict_eye_samples.py

Input: datasets/eyes-yolo/valid/images

Output: annotated images with eye bounding boxes:

runs/eye_yolo/pred_samples_tiny/

## Feature Extraction
Script: extract_eye_features.py

Model: runs/eye_yolo/exp_eye_tiny2/weights/best.pt

Input images: datasets/eyes-yolo/valid/images

Per‑image features saved to eye_features.csv:

num_eyes – number of detected eyes

avg_aspect_ratio – average width/height of eye boxes (shape)

avg_brightness – mean grayscale intensity inside eye crops

avg_openness – average eye box height divided by image height

symmetry_diff_aspect – difference in aspect ratio between left and right eyes

symmetry_diff_brightness – difference in brightness between left and right eyes

The result for this is visible as an excel sheet that is named as eye_features.csv



