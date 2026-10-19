"""Run a pretrained YOLO detector on one image and save an annotated copy."""

import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Paths are built from this file's location, so the script works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "results" / "bus_annotated.jpg"

MODEL_NAME = "yolo26n.pt"   # nano model: smallest and fastest
CONF_THRESHOLD = 0.25       # discard detections below this confidence
DEVICE = 0 if torch.cuda.is_available() else "cpu"   # GPU 0 if present, else CPU


def main() -> None:
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        sys.exit(1)

    model = YOLO(MODEL_NAME)
    results = model.predict(source=str(IMAGE_PATH), conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    result = results[0]

    device_label = torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU"
    print(f"Model: {MODEL_NAME} | confidence threshold: {CONF_THRESHOLD} | device: {device_label}")
    print(f"Detections found: {len(result.boxes)}")

    for box in result.boxes:
        class_name = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {class_name:<10} conf={confidence:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    annotated = result.plot()   # image with boxes drawn (NumPy array)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), annotated)
    print(f"Saved annotated image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
