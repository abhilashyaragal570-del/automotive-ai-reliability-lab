"""Day 2, Step 5: how do class counts change with the confidence threshold?"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "road_clip.mp4"

MODEL_NAME = "yolo26n.pt"
BASE_CONF = 0.25                      # lowest threshold we keep
THRESHOLDS = [0.25, 0.40, 0.50, 0.60]
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        sys.exit(1)

    model = YOLO(MODEL_NAME)
    detections = []                   # one entry per box: (class name, confidence)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(source=frame, conf=BASE_CONF, device=DEVICE, verbose=False)[0]
        for box in result.boxes:
            detections.append((result.names[int(box.cls[0])], float(box.conf[0])))
    cap.release()

    classes = Counter(name for name, _ in detections).most_common()
    header = f"{'class':<15}" + "".join(f"{'>=' + str(t):>8}" for t in THRESHOLDS)
    print(header)
    for name, _ in classes:
        row = f"{name:<15}"
        for t in THRESHOLDS:
            n = sum(1 for cls, conf in detections if cls == name and conf >= t)
            row += f"{n:>8}"
        print(row)


if __name__ == "__main__":
    main()
