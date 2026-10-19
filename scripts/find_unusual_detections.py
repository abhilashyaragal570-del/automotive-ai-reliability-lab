"""Day 2, Step 4: save a few evidence frames for unusual detected classes."""

import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "road_clip.mp4"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "suspect"

MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Classes that look unusual for a road scene (our hypothesis, to be checked by eye).
UNUSUAL = {"airplane", "train", "laptop", "bench", "bird", "traffic light"}
MAX_PER_CLASS = 3


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_NAME)
    saved = Counter()
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(source=frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)[0]

        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name in UNUSUAL and saved[name] < MAX_PER_CLASS:
                conf = float(box.conf[0])
                seconds = frame_index / fps
                filename = f"{name.replace(' ', '_')}_f{frame_index:04d}_conf{conf:.2f}.jpg"
                cv2.imwrite(str(OUT_DIR / filename), result.plot())
                saved[name] += 1
                print(f"{name:<14} frame {frame_index:>3} (t={seconds:4.1f}s) conf={conf:.2f} -> {filename}")

        frame_index += 1

    cap.release()
    print(f"Done. Frames saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
