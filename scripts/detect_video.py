"""Day 2, Step 3: run the detector on every video frame and save an annotated video."""

import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "road_clip.mp4"
OUT_PATH = PROJECT_ROOT / "experiments" / "results" / "clip_annotated.mp4"

MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (width, height))  # (width, height)!
    if not writer.isOpened():
        print(f"Could not create output video: {OUT_PATH}")
        sys.exit(1)

    model = YOLO(MODEL_NAME)
    counts = Counter()
    frames_done = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(source=frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)[0]

        for box in result.boxes:
            counts[result.names[int(box.cls[0])]] += 1

        writer.write(result.plot())     # frame with boxes drawn
        frames_done += 1
        if frames_done % 50 == 0:
            print(f"  processed {frames_done} frames...")

    cap.release()
    writer.release()

    print(f"Frames processed: {frames_done}")
    print(f"Detections by class (all frames, conf >= {CONF_THRESHOLD}):")
    for name, n in counts.most_common():
        print(f"  {name:<15} {n}")
    print(f"Saved annotated video to: {OUT_PATH}")


if __name__ == "__main__":
    main()
