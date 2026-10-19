"""Day 2, Step 2b: read a video frame by frame and report its properties."""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "road_clip.mp4"
OUT_DIR = PROJECT_ROOT / "experiments" / "results"


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        sys.exit(1)

    # Properties reported by the file (these can be approximate).
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Size: {width} x {height} | FPS: {fps:.2f} | Reported frames: {reported_frames}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_read = 0
    while True:
        ok, frame = cap.read()      # ok is False when the video ends
        if not ok:
            break
        if frames_read == 0:
            cv2.imwrite(str(OUT_DIR / "clip_first_frame.jpg"), frame)
            print(f"First frame shape (height, width, channels): {frame.shape}")
        frames_read += 1

    cap.release()

    duration = frames_read / fps if fps > 0 else 0
    print(f"Frames actually read: {frames_read} (about {duration:.1f} seconds)")
    print(f"Saved first frame to: {OUT_DIR / 'clip_first_frame.jpg'}")


if __name__ == "__main__":
    main()
