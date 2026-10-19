"""Day 2, Step 1: basic OpenCV operations on a still image."""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
OUT_DIR = PROJECT_ROOT / "experiments" / "results"


def main() -> None:
    # 1. Read the image. imread returns None (not an error) if the file is bad.
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print(f"Could not read image: {IMAGE_PATH}")
        sys.exit(1)

    # 2. Inspect it.
    height, width, channels = image.shape
    print(f"Shape (height, width, channels): {image.shape}")
    print(f"Width x height: {width} x {height}")
    print(f"Data type: {image.dtype}")
    print(f"Pixel at x=100, y=100 (B, G, R): {image[100, 100]}")

    # 3. Resize to half size. Note: resize takes (width, height).
    half = cv2.resize(image, (width // 2, height // 2))
    print(f"Half-size shape: {half.shape}")

    # 4. Draw on a copy, so the original stays untouched.
    drawn = image.copy()
    cv2.rectangle(drawn, (50, 50), (300, 300), (0, 255, 0), 3)   # green, 3 px thick
    cv2.putText(drawn, "test box", (50, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # 5. Save the results.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "bus_half.jpg"), half)
    cv2.imwrite(str(OUT_DIR / "bus_drawn.jpg"), drawn)
    print(f"Saved bus_half.jpg and bus_drawn.jpg to: {OUT_DIR}")


if __name__ == "__main__":
    main()
