# Automotive AI Perception & Reliability Lab

Work in progress. A learning and portfolio project that runs a pretrained YOLO
detector on recorded road images and video and will later evaluate how reliably
it performs across driving conditions.

This is a recorded-data engineering project. It does not control a vehicle and
makes no safety or certification claims.

## Status

Phase 1 (environment and perception) is done. Phase 2 (evaluation on labelled data) is in progress: it uses BDD100K ground truth to measure recall and false positives for cars, day versus night, in three YOLO models, and latency is measured in [docs/latency.md](docs/latency.md). See [docs/findings_night_recall.md](docs/findings_night_recall.md). The Phase 1 clip below has no ground truth, so its numbers are detection counts only. Precision over the whole label set and mAP are not computed yet.

## Setup (Windows, PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install ultralytics

A CPU-only PyTorch is installed by default. For an NVIDIA GPU, choose the CUDA
build on pytorch.org, uninstall torch and torchvision first, then install it.
Scripts pick the GPU automatically if available and fall back to the CPU.

Tested with: Windows 11, Python 3.14.8, ultralytics 8.4.182,
torch 2.14.1+cu126, OpenCV 5.0.0, RTX 3050 Laptop GPU (4 GB), model yolo26n.pt.

## Scripts

| Script | Purpose |
|---|---|
| scripts/detect_one_image.py | Detect objects in one image, save an annotated copy |
| scripts/opencv_basics.py | Learning script: read, resize, draw, save images |
| scripts/read_video.py | Read a video frame by frame, report its properties |
| scripts/detect_video.py | Detect on every frame, save an annotated video |
| scripts/find_unusual_detections.py | Save evidence frames for unusual classes |
| scripts/threshold_experiment.py | Count detections per class at several confidence thresholds |

Images and videos in data/ are not committed: the sample image is third-party
and the test video is private footage.

## First observations (one 9-second clip, informal)

The clip is a phone video from a car passenger seat through the windshield
(576x1024, about 30 FPS, 269 frames). Detections below are summed over all
frames, so they are not object counts and not accuracy.

| class | >=0.25 | >=0.40 | >=0.50 | >=0.60 |
|---|---|---|---|---|
| car | 177 | 100 | 69 | 44 |
| person | 128 | 99 | 74 | 54 |
| truck | 126 | 62 | 21 | 5 |
| bus | 115 | 101 | 78 | 43 |
| airplane | 11 | 1 | 1 | 0 |
| train | 7 | 1 | 0 | 0 |
| laptop | 5 | 2 | 2 | 0 |
| bench | 4 | 1 | 0 | 0 |
| traffic light | 1 | 0 | 0 | 0 |
| bird | 1 | 1 | 0 | 0 |

Frames saved for airplane, train, laptop, bench, bird and traffic light were
reviewed by eye. In those frames the boxes sat on a side mirror, a roadside
guardrail, litter, a whole-frame region of the car interior, and the front of an
auto-rickshaw. None of the saved frames showed a real airplane or train. The saved frames had confidence 0.25 to 0.53; across the whole clip no detection of these classes reached 0.60.

## Limitations

- One short clip, not a dataset. No conclusions about general reliability.
- The Phase 1 clip has no ground truth. The BDD100K study in the findings file does, but it covers cars only, 600 images, and no mAP.
- The model has 80 fixed classes and no auto-rickshaw class.
- Confidence is the model's own certainty, not accuracy.
- Higher thresholds remove detections but not necessarily only wrong ones.

## Licence

Not yet decided. The licence terms of the detector library and model weights
still need to be reviewed before this project is published as a portfolio piece.


## Findings: day vs night recall on BDD100K

A study of why small YOLO detectors miss more cars at night. Full write-up with tables: [docs/findings_night_recall.md](docs/findings_night_recall.md). Latency measurements: [docs/latency.md](docs/latency.md).

Setup: 600 BDD100K images (300 day, 300 night), cars only, labeled boxes 20+ px, IoU 0.5, models yolo26n, yolo26s and yolov8s.

- At conf 0.25, YOLO finds about 24 points fewer cars at night than by day, in all three models.
- For large cars (80+ px), brightness explains most of the gap. For small cars (20-40 px) it does not, and the cause is untested.
- Lowering conf from 0.25 to 0.05 narrows the gap to about 13 points, but night gets more false positives per image than day (8.71 vs 6.01 for yolo26s).
- Caveats: BDD labels are incomplete, so false positives are overstated. Images are shared across models, so real uncertainty is wider than the simple estimates.

Scripts (in `scripts/`): `compare_yolo_bdd_blur.py`, `compare_yolo_bdd_size_bright.py`, `compare_yolo_bdd_fp.py`, `measure_latency.py`. Run from the repo root, for example `python scripts\compare_yolo_bdd_fp.py yolo26s.pt 300`.

## Data setup

The BDD100K data is not stored in this repo (it is large and has its own licence, see https://bdd-data.berkeley.edu). The scripts expect it under `data/bdd100k/`, with a `samples.json` file listing each image's path, time-of-day tag, size and labeled boxes. Place your copy there and the scripts will run as written. Results are written to `experiments/results/`.
