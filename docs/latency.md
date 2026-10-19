# Latency measurements

Hardware: Ryzen 5 5600H CPU, RTX 3050 Laptop GPU (4 GB), Windows 11, laptop plugged in. Script: `scripts/measure_latency.py`. Raw rows: `experiments/results/latency.csv`.

Method: 100 BDD100K images (50 day, 50 night, seed 0, 1280x720), batch 1, imgsz 640, conf 0.25. Images are preloaded, so disk and decode time are excluded. Warm-up images are untimed. GPU timing synchronizes before and after each call.

| Model | Device | Warm-up | Median ms | p95 ms | FPS (median) |
|---|---|---|---|---|---|
| yolo26n | GPU | 50 | 20.6 / 19.1 (two runs) | 31.7 / 25.7 | 48.6 / 52.4 |
| yolo26s | GPU | 50 | 21.4 | 30.8 | 46.8 |
| yolov8s | GPU | 50 | 14.8 | 22.0 | 67.5 |
| yolo26n | CPU | 10 | 34.7 | 41.4 | 28.8 |
| yolo26s | CPU | 10 | 68.9 | 75.1 | 14.5 |
| yolov8s | CPU | 10 | 72.8 | 89.0 | 13.7 |

- A first GPU run of yolo26n (10 warm-up) read 27.7 ms median. It was discarded as a warm-up or power-state artifact after a 50 warm-up rerun gave about 20 ms. First-run GPU rows for yolo26s (22.5 ms) and yolov8s (15.8 ms) agree with their reruns.
- Run-to-run variation on GPU was about 7% for yolo26n, so differences under about 10% are not treated as real. yolo26n and yolo26s cannot be separated on GPU.
- yolov8s is about 30% faster than the yolo26 models on this GPU. The cause was not investigated.
- CPU rows are single runs.

Not measured: video decoding, resizing from disk, tracking, battery or power-limited operation, batch sizes above 1, and any model above s.
