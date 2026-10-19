import json, random, sys, time, csv, os, platform
import numpy as np
import cv2
import torch
import ultralytics
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 50
IMGSZ = int(sys.argv[4]) if len(sys.argv) > 4 else 640
CONF = 0.25
WARMUP = int(sys.argv[5]) if len(sys.argv) > 5 else 10
ROOT = "data/bdd100k/"

use_cuda = device == "cuda"
if use_cuda and not torch.cuda.is_available():
    sys.exit("CUDA not available in this environment; run with cpu instead")
dev = 0 if use_cuda else "cpu"

samples = json.load(open(ROOT + "samples.json"))["samples"]

def tag(s, k):
    return (s.get(k) or {}).get("label")

random.seed(0)
paths = []
for t in ["daytime", "night"]:
    pool = [s for s in samples if tag(s, "timeofday") == t]
    paths += [s["filepath"] for s in random.sample(pool, min(n_per, len(pool)))]

imgs = [cv2.imread(ROOT + p) for p in paths]
imgs = [i for i in imgs if i is not None]
print(len(imgs), "images loaded, size", imgs[0].shape[1], "x", imgs[0].shape[0])

model = YOLO(model_name)

def sync():
    if use_cuda:
        torch.cuda.synchronize()

for i in range(WARMUP):
    model.predict(imgs[i % len(imgs)], imgsz=IMGSZ, conf=CONF, device=dev, verbose=False)

total, pre, inf, post = [], [], [], []
for img in imgs:
    sync()
    t0 = time.perf_counter()
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, device=dev, verbose=False)[0]
    sync()
    total.append((time.perf_counter() - t0) * 1000)
    pre.append(r.speed["preprocess"])
    inf.append(r.speed["inference"])
    post.append(r.speed["postprocess"])

def med(a): return float(np.median(a))
def p95(a): return float(np.percentile(a, 95))

hw = torch.cuda.get_device_name(0) if use_cuda else platform.processor()
print("model:", model_name, "| device:", device, "|", hw)
print("imgsz:", IMGSZ, "| conf:", CONF, "| batch 1 | images timed:", len(total), "| warmup:", WARMUP)
print(f"  end-to-end ms: median {med(total):.1f}  mean {np.mean(total):.1f}  p95 {p95(total):.1f}  max {max(total):.1f}")
print(f"  stages median ms: preprocess {med(pre):.1f}  inference {med(inf):.1f}  postprocess {med(post):.1f}")
print(f"  FPS from median end-to-end: {1000 / med(total):.1f}")

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/latency.csv"
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["model", "device", "hardware", "imgsz", "conf", "n", "warmup", "median_ms", "mean_ms", "p95_ms", "max_ms",
                    "pre_ms", "inf_ms", "post_ms", "fps", "torch", "ultralytics", "timestamp"])
    w.writerow([model_name, device, hw, IMGSZ, CONF, len(total), WARMUP, round(med(total), 1), round(float(np.mean(total)), 1),
                round(p95(total), 1), round(max(total), 1), round(med(pre), 1), round(med(inf), 1), round(med(post), 1),
                round(1000 / med(total), 1), torch.__version__, ultralytics.__version__, time.strftime("%Y-%m-%d %H:%M")])
print("appended to", out)
