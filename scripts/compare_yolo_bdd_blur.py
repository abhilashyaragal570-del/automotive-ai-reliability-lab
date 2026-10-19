import json, random, sys, csv, os
import numpy as np
import cv2
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
ROOT = "data/bdd100k/"
MIN_H = 20
CAR_H = 80
MAP = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
       "motorcycle": "motorcycle", "pedestrian": "person", "rider": "person",
       "traffic light": "traffic light"}
KEEP = set(MAP.values())
BRIGHT = [("<30", 0, 30), ("30-50", 30, 50), ("50-80", 50, 80), ("80+", 80, 256)]
LABELS = ["blurrier", "middle", "sharper"]

samples = json.load(open(ROOT + "samples.json"))["samples"]

def tag(s, k):
    return (s.get(k) or {}).get("label")

random.seed(0)
chosen = {}
for t in ["daytime", "night"]:
    pool = [s for s in samples if tag(s, "timeofday") == t]
    for s in random.sample(pool, min(n_per, len(pool))):
        chosen[s["filepath"]] = (t, s)
print(len(chosen), "images selected")

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0

def gt_boxes(s):
    w = s["metadata"]["width"]
    h = s["metadata"]["height"]
    out = []
    for d in (s.get("detections") or {}).get("detections", []):
        cls = MAP.get(d["label"])
        if not cls:
            continue
        x, y, bw, bh = d["bounding_box"]
        box = [x*w, y*h, (x+bw)*w, (y+bh)*h]
        if box[3] - box[1] >= MIN_H:
            out.append((cls, box))
    return out

def matched(gts, preds):
    used = set()
    for cls, conf, b in sorted(preds, key=lambda t: -t[1]):
        best, bi = 0, None
        for j, (gc, gb) in enumerate(gts):
            if gc != cls or j in used:
                continue
            v = iou(b, gb)
            if v > best:
                best, bi = v, j
        if bi is not None and best >= 0.5:
            used.add(bi)
    return used

model = YOLO(model_name)
names = model.names
paths = list(chosen)
per_image = {}
for i in range(0, len(paths), 16):
    batch = paths[i:i+16]
    results = model.predict([ROOT + p for p in batch], conf=CONF, verbose=False)
    for p, r in zip(batch, results):
        preds = []
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            cn = names[int(k)]
            if cn in KEEP and b[3] - b[1] >= MIN_H:
                preds.append((cn, c, b))
        per_image[p] = preds

records = []
for p, (t, s) in chosen.items():
    img = cv2.imread(ROOT + p)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    gts = gt_boxes(s)
    hit = matched(gts, per_image[p])
    for j, (cls, box) in enumerate(gts):
        if cls != "car" or box[3] - box[1] < CAR_H:
            continue
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(W, int(box[2])), min(H, int(box[3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop = gray[y1:y2, x1:x2]
        m = float(crop.mean())
        small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
        records.append((t, m, sharp, j in hit))

def cell(rows):
    k = len(rows)
    return k, (100.0 * sum(1 for r in rows if r[3]) / k if k else 0.0)

def gapstr(dn, dh, nn, nh):
    return f"{nh - dh:>6.0f}" if dn >= 30 and nn >= 30 else "   n/a"

print("model:", model_name, "conf:", CONF, "| large cars (>=80px); sharpness = raw Laplacian variance of the label crop resized to 64x64 (higher = more fine detail, which can also mean more noise)")
print("A) per brightness bin: how well matched are day and night, and does night differ in sharpness?")
print(f"{'bright':8}{'time':9}{'n':>6}{'med gray':>10}{'med sharp':>11}{'hit%':>7}")
for bname, lo, hi in BRIGHT:
    for t in ["daytime", "night"]:
        rows = [r for r in records if r[0] == t and lo <= r[1] < hi]
        if rows:
            k, h = cell(rows)
            print(f"{bname:8}{t:9}{k:>6}{np.median([r[1] for r in rows]):>10.0f}{np.median([r[2] for r in rows]):>11.1f}{h:>7.0f}")
        else:
            print(f"{bname:8}{t:9}{0:>6}")

print("B) inside each brightness bin: sharpness terciles (cut points computed inside the bin, day and night pooled)")
print(f"{'bright':8}{'sharpness':10}{'day n':>7}{'day hit%':>10}{'night n':>9}{'night hit%':>12}{'gap':>6}")
for bname, lo, hi in BRIGHT:
    rows = [r for r in records if lo <= r[1] < hi]
    nd = sum(1 for r in rows if r[0] == "daytime")
    nn_ = sum(1 for r in rows if r[0] == "night")
    if nd < 30 or nn_ < 30:
        print(f"{bname:8}skipped (needs 30+ day and 30+ night cars; has {nd} / {nn_})")
        continue
    cuts = np.percentile([r[2] for r in rows], [100 / 3, 200 / 3])
    tk = lambda v, c=cuts: 0 if v < c[0] else (1 if v < c[1] else 2)
    for k, lab in enumerate(LABELS):
        d = [r for r in rows if r[0] == "daytime" and tk(r[2]) == k]
        n2 = [r for r in rows if r[0] == "night" and tk(r[2]) == k]
        (dn, dh), (nn, nh) = cell(d), cell(n2)
        print(f"{bname:8}{lab:10}{dn:>7}{dh:>10.0f}{nn:>9}{nh:>12.0f}{gapstr(dn, dh, nn, nh)}")

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_blur_cars_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "brightness", "sharpness_raw", "hit"])
    for t, m, sh_, h in records:
        w.writerow([t, round(m, 2), round(sh_, 3), int(h)])
print("saved", out, len(records), "cars")
