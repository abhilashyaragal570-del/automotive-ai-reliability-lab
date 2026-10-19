import json, random, sys, csv, os
import numpy as np
import cv2
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
ROOT = "data/bdd100k/"
MIN_H = 20
MINN = 30
MAP = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
       "motorcycle": "motorcycle", "pedestrian": "person", "rider": "person",
       "traffic light": "traffic light"}
KEEP = set(MAP.values())
BANDS = [("20-40px", 20, 40), ("40-80px", 40, 80), ("80px+", 80, 10**9)]
BRIGHT = [("<30", 0, 30), ("30-50", 30, 50), ("50-80", 50, 80), ("80+", 80, 256)]

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
        if cls != "car":
            continue
        h = box[3] - box[1]
        band = None
        for bname, lo, hi in BANDS:
            if lo <= h < hi:
                band = bname
        if band is None:
            continue
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(W, int(box[2])), min(H, int(box[3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        records.append((t, band, float(gray[y1:y2, x1:x2].mean()), j in hit))

def cell(rows):
    k = len(rows)
    return k, (100.0 * sum(1 for r in rows if r[3]) / k if k else 0.0)

def dark(rows):
    return 100.0 * sum(1 for r in rows if r[2] < 30) / len(rows) if rows else 0.0

print("model:", model_name, "conf:", CONF, "| cars only; brightness = mean gray inside the label box; a gap is shown only if both sides have", MINN, "or more cars")
out_rows = []
for bname, _, _ in BANDS:
    rb = [r for r in records if r[1] == bname]
    d_all = [r for r in rb if r[0] == "daytime"]
    n_all = [r for r in rb if r[0] == "night"]
    print("band:", bname, f"| cars day {len(d_all)} night {len(n_all)} | share with gray below 30: day {dark(d_all):.0f}%, night {dark(n_all):.0f}%")
    print(f"  {'bright':8}{'day n':>7}{'day hit%':>10}{'night n':>9}{'night hit%':>12}{'gap':>6}")
    use = []
    for gname, lo, hi in BRIGHT:
        d = [r for r in d_all if lo <= r[2] < hi]
        n = [r for r in n_all if lo <= r[2] < hi]
        (dn, dh), (nn, nh) = cell(d), cell(n)
        ok = dn >= MINN and nn >= MINN
        gap = f"{nh - dh:>6.0f}" if ok else "   n/a"
        print(f"  {gname:8}{dn:>7}{dh:>10.0f}{nn:>9}{nh:>12.0f}{gap}")
        out_rows.append([bname, gname, dn, round(dh, 1), nn, round(nh, 1)])
        if ok:
            use.append((dn, dh, nn, nh))
    (dn, dh), (nn, nh) = cell(d_all), cell(n_all)
    print(f"  {'all':8}{dn:>7}{dh:>10.0f}{nn:>9}{nh:>12.0f}{nh - dh:>6.0f}")
    if use:
        wd = sum(u[0] for u in use)
        day_h = sum(u[0] * u[1] for u in use) / wd
        night_rw = sum(u[0] * u[3] for u in use) / wd
        cov_d = 100.0 * wd / len(d_all)
        cov_n = 100.0 * sum(u[2] for u in use) / len(n_all)
        print(f"  matched-brightness gap: day {day_h:.0f}%, night reweighted to day mix {night_rw:.0f}%, gap {night_rw - day_h:.0f}; usable bins cover {cov_d:.0f}% of day cars and {cov_n:.0f}% of night cars")
    else:
        print("  no bin has enough cars on both sides")

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_size_bright_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["band", "brightness", "day_n", "day_hit_pct", "night_n", "night_hit_pct"])
    w.writerows(out_rows)
print("saved", out)
