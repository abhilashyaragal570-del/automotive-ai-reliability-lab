import json, random, sys, csv, os
import cv2
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26s.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
N_REV = int(sys.argv[4]) if len(sys.argv) > 4 else 25
ROOT = "data/bdd100k/"
MIN_H = 20
REV_H = 40
PAD = 100
BANDS = [("20-40px", 20, 40), ("40-80px", 40, 80), ("80px+", 80, 10**9)]

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

def band_of(h):
    for name, lo, hi in BANDS:
        if lo <= h < hi:
            return name
    return None

def gt_cars(s):
    w = s["metadata"]["width"]
    h = s["metadata"]["height"]
    out = []
    for d in (s.get("detections") or {}).get("detections", []):
        if d["label"] != "car":
            continue
        x, y, bw, bh = d["bounding_box"]
        out.append([x*w, y*h, (x+bw)*w, (y+bh)*h])
    return out

model = YOLO(model_name)
names = model.names
paths = list(chosen)
counts = {}
cands = {"daytime": [], "night": []}
for i in range(0, len(paths), 16):
    batch = paths[i:i+16]
    results = model.predict([ROOT + p for p in batch], conf=CONF, verbose=False)
    for p, r in zip(batch, results):
        t, s = chosen[p]
        gt_all = gt_cars(s)
        gt_main = [g for g in gt_all if g[3] - g[1] >= MIN_H]
        preds = [(c, b) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                 if names[int(k)] == "car" and b[3] - b[1] >= MIN_H]
        preds.sort(key=lambda x: -x[0])
        used = set()
        for c, b in preds:
            best, bi = 0, None
            for j, gb in enumerate(gt_main):
                if j in used:
                    continue
                v = iou(b, gb)
                if v > best:
                    best, bi = v, j
            if bi is not None and best >= 0.5:
                used.add(bi)
                continue
            m = max([iou(b, g) for g in gt_all], default=0)
            cat = "dup" if m >= 0.5 else ("partial" if m >= 0.1 else "none")
            h = b[3] - b[1]
            key = (t, band_of(h), cat)
            counts[key] = counts.get(key, 0) + 1
            if h >= REV_H:
                cands[t].append((p, b, c, cat))

print("model:", model_name, "conf:", CONF, "| unmatched car predictions (FP) by predicted height. on labeled car = IoU 0.5+ with a label already matched (duplicate or extra box); poor overlap = IoU 0.1 to 0.5; no label near = IoU below 0.1 with every labeled car")
print(f"  {'time':9}{'band':9}{'FP':>6}{'on labeled car':>16}{'poor overlap':>14}{'no label near':>15}")
for t in ["daytime", "night"]:
    for bname, _, _ in BANDS:
        d = counts.get((t, bname, "dup"), 0)
        pa = counts.get((t, bname, "partial"), 0)
        no = counts.get((t, bname, "none"), 0)
        print(f"  {t:9}{bname:9}{d + pa + no:>6}{d:>16}{pa:>14}{no:>15}")

outdir = "experiments/results/fp_review_" + model_name.replace(".pt", "") + "_conf" + str(CONF)
os.makedirs(outdir, exist_ok=True)
rng = random.Random(1)
index = []
for t in ["daytime", "night"]:
    pick = rng.sample(cands[t], min(N_REV, len(cands[t])))
    for k, (p, b, c, cat) in enumerate(pick):
        img = cv2.imread(ROOT + p)
        if img is None:
            continue
        H, W = img.shape[:2]
        for g in gt_cars(chosen[p][1]):
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 2)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        x1, y1 = max(0, int(b[0]) - PAD), max(0, int(b[1]) - PAD)
        x2, y2 = min(W, int(b[2]) + PAD), min(H, int(b[3]) + PAD)
        name = f"{t}_{k:02d}.jpg"
        cv2.imwrite(os.path.join(outdir, name), img[y1:y2, x1:x2])
        index.append([name, p, round(c, 2), round(b[3] - b[1]), cat])
with open(os.path.join(outdir, "index.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "image", "conf", "pred_height", "auto_category"])
    w.writerows(index)
print("saved", len(index), "crops to", outdir)
