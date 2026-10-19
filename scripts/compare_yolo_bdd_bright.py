import json, random, sys, csv, os, collections as C
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
BINS = [("0-15", 0, 15), ("15-30", 15, 30), ("30-50", 30, 50), ("50+", 50, 256), ("all", 0, 256)]

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

def area(a):
    return max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))

def inter(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy

def iou(a, b):
    i = inter(a, b)
    ua = area(a) + area(b) - i
    return i / ua if ua > 0 else 0

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

stat = C.defaultdict(lambda: [0, 0, 0])
for p, (t, s) in chosen.items():
    img = cv2.imread(ROOT + p)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    gts = gt_boxes(s)
    preds = per_image[p]
    hit = matched(gts, preds)
    for j, (cls, box) in enumerate(gts):
        if cls != "car" or box[3] - box[1] < CAR_H:
            continue
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(W, int(box[2])), min(H, int(box[3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        m = float(gray[y1:y2, x1:x2].mean())
        is_hit = j in hit
        part = False
        if not is_hit:
            for cn, c, b in preds:
                if cn == "car" and area(b) > 0:
                    i = inter(b, box)
                    if i / area(b) >= 0.8 and i / max(area(box), 1e-9) >= 0.2:
                        part = True
                        break
        for bname, lo, hi in BINS:
            if lo <= m < hi:
                st = stat[(t, bname)]
                st[0] += 1
                st[1] += 1 if is_hit else 0
                st[2] += 1 if part else 0

print("model:", model_name, "conf:", CONF, "| large cars (>=80px) by mean gray level inside the label box")
print(f"{'bright':8}{'day n':>7}{'day hit%':>10}{'day part%':>11}{'night n':>9}{'night hit%':>12}{'night part%':>13}{'gap':>6}")
rows = []
for bname, lo, hi in BINS:
    d = stat[("daytime", bname)]
    n = stat[("night", bname)]
    dh = 100 * d[1] / d[0] if d[0] else 0
    dp = 100 * d[2] / d[0] if d[0] else 0
    nh = 100 * n[1] / n[0] if n[0] else 0
    npart = 100 * n[2] / n[0] if n[0] else 0
    gap = f"{nh - dh:>6.0f}" if d[0] >= 30 and n[0] >= 30 else "   n/a"
    print(f"{bname:8}{d[0]:>7}{dh:>10.0f}{dp:>11.0f}{n[0]:>9}{nh:>12.0f}{npart:>13.0f}{gap}")
    rows.append([bname, d[0], round(dh, 1), round(dp, 1), n[0], round(nh, 1), round(npart, 1)])

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_bright_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["brightness", "day_n", "day_hit_pct", "day_partial_pct", "night_n", "night_hit_pct", "night_partial_pct"])
    w.writerows(rows)
print("saved", out)
