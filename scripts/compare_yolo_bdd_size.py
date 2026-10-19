import json, random, sys, csv, os, collections as C
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
ROOT = "data/bdd100k/"
MIN_H = 20
MAP = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
       "motorcycle": "motorcycle", "pedestrian": "person", "rider": "person",
       "traffic light": "traffic light"}
KEEP = set(MAP.values())
BUCKETS = [("20-40px", 20, 40), ("40-80px", 40, 80), ("80px+", 80, 10**9)]

samples = json.load(open(ROOT + "samples.json"))["samples"]

def tag(s, k):
    return (s.get(k) or {}).get("label")

random.seed(0)
chosen = {}
for t in ["daytime", "night"]:
    pool = [s for s in samples if tag(s, "timeofday") == t]
    print(f"{t:10} pool {len(pool)}")
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

stat = C.defaultdict(lambda: [0, 0])
heights = {"daytime": [], "night": []}
for p, (t, s) in chosen.items():
    gts = gt_boxes(s)
    hit = matched(gts, per_image[p])
    for j, (cls, box) in enumerate(gts):
        h = box[3] - box[1]
        heights[t].append(h)
        groups = ["all"] + (["car"] if cls == "car" else [])
        for bname, lo, hi in BUCKETS:
            if lo <= h < hi:
                for g in groups:
                    stat[(t, g, bname)][0] += 1
                    stat[(t, g, bname)][1] += 1 if j in hit else 0

print("model:", model_name, "conf:", CONF)
for t in heights:
    heights[t].sort()
print("median labeled box height (px):", {t: round(h[len(h)//2]) for t, h in heights.items()})

rows = []
for g in ["all", "car"]:
    print("group:", g)
    print(f"  {'height':8}{'day n':>8}{'day rec':>9}{'night n':>9}{'night rec':>10}{'gap':>6}")
    for bname, lo, hi in BUCKETS:
        d = stat[("daytime", g, bname)]
        n = stat[("night", g, bname)]
        dr = d[1] / d[0] if d[0] else 0
        nr = n[1] / n[0] if n[0] else 0
        print(f"  {bname:8}{d[0]:>8}{dr:>9.2f}{n[0]:>9}{nr:>10.2f}{(nr-dr)*100:>6.0f}")
        rows.append([g, bname, d[0], round(dr, 3), n[0], round(nr, 3)])

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_size_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "height", "day_n", "day_recall", "night_n", "night_recall"])
    w.writerows(rows)
print("saved", out)
