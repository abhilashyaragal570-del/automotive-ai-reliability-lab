import json, random, sys, csv, os, collections as C
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
LOW = 0.05
ROOT = "data/bdd100k/"
MIN_H = 20
MAP = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
       "motorcycle": "motorcycle", "pedestrian": "person", "rider": "person",
       "traffic light": "traffic light"}
KEEP = set(MAP.values())
BUCKETS = [("20-40px", 20, 40), ("40-80px", 40, 80), ("80px+", 80, 10**9), ("any", 20, 10**9)]
CATS = ["hit", "claimed", "low-conf", "wrong-class", "poor-box", "nothing"]

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

def why(cls, box, preds):
    best_same, other, partial = -1.0, False, False
    for cn, c, b in preds:
        v = iou(b, box)
        if cn == cls and v >= 0.5:
            best_same = max(best_same, c)
        elif cn != cls and v >= 0.5:
            other = True
        elif v >= 0.1:
            partial = True
    if best_same >= CONF:
        return "claimed"
    if best_same >= 0:
        return "low-conf"
    if other:
        return "wrong-class"
    if partial:
        return "poor-box"
    return "nothing"

model = YOLO(model_name)
names = model.names
paths = list(chosen)
per_image = {}
for i in range(0, len(paths), 16):
    batch = paths[i:i+16]
    results = model.predict([ROOT + p for p in batch], conf=LOW, verbose=False)
    for p, r in zip(batch, results):
        per_image[p] = [(names[int(k)], c, b) for b, c, k in
                        zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

stat = C.defaultdict(C.Counter)
for p, (t, s) in chosen.items():
    gts = gt_boxes(s)
    strong = [(cn, c, b) for cn, c, b in per_image[p]
              if cn in KEEP and c >= CONF and b[3] - b[1] >= MIN_H]
    hit = matched(gts, strong)
    for j, (cls, box) in enumerate(gts):
        h = box[3] - box[1]
        cat = "hit" if j in hit else why(cls, box, per_image[p])
        groups = ["all"] + (["car"] if cls == "car" else [])
        for bname, lo, hi in BUCKETS:
            if lo <= h < hi:
                for g in groups:
                    stat[(g, t, bname)][cat] += 1

print("model:", model_name, "conf:", CONF, "(share of labeled objects, %)")
rows = []
for g in ["all", "car"]:
    print(f"{'grp':5}{'time':9}{'height':9}{'n':>6}" + "".join(f"{c:>12}" for c in CATS))
    for bname, lo, hi in BUCKETS:
        for t in ["daytime", "night"]:
            cnt = stat[(g, t, bname)]
            n = sum(cnt.values())
            pcts = [100 * cnt[c] / n if n else 0 for c in CATS]
            print(f"{g:5}{t:9}{bname:9}{n:>6}" + "".join(f"{x:>12.0f}" for x in pcts))
            rows.append([g, t, bname, n] + [round(x, 1) for x in pcts])

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_misses_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "time", "height", "n"] + CATS)
    w.writerows(rows)
print("saved", out)
