import json, random, sys, csv, os, collections as C
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 50
ROOT = "data/bdd100k/"
MIN_H = 20
CONF = 0.25
MAP = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
       "motorcycle": "motorcycle", "pedestrian": "person", "rider": "person",
       "traffic light": "traffic light"}
KEEP = set(MAP.values())
SLICES = [("weather", "clear"), ("weather", "rainy"), ("weather", "snowy"),
          ("weather", "overcast"), ("timeofday", "daytime"), ("timeofday", "night"),
          ("scene", "city street"), ("scene", "highway")]

samples = json.load(open(ROOT + "samples.json"))["samples"]

def tag(s, k):
    return (s.get(k) or {}).get("label")

random.seed(0)
chosen = {}
for k, v in SLICES:
    pool = [s for s in samples if tag(s, k) == v]
    for s in random.sample(pool, min(n_per, len(pool))):
        chosen[s["filepath"]] = s
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

def score(gts, preds):
    tp = fp = 0
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
            tp += 1
            used.add(bi)
        else:
            fp += 1
    return tp, fp, len(gts) - len(used)

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

agg = C.defaultdict(lambda: [0, 0, 0, 0])
for p, s in chosen.items():
    tp, fp, fn = score(gt_boxes(s), per_image[p])
    for k, v in SLICES:
        if tag(s, k) == v:
            a = agg[(k, v)]
            a[0] += 1
            a[1] += tp
            a[2] += fp
            a[3] += fn

print("model:", model_name)
print(f"{'slice':26}{'imgs':>6}{'TP':>7}{'FP':>7}{'FN':>7}{'prec':>7}{'recall':>8}")
rows = []
for (k, v), (n, tp, fp, fn) in agg.items():
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    print(f"{k + ':' + v:26}{n:>6}{tp:>7}{fp:>7}{fn:>7}{pr:>7.2f}{rc:>8.2f}")
    rows.append([k, v, n, tp, fp, fn, round(pr, 3), round(rc, 3)])

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_slices_" + model_name.replace(".pt", "") + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attr", "value", "images", "tp", "fp", "fn", "precision", "recall"])
    w.writerows(rows)
print("saved", out)
