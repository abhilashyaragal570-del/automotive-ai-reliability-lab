import json, random, sys, csv, os, collections as C
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ROOT = "data/bdd100k/"
MIN_H = 20
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
MAP = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle",
       "motorcycle": "motorcycle", "pedestrian": "person", "rider": "person",
       "traffic light": "traffic light"}
KEEP = set(MAP.values())

samples = json.load(open(ROOT + "samples.json"))["samples"]

def tag(s, k):
    return (s.get(k) or {}).get("label")

SLICES = [("daytime", lambda s: tag(s, "timeofday") == "daytime"),
          ("night", lambda s: tag(s, "timeofday") == "night")]
for t in ["daytime", "night"]:
    for w in ["clear", "overcast", "rainy", "snowy"]:
        SLICES.append((t + "+" + w, lambda s, t=t, w=w: tag(s, "timeofday") == t and tag(s, "weather") == w))

random.seed(0)
chosen = {}
for name, pred in SLICES:
    pool = [s for s in samples if pred(s)]
    print(f"{name:20} pool {len(pool)}")
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
    tp = C.Counter()
    fp = C.Counter()
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
            tp[cls] += 1
            used.add(bi)
        else:
            fp[cls] += 1
    fn = C.Counter(gc for j, (gc, gb) in enumerate(gts) if j not in used)
    return tp, fp, fn

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

agg = {name: [0, C.Counter(), C.Counter(), C.Counter()] for name, _ in SLICES}
for p, s in chosen.items():
    tp, fp, fn = score(gt_boxes(s), per_image[p])
    for name, pred in SLICES:
        if pred(s):
            a = agg[name]
            a[0] += 1
            a[1].update(tp)
            a[2].update(fp)
            a[3].update(fn)

print("model:", model_name)
print(f"{'slice':20}{'imgs':>6}{'TP':>7}{'FP':>7}{'FN':>7}{'prec':>7}{'recall':>8}")
rows = []
for name, (n, tp, fp, fn) in agg.items():
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    pr = T / (T + F) if T + F else 0
    rc = T / (T + N) if T + N else 0
    print(f"{name:20}{n:>6}{T:>7}{F:>7}{N:>7}{pr:>7.2f}{rc:>8.2f}")
    rows.append([name, n, T, F, N, round(pr, 3), round(rc, 3)])

for name in ["daytime", "night"]:
    n, tp, fp, fn = agg[name]
    print("per class:", name)
    for c in sorted(KEEP):
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0
        print(f"  {c:15}labeled {tp[c] + fn[c]:6}  recall {rc:5.2f}  FP {fp[c]:6}")

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_crossed_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slice", "images", "tp", "fp", "fn", "precision", "recall"])
    w.writerows(rows)
print("saved", out)
