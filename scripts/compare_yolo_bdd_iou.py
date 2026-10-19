import json, random, sys, csv, os, collections as C
import cv2
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
IOUS = [0.3, 0.5, 0.7]
PER_CAT = 3
OUTDIR = "experiments/results/miss_examples"

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

def matched(gts, preds, thr):
    used = set()
    for cls, conf, b in sorted(preds, key=lambda t: -t[1]):
        best, bi = 0, None
        for j, (gc, gb) in enumerate(gts):
            if gc != cls or j in used:
                continue
            v = iou(b, gb)
            if v > best:
                best, bi = v, j
        if bi is not None and best >= thr:
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

def save_crop(p, box, preds, cat, k):
    img = cv2.imread(ROOT + p)
    if img is None:
        return False
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    m = int(max(60, 0.6 * max(x2 - x1, y2 - y1)))
    cx1, cy1 = max(0, int(x1) - m), max(0, int(y1) - m)
    cx2, cy2 = min(W, int(x2) + m), min(H, int(y2) + m)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    for cn, c, b in preds:
        if iou(b, box) >= 0.1:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
            cv2.putText(img, cn + " " + format(c, ".2f"), (int(b[0]), max(12, int(b[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    crop = img[cy1:cy2, cx1:cx2]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(OUTDIR + "/night_car_" + cat + "_" + str(k) + ".jpg", crop)
    return True

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

os.makedirs(OUTDIR, exist_ok=True)
stat = C.defaultdict(lambda: [0, 0])
saved = C.Counter()
for p, (t, s) in chosen.items():
    gts = gt_boxes(s)
    preds_all = per_image[p]
    strong = [(cn, c, b) for cn, c, b in preds_all
              if cn in KEEP and c >= CONF and b[3] - b[1] >= MIN_H]
    hits = {thr: matched(gts, strong, thr) for thr in IOUS}
    for j, (cls, box) in enumerate(gts):
        groups = ["all"] + (["car"] if cls == "car" else [])
        for g in groups:
            for thr in IOUS:
                stat[(t, g, thr)][0] += 1
                stat[(t, g, thr)][1] += 1 if j in hits[thr] else 0
        if t == "night" and cls == "car" and box[3] - box[1] >= 80 and j not in hits[0.5]:
            cat = why(cls, box, preds_all)
            if saved[cat] < PER_CAT:
                if save_crop(p, box, preds_all, cat, saved[cat] + 1):
                    saved[cat] += 1

print("model:", model_name, "conf:", CONF, "(recall; a hit needs box overlap >= IoU threshold)")
print(f"{'grp':5}{'time':9}{'IoU0.3':>9}{'IoU0.5':>9}{'IoU0.7':>9}{'n':>8}")
rows = []
for g in ["all", "car"]:
    for t in ["daytime", "night"]:
        n = stat[(t, g, 0.5)][0]
        vals = [stat[(t, g, thr)][1] / n if n else 0 for thr in IOUS]
        print(f"{g:5}{t:9}" + "".join(f"{v:>9.2f}" for v in vals) + f"{n:>8}")
        rows.append([g, t] + [round(v, 3) for v in vals] + [n])
print("saved example crops:", dict(saved), "in", OUTDIR)

out = "experiments/results/bdd_iou_" + model_name.replace(".pt", "") + "_conf" + str(CONF) + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "time", "recall_iou0.3", "recall_iou0.5", "recall_iou0.7", "n"])
    w.writerows(rows)
print("saved", out)
