import json, random, sys, csv, os
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26s.pt"
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ROOT = "data/bdd100k/"
MIN_H = 20
THRS = [0.05, 0.10, 0.25, 0.40, 0.50]
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
        box = [x*w, y*h, (x+bw)*w, (y+bh)*h]
        if box[3] - box[1] >= MIN_H:
            out.append(box)
    return out

model = YOLO(model_name)
names = model.names
paths = list(chosen)
data = {}
for i in range(0, len(paths), 16):
    batch = paths[i:i+16]
    results = model.predict([ROOT + p for p in batch], conf=min(THRS), verbose=False)
    for p, r in zip(batch, results):
        preds = []
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if names[int(k)] == "car" and b[3] - b[1] >= MIN_H:
                preds.append((c, b))
        t, s = chosen[p]
        data[p] = (t, gt_cars(s), preds)

print("model:", model_name, "| cars only; recall by labeled height; FP per image by predicted height; labels are incomplete so FP is overstated, compare day vs night rather than reading absolute values")
out_rows = []
for thr in THRS:
    print(f"conf {thr}")
    print(f"  {'time':9}{'imgs':>5}{'rec20-40':>9}{'rec40-80':>9}{'rec80+':>8}{'rec all':>8}{'preds':>7}{'TP':>6}{'FP':>6}{'prec%':>7}{'FP/img':>7}{'FP/img by pred band (s/m/l)':>30}")
    for t in ["daytime", "night"]:
        gt_n = {b[0]: 0 for b in BANDS}
        hit = {b[0]: 0 for b in BANDS}
        fp = {b[0]: 0 for b in BANDS}
        imgs = 0
        npred = 0
        for p, (tt, gts, preds) in data.items():
            if tt != t:
                continue
            imgs += 1
            ps = sorted([x for x in preds if x[0] >= thr], key=lambda x: -x[0])
            npred += len(ps)
            used = set()
            for c, b in ps:
                best, bi = 0, None
                for j, gb in enumerate(gts):
                    if j in used:
                        continue
                    v = iou(b, gb)
                    if v > best:
                        best, bi = v, j
                if bi is not None and best >= 0.5:
                    used.add(bi)
                else:
                    fp[band_of(b[3] - b[1])] += 1
            for j, gb in enumerate(gts):
                bd = band_of(gb[3] - gb[1])
                gt_n[bd] += 1
                if j in used:
                    hit[bd] += 1
        tot_gt = sum(gt_n.values())
        tp = sum(hit.values())
        fpt = sum(fp.values())
        rec = {k: 100.0 * hit[k] / gt_n[k] if gt_n[k] else 0.0 for k in gt_n}
        rec_all = 100.0 * tp / tot_gt if tot_gt else 0.0
        prec = 100.0 * tp / npred if npred else 0.0
        fpi = {k: fp[k] / imgs for k in fp}
        fp_str = " / ".join(f"{fpi[b[0]]:.2f}" for b in BANDS)
        print(f"  {t:9}{imgs:>5}{rec['20-40px']:>9.0f}{rec['40-80px']:>9.0f}{rec['80px+']:>8.0f}{rec_all:>8.0f}{npred:>7}{tp:>6}{fpt:>6}{prec:>7.0f}{fpt/imgs:>7.2f}{fp_str:>30}")
        out_rows.append([thr, t, imgs, round(rec['20-40px'], 1), round(rec['40-80px'], 1), round(rec['80px+'], 1), round(rec_all, 1), npred, tp, fpt, round(prec, 1), round(fpt/imgs, 3)])

os.makedirs("experiments/results", exist_ok=True)
out = "experiments/results/bdd_car_fp_" + model_name.replace(".pt", "") + ".csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["conf", "time", "images", "rec_20_40", "rec_40_80", "rec_80p", "rec_all", "preds", "tp", "fp", "precision_pct", "fp_per_image"])
    w.writerows(out_rows)
print("saved", out)
