# Findings: small YOLO models lose recall at night on BDD100K

Status: Day 2, Phase 2. Four COCO-pretrained models (n and s sizes of two families) and one dataset sample. This is a measured baseline, not a general claim about YOLO.

## Setup
- Models: yolo26n.pt, yolo26s.pt, yolov8n.pt, yolov8s.pt (Ultralytics, no fine-tuning). Sections 1 to 8 use the n models unless stated.
- Data: BDD100K images with time-of-day, weather and scene tags (data/bdd100k/samples.json).
- Classes compared: car, truck, bus, bicycle, motorcycle, person (BDD pedestrian and rider), traffic light. Labeled objects shorter than 20 px are ignored.
- A detection is a hit if the class matches and IoU >= 0.5 (greedy, highest confidence first). Default confidence threshold 0.25.
- Slices are sampled per cell with random seed 0. Both models see identical images, so agreement between them is not independent evidence.
- Scripts in scripts/: compare_yolo_bdd.py (slices), compare_yolo_bdd_crossed.py (time of day x weather, per class, optional conf argument), compare_yolo_bdd_size.py, compare_yolo_bdd_misses.py, compare_yolo_bdd_iou.py, compare_yolo_bdd_bright.py, make_contact_sheet.py.
- Result CSVs are written to experiments/results/ (git-ignored).

## 1. Night recall is lower in every weather type
Crossed slices, 1,332 images (150 sampled per cell), conf 0.25.

| Slice | yolo26n recall | yolov8n recall | yolo26n precision | yolov8n precision |
|---|---|---|---|---|
| daytime | 0.46 | 0.49 | 0.78 | 0.74 |
| night | 0.27 | 0.29 | 0.73 | 0.66 |
| daytime+clear | 0.46 | 0.47 | 0.78 | 0.74 |
| night+clear | 0.28 | 0.30 | 0.73 | 0.67 |
| daytime+rainy | 0.45 | 0.47 | 0.77 | 0.73 |
| night+rainy | 0.25 | 0.26 | 0.70 | 0.63 |
| daytime+snowy | 0.46 | 0.48 | 0.80 | 0.74 |
| night+snowy | 0.28 | 0.28 | 0.75 | 0.66 |
| daytime+overcast | 0.49 | 0.52 | 0.78 | 0.75 |

- night+overcast is excluded (only 18 images exist).
- With weather held fixed, night is 17 to 21 recall points below daytime in both models.
- Most BDD "clear" images are night images (night+clear pool 3,274), which is why plain "clear" looked worse than "overcast".
- In daylight, clear, rainy and snowy are within about 1 point of each other. Overcast is 3 to 5 points higher, unexplained (possibly scene mix).
- Rainy night is 3 to 4 points below clear night in both models. This is a hint only, not a finding.

## 2. The gap depends on the confidence threshold, but never closes
yolo26n, same 1,332 images.

| conf | day recall | night recall | day precision | night precision |
|---|---|---|---|---|
| 0.10 | 0.59 | 0.44 | 0.53 | 0.40 |
| 0.25 | 0.46 | 0.27 | 0.78 | 0.73 |
| 0.50 | 0.30 | 0.15 | 0.92 | 0.92 |

- At conf 0.50 both slices have precision 0.92, and night recall is exactly half of daytime.
- Night detections are less confident, so a strict threshold penalizes night more.

## 3. Per-class recall, day to night (conf 0.25)
| Class | labeled day / night | yolo26n | yolov8n |
|---|---|---|---|
| car | 5742 / 4311 | 0.56 to 0.33 | 0.59 to 0.34 |
| truck | 407 / 125 | 0.33 to 0.16 | 0.32 to 0.13 |
| person | 1077 / 460 | 0.30 to 0.21 | 0.33 to 0.25 |
| traffic light | 1045 / 864 | 0.21 to 0.06 | 0.19 to 0.10 |

- Cars lose the most in absolute points and are about 74% of labeled night objects, so the overall night gap is mostly a car gap.
- Night traffic lights are weakly detected (0.06 in yolo26n at 0.25), but at conf 0.10 yolo26n reaches 0.20, with many false alarms.
- Bus, bicycle and motorcycle have 17 to 38 labeled objects at night and are too few to read.
- yolov8n's lower precision comes mostly from extra false-alarm cars (daytime) and traffic lights (night).

## 4. Object size does not explain it
600 images (300 per time of day). Median labeled box height is 44 px by day and 45 px by night.

| Cars, box height | n day / night | yolo26n day to night | yolov8n day to night |
|---|---|---|---|
| 20-40 px | 1026 / 867 | 0.29 to 0.08 | 0.32 to 0.06 |
| 40-80 px | 748 / 712 | 0.63 to 0.33 | 0.68 to 0.34 |
| 80+ px | 672 / 574 | 0.87 to 0.68 | 0.87 to 0.71 |

- Night recall is lower in every size band. Relative loss grows as objects get smaller.

## 5. What the misses are (large cars, 80+ px, night)
Percent of labeled large night cars, conf 0.25 (detections down to 0.05 inspected).

| Outcome | yolo26n | yolov8n |
|---|---|---|
| hit | 68 | 71 |
| low-conf | 18 | 14 |
| poor-box | 10 | 10 |
| wrong-class | 2 | 1 |
| nothing | 2 | 4 |
| claimed | 0 | 0 |

- low-conf: a same-class detection with IoU >= 0.5 and confidence between 0.05 and 0.25.
- poor-box: a detection overlapping the label with IoU 0.1 to 0.5 (not necessarily the same car).
- nothing: no detection at IoU >= 0.1. This is about 2 to 4% of large night cars.
- Daytime large cars: 87% hit, with misses mostly wrong-class (4 to 6%).
- Most night misses are objects the model locates weakly, not objects it never sees.

## 6. Box fit is not the cause (yolo26n only)
Recall at IoU 0.3 / 0.5 / 0.7: all classes day 0.48 / 0.46 / 0.39, night 0.30 / 0.27 / 0.20. Cars: day 0.58 / 0.55 / 0.48, night 0.35 / 0.32 / 0.24. The day-night gap is the same size at all three thresholds (18 to 19 points all classes, 23 to 24 points cars). Of matched cars, 75% still match at IoU 0.7 at night against 87% by day.

## 7. About half of the large-car gap is darkness of the label region
Large cars (80+ px), hit rate by mean gray level inside the label box.

| Brightness | cars day / night | yolo26n day / night | yolov8n day / night |
|---|---|---|---|
| 0-15 | 0 / 78 | n/a / 44% | n/a / 44% |
| 15-30 | 3 / 146 | 67% / 64% | 67% / 75% |
| 30-50 | 70 / 152 | 94% / 67% | 94% / 68% |
| 50+ | 599 / 198 | 86% / 80% | 86% / 81% |
| all | 672 / 574 | 87% / 68% | 87% / 71% |

- 65% of large night cars have a dark label box (gray below 50), against about 11% by day.
- At matched brightness (50+) the gap is 5 to 7 points. Reweighting night hit rates to the daytime brightness mix gives about 79 to 80% (rough estimate), so about half of the 16 to 19 point gap is brightness mix and 7 to 9 points remain.
- At 30-50 the gap is still 27 points, so brightness alone does not explain it. Mean gray does not capture blur, noise or contrast.
- Only the darkest bin is clearly worst in both models.
- About a fifth of large-night-car misses are confident partial detections (same-class, conf >= 0.25, mostly inside the label, covering at least 20% of it). Counting them as hits narrows the gap to about 12 to 14 points.

## 8. Notes from 13 example crops (yolo26n, not a measurement)
- Some night labels extend well beyond the visible vehicle, which lowers IoU even when the model finds the car.
- Wrong-class cases were vans and SUVs labeled "car" and detected as bus or truck (ordinary class ambiguity, not a night effect).
- A few labeled cars are nearly invisible in the image.

## Not established
- Why night cars at matched brightness are harder (blur and contrast are untested).
- Whether night labels are looser than daytime labels (no daytime crops compared yet).
- Rain at night versus clear at night.
- Anything about models larger than the s variants, other datasets or fine-tuned models.
- Results for yolov8n at other confidence thresholds or IoU thresholds.


## 9. Larger models (s variants) do not close the gap
Crossed slices, same 1,332 images, conf 0.25.

| Model | day recall | night recall | gap (points) | night / day | day precision | night precision |
|---|---|---|---|---|---|---|
| yolo26n | 0.46 | 0.27 | -19 | 0.59 | 0.78 | 0.73 |
| yolo26s | 0.59 | 0.39 | -20 | 0.66 | 0.76 | 0.67 |
| yolov8n | 0.49 | 0.29 | -20 | 0.60 | 0.74 | 0.66 |
| yolov8s | 0.59 | 0.39 | -19 | 0.67 | 0.74 | 0.68 |

- Recall rises by about 12 to 13 points in both slices from n to s. The point gap stays at 19 to 20; the night/day ratio improves from about 0.60 to about 0.66.
- With weather held fixed, night is 17 to 21 points below daytime in both s models (yolo26s: clear -18, rainy -21, snowy -21; yolov8s: clear -17, rainy -21, snowy -20).
- Night+rainy is 3 to 5 points below night+clear in all four models. All four models see the same images, so this is one sample, not four confirmations. Not established.
- Per-class night recall (day to night): car 0.68 to 0.44 (yolo26s) and 0.68 to 0.45 (yolov8s); traffic light 0.35 to 0.20 in both; person 0.46 to 0.31 and 0.43 to 0.31; truck 0.50 to 0.27 and 0.47 to 0.20.
- The s models detect more night traffic lights than the n models at conf 0.25 (0.20 vs 0.06 for yolo26n), with about 350 to 390 false alarms.
- At conf 0.25 the s models are not more precise than the n models (yolo26s: 0.76 day, 0.67 night), so these are different operating points. A matched-precision comparison (as in section 2) has not been done for the s models.

## 10. Large cars: brightness-matched comparison (yolo26n and yolo26s)
Large cars (80+ px), 600 images (300 per time of day), conf 0.25. Brightness is the mean gray level inside the label box.

| Brightness | cars day / night | median gray day / night | yolo26n hit day / night | yolo26s hit day / night |
|---|---|---|---|---|
| <30 | 3 / 224 | 28 / 18 | n/a / 57% | n/a / 69% |
| 30-50 | 70 / 152 | 44 / 39 | 94% / 67% | 97% / 82% |
| 50-80 | 275 / 120 | 65 / 63 | 89% / 78% | 92% / 89% |
| 80+ | 324 / 78 | 101 / 100 | 84% / 83% | 88% / 92% |
| all | 672 / 574 | | 87% / 68% | 90% / 80% |

- The day-night gap shrinks as the label region gets brighter: -27, -11 and -1 points for yolo26n, and -15, -3 and +4 for yolo26s (30-50, 50-80, 80+). The 80+ night cell has 78 cars (standard error about 4 points).
- Reweighting night hit rates to the daytime brightness mix gives about 79% for yolo26n (87% by day) and about 89.5% for yolo26s (90% by day). These are rough and rely on the night hit rate in the 80+ bin. Brightness mix therefore explains nearly all of yolo26s's large-car gap and about 11 of yolo26n's 19 points.
- Large-car gap overall: -19 points for yolo26n, -11 for yolo26s. The overall all-object gap stays at about 20 points for the s models (section 9), so the all-object gap and the large-car gap differ for the s models.
- 39% of large night cars (224 of 574) have a label box darker than 30 mean gray, and only 3 daytime cars are that dark, so nothing can be said about how a daytime car that dark would do.
- Daytime hit rate for yolo26n is not monotonic in brightness (94% at 30-50, 84% at 80+). Not investigated.

Sharpness proxy: raw variance of the Laplacian of the label crop resized to 64x64. A first version normalized by brightness was discarded because it tracked brightness (night scored sharper than day).
- Median proxy value is slightly lower at night than by day within each brightness bin (871 vs 903, 1181 vs 1237, 1406 vs 1608).
- Inside the 50-80 bin the day-night gap does not shrink in the sharper terciles (yolo26n: -8, -11, -14; yolo26s: -4, -3, -2). Cells hold about 40 night cars, so these differences are within noise. Most cells in the 30-50 and 80+ bins have fewer than 30 night cars and are not reported.
- The proxy mixes blur, noise and texture. It shows no sign that sharpness explains the remaining gap. It does not rule blur out.

Not established: what causes the remaining yolo26n gap at matched brightness (about 8 points), and whether yolov8s behaves like yolo26s on large cars.
