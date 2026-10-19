import cv2
import numpy as np

D = "experiments/results/miss_examples/"
W, H = 640, 360
rows = []
for cat in ["low-conf", "poor-box", "nothing"]:
    tiles = []
    for k in (1, 2, 3):
        canvas = np.zeros((H, W, 3), np.uint8)
        img = cv2.imread(D + "night_car_" + cat + "_" + str(k) + ".jpg")
        if img is not None:
            s = min(W / img.shape[1], H / img.shape[0])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            canvas[:img.shape[0], :img.shape[1]] = img
        cv2.putText(canvas, cat + " " + str(k), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(canvas)
    rows.append(np.hstack(tiles))
cv2.imwrite(D + "contact_sheet.jpg", np.vstack(rows))
print("saved", D + "contact_sheet.jpg")
