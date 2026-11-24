#!/usr/bin/env python3
"""Comparison cards: raw image | model prediction | ground truth, uncropped, one card per image."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
model = YOLO(str(BASE / "model" / "best.pt"))
NAMES = model.names
SAMPLES = BASE / "samples"
OUT = BASE / "out" / "compare"
OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.05
PANEL_W, PANEL_H = 520, 420


def gt_boxes(stem):
    lp = SAMPLES / (stem + ".txt")
    out = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return out


def make_panel(img, items, color, title):
    """items: list of (label, x0, y0, x1, y1) in normalized coords on the source image."""
    h, w = img.shape[:2]
    s = min(PANEL_W / w, PANEL_H / h)
    dw, dh = int(w * s), int(h * s)
    canvas = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)
    ox, oy = (PANEL_W - dw) // 2, (PANEL_H - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(img, (dw, dh))
    for label, x0, y0, x1, y1 in items:
        px0, py0 = ox + int(x0 * dw), oy + int(y0 * dh)
        px1, py1 = ox + int(x1 * dw), oy + int(y1 * dh)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = py0 - th - 6 if py0 - th - 6 > 4 else py1 + 4
        tx = min(max(2, px0), PANEL_W - tw - 10)
        cv2.rectangle(canvas, (tx, ty), (tx + tw + 6, ty + th + 6), color, -1)
        cv2.putText(canvas, label, (tx + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    bar = np.zeros((28, PANEL_W, 3), np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def card_for(path):
    img = cv2.imread(str(path))
    H, W = img.shape[:2]
    gts = gt_boxes(path.stem)
    r = model.predict(str(path), conf=CONF, imgsz=640, device="cpu", verbose=False)[0]
    preds = []
    for b in r.boxes:
        c = int(b.cls)
        cf = float(b.conf)
        x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
        preds.append((c, cf, x0 / W, y0 / H, x1 / W, y1 / H))

    gt_items = [(NAMES[c], x0, y0, x1, y1) for c, x0, y0, x1, y1 in gts]
    pr_items = [(f"{NAMES[c]} {cf:.2f}", x0, y0, x1, y1) for c, cf, x0, y0, x1, y1 in preds]

    raw = make_panel(img, [], (200, 200, 200), "RAW IMAGE")
    pred_p = make_panel(img, pr_items, (0, 0, 255), f"MODEL PREDICTION (red, conf>={CONF})")
    gt_p = make_panel(img, gt_items, (0, 200, 0), "REFERENCE TRUTH (green)")

    sep = np.full((raw.shape[0], 6, 3), 90, np.uint8)
    row = np.hstack([raw, sep, pred_p, sep, gt_p])

    gt_names = sorted({NAMES[c] for c, *_ in gts})
    pr_desc = ", ".join(f"{NAMES[c]} {cf:.2f}" for c, cf, *_ in preds) or "none"
    head = np.zeros((36, row.shape[1], 3), np.uint8)
    txt = f"{path.stem[:34]}  |  GT: {', '.join(gt_names)}  |  preds: {pr_desc}"
    cv2.putText(head, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([head, row])


ORDER = ["10206", "10076", "10792", "10720"]  # scratch hit, class confusion, particle hit, weak/miss
files = {p.stem[:5]: p for p in sorted(SAMPLES.glob("*.jpg"))}
cards = {}
for key, p in files.items():
    cards[key] = card_for(p)
    cv2.imwrite(str(OUT / f"card_{key}.jpg"), cards[key], [cv2.IMWRITE_JPEG_QUALITY, 90])


def montage(keys, path):
    rows = [cards[k] for k in keys]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    sep = np.full((8, w, 3), 90, np.uint8)
    img = np.vstack([r if i == len(rows) - 1 else np.vstack([r, sep]) for i, r in enumerate(rows)])
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved", path, img.shape)


montage(ORDER[:2], str(BASE / "out" / "compare_a.jpg"))
montage(ORDER[2:], str(BASE / "out" / "compare_b.jpg"))
print("CARDS_DONE")
