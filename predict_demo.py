#!/usr/bin/env python3
"""Run the trained wafer-defect detector locally on sample images and save annotated outputs."""
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
MODEL = BASE / "model" / "best.pt"
SAMPLES = BASE / "samples"
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

model = YOLO(str(MODEL))
paths = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
print(f"Running {len(paths)} images on CPU ...")
results = model.predict(source=[str(p) for p in paths], conf=0.25, imgsz=640,
                        device="cpu", verbose=False)
for r in results:
    dets = [(model.names[int(b.cls)], round(float(b.conf), 3)) for b in r.boxes]
    stem = Path(r.path).stem
    im = r.plot(line_width=2)  # BGR ndarray with boxes/labels/conf drawn
    cv2.imwrite(str(OUT / f"{stem}_pred.jpg"), im)
    print(f"{Path(r.path).name}: {dets if dets else 'no detections'}")
print("PREDICT_DONE")
