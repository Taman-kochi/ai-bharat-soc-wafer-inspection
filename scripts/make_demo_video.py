#!/usr/bin/env python3
"""Render a video from actual local YOLO predictions, not a screen recording."""
from pathlib import Path
import json
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
VIDEO = OUT / "wafer_detector_demo.mp4"
REPORT = OUT / "wafer_detector_demo_results.json"
W, H, FPS = 1440, 720, 12
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
small = ImageFont.truetype(FONT_PATH, 20)
medium = ImageFont.truetype(FONT_BOLD, 24)
large = ImageFont.truetype(FONT_BOLD, 40)
white, grey, red, green = "#eef3fa", "#aab9c9", "#ff6262", "#58e59e"
model = YOLO(str(ROOT / "model" / "best.pt"))
paths = sorted((ROOT / "samples").glob("*.jpg"))
if VIDEO.exists() or REPORT.exists():
    raise SystemExit("Output already exists; choose a new name rather than overwrite it.")


def title_screen(title, lines):
    im = Image.new("RGB", (W, H), "#101927")
    d = ImageDraw.Draw(im)
    d.text((100, 150), title, font=large, fill=white)
    for i, line in enumerate(lines):
        d.text((100, 245 + i * 47), line, font=medium, fill=grey)
    return im


def panel(canvas, img, x, heading, boxes, color):
    d = ImageDraw.Draw(canvas)
    d.text((x, 140), heading, font=medium, fill=color if boxes is not None else white)
    panel_x, panel_y, panel_w, panel_h = x, 184, 440, 390
    d.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=6, fill="#253344")
    scale = min(panel_w / img.width, panel_h / img.height)
    nw, nh = round(img.width * scale), round(img.height * scale)
    left = panel_x + (panel_w - nw) // 2
    top = panel_y + (panel_h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.Resampling.LANCZOS), (left, top))
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            d.rectangle((left + x1 * scale, top + y1 * scale,
                         left + x2 * scale, top + y2 * scale), outline=color, width=3)


def text_lines(draw, text, x, y, max_width, fill):
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=small) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, line in enumerate(lines[:3]):
        draw.text((x, y + index * 25), line, font=small, fill=fill)


def render_result(path, result, index):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    predictions = []
    for box in result.boxes:
        cls = int(box.cls.item())
        predictions.append({"class": model.names[cls], "confidence": round(float(box.conf.item()), 4),
                            "xyxy": [round(float(v), 2) for v in box.xyxy[0]]})
    truth = []
    for line in path.with_suffix(".txt").read_text().splitlines():
        c, xc, yc, bw, bh = line.split()
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        truth.append({"class": model.names[int(c)],
                      "xyxy": [(xc-bw/2)*width, (yc-bh/2)*height,
                               (xc+bw/2)*width, (yc+bh/2)*height]})
    canvas = Image.new("RGB", (W, H), "#101927")
    d = ImageDraw.Draw(canvas)
    d.text((38, 35), f"Wafer defect detection  |  sample {index}/{len(paths)}", font=large, fill=white)
    d.text((40, 96), "Actual CPU inference  |  confidence threshold 0.25  |  boxes are not masks", font=small, fill=grey)
    panel(canvas, image, 35, "RAW IMAGE", None, white)
    panel(canvas, image, 500, "PREDICTION", predictions, red)
    panel(canvas, image, 965, "REFERENCE LABEL", truth, green)
    d = ImageDraw.Draw(canvas)
    text_lines(d, path.name.split(".rf.")[0], 35, 596, 440, grey)
    pred_text = ", ".join(f"{p['class']} {p['confidence']:.2f}" for p in predictions) or "No detection >= 0.25"
    text_lines(d, pred_text, 500, 596, 440, red)
    text_lines(d, ", ".join(t["class"] for t in truth) or "No label", 965, 596, 440, green)
    d.text((40, 680), "Reference: annotated boxes, not model output. Low-confidence detections may be omitted.", font=small, fill=grey)
    return canvas, {"image": path.name, "predictions": predictions, "reference": truth}


if len(paths) != 7:
    raise SystemExit(f"Expected seven sample images; found {len(paths)}")
frames = [(title_screen("Wafer defect detector", [
    "Actual inference on this computer's CPU",
    "Seven held-out sample images: raw | prediction | reference",
    "YOLOv8n  |  confidence threshold 0.25  |  no audio",
]), 3)]
records = []
for index, path in enumerate(paths, 1):
    start = time.perf_counter()
    result = model.predict(str(path), conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
    elapsed = time.perf_counter() - start
    frame, record = render_result(path, result, index)
    record["elapsed_seconds"] = round(elapsed, 4)
    records.append(record)
    frames.append((frame, 4))
    print(f"Processed {index}/{len(paths)}: {path.name}: {len(record['predictions'])} detection(s)", flush=True)
frames.append((title_screen("Experimental demonstration", [
    "Predictions may be wrong or missing; no detection does not mean defect-free.",
    "Images: Wafer Defect by Wafer, Roboflow Universe, CC BY 4.0.",
    "Rendered from actual local inference; not a live camera or screen recording.",
]), 4))

cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
       "-pixel_format", "rgb24", "-video_size", f"{W}x{H}", "-framerate", str(FPS),
       "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(VIDEO)]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
try:
    for frame, seconds in frames:
        data = frame.tobytes()
        for _ in range(seconds * FPS):
            proc.stdin.write(data)
finally:
    proc.stdin.close()
if proc.wait() != 0:
    raise SystemExit("Video encoding failed")
REPORT.write_text(json.dumps({"confidence_threshold": 0.25, "samples": records}, indent=2) + "\n")
print(f"Saved {VIDEO} and {REPORT}")
