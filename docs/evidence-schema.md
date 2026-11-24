# evidence.json — the vision → advisory interface

Status: **draft for review** (2026-11-23). The advisory pipeline receives only this
document — never pixels — and must abstain when the evidence is insufficient.

## Schema

```json
{
  "schema_version": "0.1",
  "generated_at": "2026-11-23T13:30:00+05:30",
  "image": {
    "id": "10206_image3",
    "width": 742,
    "height": 576,
    "source": "file | capture | monitor_replay",
    "sha256": "<hex>"
  },
  "model": {
    "name": "yolov8n",
    "weights_sha256": "d6600582ec07d96dc7fc25e1e123098c647c0d85823fd42f32bb77a76d7198b1",
    "task": "detect",
    "imgsz": 640,
    "conf_threshold": 0.25,
    "iou_threshold": 0.7,
    "runtime": "ultralytics 8.4.216 / torch 2.15.0+cpu",
    "latency_ms": 312.0
  },
  "detections": [
    {
      "class_id": 5,
      "class_name": "SCRATCH",
      "confidence": 0.82,
      "box_xyxy_px": [12, 0, 731, 576],
      "box_xyxy_norm": [0.016, 0.0, 0.985, 1.0],
      "area_frac": 0.97
    }
  ],
  "counts_by_class": { "SCRATCH": 1 },
  "summary": {
    "n_detections": 1,
    "max_confidence": 0.82,
    "min_confidence": 0.82,
    "dominant_class": "SCRATCH",
    "total_area_frac": 0.97
  },
  "abstain": {
    "value": false,
    "reason": null,
    "reasons_vocabulary": [
      "NO_DETECTIONS",
      "LOW_CONFIDENCE",
      "WEAK_CLASS",
      "AMBIGUOUS_CLASS_PAIR",
      "TINY_AREA"
    ]
  },
  "limitations": [
    "single-image, single-model evidence",
    "no root-cause attribution (dataset has no cause labels)",
    "class confusion possible: BLOCK ETCH <-> COATING BAD"
  ]
}
```

## Abstain rules (proposed — advisory layer decides how to phrase)

| Reason | Trigger | Rationale |
|---|---|---|
| `NO_DETECTIONS` | `n_detections == 0` | nothing to advise on |
| `LOW_CONFIDENCE` | `max_confidence < 0.45` | below this the model is mostly noise on hard classes |
| `WEAK_CLASS` | dominant class in {COATING BAD (R 0.449), PO CONTAMINATION (R 0.587), SEZ BURNT (R 0.596)} | per-class test recall below ~0.65 |
| `AMBIGUOUS_CLASS_PAIR` | top-2 classes are BLOCK ETCH and COATING BAD | observed real confusion |
| `TINY_AREA` | every detection has `area_frac < 0.005` | small detections on 640 px inputs are unreliable |

Evidence found at low confidence (like the 0.06 SEZ BURNT sample) should still be
*surfaced with its confidence*, not silently dropped — but the advisory must say
"low confidence" and not prescribe.

## Rules of the interface

1. No raw pixels, no image paths accessible to the advisory process — only this JSON.
2. The advisory never re-scores or invents detections; it may only cite them.
3. Every advisory sentence must trace to a detection field or a limitation entry.
4. Class names and ordering are fixed by the model (see `class_id` map below).
5. Version both sides: `schema_version` here, and the `.pte` build id on the advisory side.

## Class map (fixed)

| id | name | test recall |
|---:|---|---:|
| 0 | BLOCK ETCH | 0.707 |
| 1 | COATING BAD | 0.449 |
| 2 | PARTICLE | 0.850 |
| 3 | PIQ PARTICLE | 0.710 |
| 4 | PO CONTAMINATION | 0.587 |
| 5 | SCRATCH | 0.831 |
| 6 | SEZ BURNT | 0.596 |

## Open questions

- Include a crop thumbnail (base64) per detection? Rule 1 forbids pixels reaching
  the advisory — keep it out; the human demo view can show crops separately.
- Threshold 0.45 for `LOW_CONFIDENCE` is a guess grounded in the recall table; tune
  against the camera-replay test once it runs.
