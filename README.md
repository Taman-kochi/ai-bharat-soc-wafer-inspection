# Wafer defect inspection

A small YOLOv8n detector for optical wafer-surface images, plus a local
text-only Llama advisory prototype that explains structured detector evidence.
Everything needed to run the demo is included: weights, samples, scripts and
short demo videos. Datasets are **not** included; sources and provenance are
listed below.

## Demo videos

- `docs/media/wafer_detector_demo.mp4` — raw image, prediction and reference label side by side.
- `docs/media/yolo_llama_advisory_demo.mp4` — evidence JSON, system prompt and Llama responses.
- `docs/media/yolo_llama_rich_evidence_demo.mp4` — richer evidence, with model errors shown honestly.

## Quick start

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

python predict_demo.py            # boxes on the seven sample images -> out/
python scripts/compare_cards.py   # raw | prediction | reference cards -> out/compare/
```

Advisory pipeline (requires a local [Ollama](https://ollama.com) service):

```bash
ollama pull llama3.2:1b
python scripts/advise.py                 # threshold policy + Llama explanation
python scripts/rich_advisory.py          # richer geometry/uncertainty evidence
```

Outputs are written under `out/` and are git-ignored.

## Repository layout

```
model/best.pt           YOLOv8n weights (6.5 MB); sha256 d6600582...d7198b1
model/best.onnx         ONNX export, opset 17, input 1x3x640x640 (12.2 MB)
samples/                seven test-split images, one per class, with reference YOLO labels
predict_demo.py         CPU inference demo
scripts/compare_cards.py    raw/prediction/reference comparison cards
scripts/advise.py           YOLO evidence -> threshold policy + Llama explanation
scripts/rich_advisory.py    enriched evidence + schema-constrained Llama advice
scripts/make_*_video.py     render the demo videos from saved outputs
docs/PROVENANCE.json        primary dataset record and split statistics
docs/PROVENANCE-secondary.json  secondary dataset record
docs/evidence-schema.md     vision -> advisory interface draft
docs/local-advisory.md      advisory pipeline notes and tested limitations
docs/figures/, docs/media/  comparison figures and demo videos
```

## Model card (v1, 2026-11-22)

| Item | Value |
|---|---|
| Task | Detect 7 wafer-surface defect classes |
| Architecture | YOLOv8n, 3.13 M parameters, 8.6 GFLOPs, 640 px |
| Training | 100 epochs, batch 16, AMP, seed 42, 25 min on an RTX 3070 |
| Dataset | Roboflow Universe "Wafer Defect" (v2-final), CC BY 4.0, 4,531 images |
| Labels | Converted from four-corner OBB export to enclosing axis-aligned boxes |
| Split | Publisher split: train 3,172 / valid 907 / test 452 (no group IDs) |

### Test-split metrics (single evaluation run)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.690 | 0.676 | 0.687 | 0.428 |
| BLOCK ETCH | 0.709 | 0.707 | 0.734 | 0.447 |
| COATING BAD | 0.567 | 0.449 | 0.519 | 0.395 |
| PARTICLE | 0.694 | 0.850 | 0.737 | 0.433 |
| PIQ PARTICLE | 0.603 | 0.710 | 0.729 | 0.459 |
| PO CONTAMINATION | 0.776 | 0.587 | 0.668 | 0.432 |
| SCRATCH | 0.800 | 0.831 | 0.853 | 0.530 |
| SEZ BURNT | 0.682 | 0.596 | 0.569 | 0.298 |

Inference: 1.0–1.7 ms/image on an RTX 3070; about 0.3 s/image on this CPU.

## Advisory pipeline findings

- The pipeline passes only structured detection data to Llama (no pixels, no reference labels).
- REVIEW/ABSTAIN is computed by Python with a provisional, uncalibrated 0.5 threshold.
- Llama 3.2 1B explains the decision. In testing it produced generic advice,
  invented a "geometry score" and truncated sentences, so its prose is kept
  separate from the deterministic policy summary and is not trusted as a decision.
- Richer evidence (box size, position, edge contact, overlaps, historical class
  metrics) was added and tested; the interface improved, but the 1B model's
  reasoning remained the bottleneck. A stronger text model is the next experiment.

## Limitations

- Experimental results only; not proof of fab-scale or physical-wafer camera performance.
- No group IDs, so lot-independent or cross-fab generalisation is not claimed.
- The dataset contains no empty label files, so false positives on clean images were not measured.
  A missing detection does **not** establish a clean wafer.
- COATING BAD and SEZ BURNT are weak classes (SEZ BURNT has 18 test instances). Predictions can be
  confidently wrong; observed BLOCK ETCH / COATING BAD confusion.
- `docs/evidence-schema.md` is an unimplemented design draft; its thresholds are illustrative.
- The advisory model is not trained for semiconductor inspection and its explanations require review.

## Data, attribution and licensing

The included images and labels are from
[Wafer Defect by Wafer, Roboflow Universe](https://universe.roboflow.com/wafer-irhuv/wafer-defect-rv1vx),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Labels were converted from
oriented boxes to enclosing axis-aligned boxes; comparison images are annotated derivatives.
Version, checksum and split counts are in `docs/PROVENANCE.json`.

Datasets are not redistributed in this repository. Obtain them from:

- Primary: <https://universe.roboflow.com/wafer-irhuv/wafer-defect-rv1vx> (v2-final, 4,531 images, CC BY 4.0)
- Secondary: <https://universe.roboflow.com/wafer-semiconductor/wafer-defect-detection-tuvq3>
  (v10, 783 images, classes `crack`/`scratch`, CC BY 4.0; see `docs/PROVENANCE-secondary.json`)

A compiled third-party set ([deeptech2026-Quadbits](https://github.com/SanskaarUndale21/deeptech2026-Quadbits))
was evaluated and **excluded**: it declares no license, contains classification folders without boxes,
and a perceptual-hash scan found many candidate overlaps with the primary dataset, including its test split.
These are candidate matches, not independently verified duplicates.

The demo uses Ultralytics; its separate licensing terms apply. No source-code license is specified here.
