#!/usr/bin/env python3
"""YOLO geometry and uncertainty -> evidence-grounded local Llama inspection notes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
# Historical aggregate results, NOT per-prediction reliability or calibration.
CLASS_RESULTS = {
    "BLOCK ETCH": (0.709, 0.707, 86), "COATING BAD": (0.567, 0.449, 49),
    "PARTICLE": (0.694, 0.850, 72), "PIQ PARTICLE": (0.603, 0.710, 31),
    "PO CONTAMINATION": (0.776, 0.587, 46), "SCRATCH": (0.800, 0.831, 154),
    "SEZ BURNT": (0.682, 0.596, 18),
}
CHECKS = {
    "INSPECT_REGION": "Inspect the reported region in the original optical image; verify the predicted class.",
    "VERIFY_CLASS": "Compare the region against the class definitions before accepting its predicted label.",
    "HIGHER_RESOLUTION": "Inspect a higher-resolution source crop; upscaling alone cannot add missing detail.",
    "CHECK_BOUNDARY": "Inspect beyond the image boundary if possible; the reported box touches an edge.",
    "RESOLVE_OVERLAP": "Check whether overlapping predictions refer to the same physical region.",
    "REVIEW_WEAK_CANDIDATE": "Inspect the low-score candidate before treating it as a real defect.",
    "MANUAL_SCREEN": "Manually inspect the source image; lack of strong detections does not establish a clean wafer.",
}
PROMPT = """You assist a human inspecting optical wafer images, using structured
detector evidence only. You cannot see pixels. Provide a concise inspection plan,
not just a repeated REVIEW/ABSTAIN decision. For each useful priority cite candidate_ids,
select a supplied allowed_check, and explain why its geometry or score makes that
check useful. Mention uncertainty and known class limitations where relevant.
Scores are not probabilities. Box area is NOT defect area or severity. Position is
in the image, not a physical wafer coordinate. Low-score candidates may be noise.
Overlapping classes are competing predictions, not confirmed coexisting defects.
Do not invent causes, physical dimensions, fabrication remedies, visual observations,
or accept/reject conclusions. Do not use historical test recall to claim a specific
prediction is correct. Return JSON only. Keep each explanation to one short sentence.
"""


class Priority(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_ids: list[int] = Field(max_length=2)
    check: str
    why: str = Field(max_length=160)


class Advice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(max_length=160)
    priorities: list[Priority] = Field(max_length=2)
    uncertainty: str = Field(max_length=160)


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def describe_candidate(idx, name, score, box, width, height):
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    cx, cy = (x1+x2)/2, (y1+y2)/2
    horizontal = "left" if cx < 1/3 else "right" if cx > 2/3 else "center"
    vertical = "top" if cy < 1/3 else "bottom" if cy > 2/3 else "middle"
    edge = [side for side, hit in zip(("left", "top", "right", "bottom"),
            (x1 <= .005, y1 <= .005, x2 >= .995, y2 >= .995)) if hit]
    weak = score < .25
    checks = ["INSPECT_REGION", "VERIFY_CLASS"]
    if weak or score < .5:
        checks.append("REVIEW_WEAK_CANDIDATE")
    # Source-pixel heuristic, not optical resolving power.
    if min(bw*width, bh*height) < 32:
        checks.append("HIGHER_RESOLUTION")
    if edge:
        checks.append("CHECK_BOUNDARY")
    precision, recall, instances = CLASS_RESULTS.get(name, (None, None, None))
    return {"id": idx, "class": name, "score": round(score, 4),
            "score_band": "below_display_threshold" if weak else "review_eligible" if score >= .5 else "tentative",
            "box_xyxy_normalized": [round(v, 6) for v in box],
            "box_size_source_pixels": [round(bw*width, 1), round(bh*height, 1)],
            "box_area_percent_of_image": round(100*bw*bh, 2),
            "box_width_height_ratio": round(bw*width/(bh*height), 2) if bh > 0 else None,
            "image_region": f"{vertical}-{horizontal}", "touches_edges": edge,
            "allowed_checks": checks,
            "historical_test_metrics": {"precision": precision, "recall": recall, "instances": instances}}


def build_evidence(result, image, weights_hash):
    height, width = result.orig_shape
    candidates = [describe_candidate(i, result.names[int(b.cls.item())], float(b.conf.item()),
                   [float(v) for v in b.xyxyn[0]], width, height) for i, b in enumerate(result.boxes)]
    overlaps = []
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            overlap = iou(a["box_xyxy_normalized"], b["box_xyxy_normalized"])
            if overlap >= .3:
                overlaps.append({"ids": [a["id"], b["id"]], "box_iou": round(overlap, 3),
                                 "different_classes": a["class"] != b["class"]})
                for c in (a, b):
                    if "RESOLVE_OVERLAP" not in c["allowed_checks"]:
                        c["allowed_checks"].append("RESOLVE_OVERLAP")
    eligible = [c["id"] for c in candidates if c["score"] >= .5]
    return {"schema_version": "inspection-2", "weights_sha256": weights_hash,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "image_dimensions": [width, height], "pixels_supplied_to_llama": False,
            "thresholds": {"candidate": .05, "display": .25, "review": .5,
                           "note": "provisional uncalibrated demo thresholds"},
            "candidate_limit": 20, "candidate_limit_reached": len(candidates) >= 20,
            "counts": {"all_candidates": len(candidates), "displayed": sum(c["score"] >= .25 for c in candidates),
                       "by_class": dict(Counter(c["class"] for c in candidates))},
            "policy": {"decision": "REVIEW" if eligible else "ABSTAIN", "eligible_ids": eligible},
            "candidates": candidates, "overlapping_predictions": overlaps,
            "check_catalog": CHECKS,
            "limitations": ["No image pixels or reference labels provided to Llama",
                "Coordinates and sizes describe enclosing boxes, not measured defect shape or physical size",
                "Historical test metrics are aggregate, not per-image probabilities",
                "BLOCK ETCH / COATING BAD confusion observed in demo; verify class",
                "No clean-image false-positive evaluation; no detection is not proof of clean material",
                "Post-NMS candidates are not full class logits; missing alternatives do not prove certainty"]}


def request_advice(evidence):
    schema = Advice.model_json_schema()
    schema["$defs"]["Priority"]["properties"]["check"]["enum"] = list(CHECKS)
    ids = [c["id"] for c in evidence["candidates"]]
    schema["$defs"]["Priority"]["properties"]["candidate_ids"]["items"]["enum"] = ids or [0]
    if not ids:
        schema["$defs"]["Priority"]["properties"]["candidate_ids"]["maxItems"] = 0
    request = {"model": "llama3.2:1b", "stream": False, "format": schema,
               "messages": [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": json.dumps(evidence)}],
               "options": {"temperature": 0, "num_predict": 420, "num_ctx": 8192}, "keep_alive": "5m"}
    start = time.perf_counter()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        raw = json.load(response)
    content = raw.get("message", {}).get("content", "")
    error = None
    advice = None
    try:
        if raw.get("done_reason") == "length":
            raise ValueError("Response truncated")
        advice = Advice.model_validate_json(content)
        by_id = {c["id"]: c for c in evidence["candidates"]}
        for p in advice.priorities:
            if p.check not in CHECKS or any(i not in by_id for i in p.candidate_ids):
                raise ValueError("Unknown check or candidate ID")
            if p.check != "MANUAL_SCREEN" and (not p.candidate_ids or any(
                    p.check not in by_id[i]["allowed_checks"] for i in p.candidate_ids)):
                raise ValueError("Check not supported by cited evidence")
    except ValueError as exc:
        error = str(exc)
        advice = None
    return {"model": "llama3.2:1b", "elapsed_seconds": round(time.perf_counter()-start, 3),
            "policy": evidence["policy"], "advisory": advice.model_dump() if advice else None,
            "validation_error": error, "raw_response": content,
            "caution": "Structural checks do not validate model prose; human review required",
            "request": request}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "advisory-rich")
    args = parser.parse_args()
    from ultralytics import YOLO
    weights = ROOT / "model" / "best.pt"
    model = YOLO(str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.images or sorted((ROOT / "samples").glob("*.jpg"))
    for image in images:
        result = model.predict(str(image), conf=.05, iou=.7, max_det=20, imgsz=640, device="cpu", verbose=False)[0]
        evidence = build_evidence(result, image, digest)
        (args.output / f"{image.stem}.evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        response = request_advice(evidence)
        (args.output / f"{image.stem}.advisory.json").write_text(json.dumps(response, indent=2) + "\n")
        print(image.name, json.dumps({k: response[k] for k in ("elapsed_seconds", "advisory", "validation_error")}), flush=True)


if __name__ == "__main__":
    main()
