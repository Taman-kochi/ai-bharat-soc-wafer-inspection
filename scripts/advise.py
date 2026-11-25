#!/usr/bin/env python3
"""Local YOLO evidence -> text-only Llama advisory (no reference labels)."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
MODEL = "llama3.2:1b"
SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["REVIEW", "ABSTAIN"]},
        "reason": {"type": "string"},
        "detection_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["decision", "reason", "detection_ids"],
    "additionalProperties": False,
}


def advise(evidence):
    eligible = [d["id"] for d in evidence["detections"] if d["confidence"] >= 0.5]
    policy_decision = "REVIEW" if eligible else "ABSTAIN"
    if eligible:
        instruction = "Request human review of these detector predictions: " + "; ".join(
            f"{d['class']} at confidence {d['confidence']}" for d in evidence["detections"] if d["id"] in eligible)
    elif evidence["detections"]:
        instruction = "There ARE detections, but all are below the provisional 0.5 review threshold: " + "; ".join(
            f"{d['class']} at confidence {d['confidence']}" for d in evidence["detections"])
    else:
        instruction = "There are no detections. This does not establish a clean wafer."
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["decision"]["enum"] = [policy_decision]
    schema["properties"]["detection_ids"] = {"type": "array", "items": {"type": "integer", "enum": eligible or [0]},
                                               "minItems": len(eligible), "maxItems": len(eligible)}
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({"model": MODEL, "stream": False, "format": schema,
                         "messages": [{"role": "system", "content":
                             "Return JSON. Copy policy_decision and eligible_detection_ids exactly. "
                             "For reason, restate the supplied instruction in one sentence. "
                             "Do not infer causes, visual details, severity, or PASS/FAIL."},
                                      {"role": "user", "content": json.dumps({"detections": evidence["detections"],
                                          "policy_decision": policy_decision, "eligible_detection_ids": eligible,
                                          "instruction": instruction})}],
                         "options": {"temperature": 0, "num_predict": 180, "num_ctx": 2048},
                         "keep_alive": "5m"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.load(response)
    elapsed = time.perf_counter() - start
    text = raw.get("message", {}).get("content", "")
    valid_ids = {d["id"] for d in evidence["detections"]}
    try:
        answer = json.loads(text)
        assert set(answer) == {"decision", "reason", "detection_ids"}
        assert answer["decision"] in ("REVIEW", "ABSTAIN")
        assert answer["decision"] == policy_decision
        assert isinstance(answer["reason"], str) and answer["reason"].strip()
        assert isinstance(answer["detection_ids"], list)
        assert all(type(i) is int and i in valid_ids for i in answer["detection_ids"])
        if answer["decision"] == "REVIEW":
            assert answer["detection_ids"]
            assert any(d["confidence"] >= 0.5 and d["id"] in answer["detection_ids"]
                       for d in evidence["detections"])
        assert raw.get("done_reason") != "length"
        validation = "passed structural and evidence-reference checks; prose is not independently verified"
    except (ValueError, TypeError, AssertionError):
        answer = {"decision": "ABSTAIN", "reason": "Model response failed validation.", "detection_ids": []}
        validation = "failed; application fallback applied"
    # Keep the experimental model prose separate from the factual policy summary.
    # Small models can hallucinate even when their JSON obeys the schema.
    return {"model": MODEL, "runtime": "local Ollama", "elapsed_seconds": round(elapsed, 3),
            "policy_decision": policy_decision,
            "policy_summary": instruction, "decision_source": "application threshold rule",
            "advisory": answer, "validation": validation, "raw_response": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="Images; defaults to the seven demo samples")
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "advisory")
    args = parser.parse_args()
    from ultralytics import YOLO
    weights = ROOT / "model" / "best.pt"
    detector = YOLO(str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.images or sorted((ROOT / "samples").glob("*.jpg"))
    for image in images:
        result = detector.predict(str(image), conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
        detections = []
        for idx, box in enumerate(result.boxes):
            detections.append({"id": idx, "class": detector.names[int(box.cls.item())],
                               "confidence": round(float(box.conf.item()), 4),
                               "box_xyxy_normalized": [round(float(x), 6) for x in box.xyxyn[0]]})
        evidence = {"schema_version": "demo-1", "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "width": result.orig_shape[1], "height": result.orig_shape[0],
                    "weights_sha256": digest, "detector_threshold": 0.25,
                    "detections": detections, "limitations": ["No image pixels supplied to Llama",
                    "Predictions may be wrong; no detection does not establish clean material",
                    "No ground-truth labels supplied; scores are not calibrated probabilities"]}
        (args.output / f"{image.stem}.evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        decision = advise(evidence)
        (args.output / f"{image.stem}.advisory.json").write_text(json.dumps(decision, indent=2) + "\n")
        print(f"{image.name}: {decision['policy_decision']} — {decision['policy_summary']} "
              f"({decision['elapsed_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
