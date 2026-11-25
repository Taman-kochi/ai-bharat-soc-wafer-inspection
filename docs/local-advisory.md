# Local text-only Llama advisory

Run from the project root with the existing Python environment and a running local Ollama service:

```bash
ollama pull llama3.2:1b
.venv/bin/python scripts/advise.py
```

Or pass one or more image paths after `scripts/advise.py`.

YOLO runs on CPU and writes structured class, confidence and normalized box data.
Llama 3.2 1B receives only this JSON, not pixels or ground-truth labels. Evidence
and responses are saved under `out/advisory/`, including raw model responses and
measured request times. This local prototype uses Ollama, not ExecuTorch.

Python computes REVIEW when at least one detector score is 0.5 or
higher; otherwise ABSTAIN. Llama receives that decision and the evidence to explain it.
The terminal prints the deterministic policy summary; model-generated prose is stored
separately in the advisory JSON. REVIEW requests human inspection, not an automatic
reject. This threshold is provisional and uncalibrated. It will also review
confidently incorrect YOLO classifications; the text model cannot independently
inspect the image or correct the detector.

Malformed responses, unknown evidence IDs and unsupported REVIEW decisions fall
back to ABSTAIN. The explanation remains model-generated and requires review;
structural validation does not prove its factual correctness. No PASS verdict is
produced. Network or runtime errors stop the script rather than fabricate results.

Local testing on seven samples showed that the 1B model makes numeric comparison
errors and sometimes claims the supplied evidence is missing. Consequently the
application computes the decision and factual summary; Llama's explanation remains
experimental. Do not treat the model prose as a validated independent decision.
