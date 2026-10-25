"""Freeze the Python answers the mobile engine has to reproduce.

Two kinds of evidence live here, because they fail differently:

* **token ids** -- a tokenizer that is one token off changes what the model reads,
  and the label can still look plausible. Exact ids, no tolerance.
* **section predictions** -- label and probability for every fixture heading, so
  an ONNX/temperature/label-order mistake shows up as a number, not as a vague
  "the app feels wrong".

Written to ``fixtures/parity.json`` and used by ``tests/mobile/run.ts``.
"""
from __future__ import annotations

import json
from pathlib import Path

from pace.api import DEFAULT_CHECKPOINT, confidence_band
from pace.cli import classify_engine
from pace.field_extract import type_sections
from pace.field_corpus import clean_html
from pace.field_extract import detect_document_sections
from pace.model import MAX_LENGTH
from transformers import AutoTokenizer

FIXTURES = Path("fixtures")

TEXTS = [
    "MATH 1A: Calculus I",
    "AP Chemistry",
    "Grading Policy",
    "Tests 40%",
    "TU TH 11:00-12:15 in Evans 10",
    "Office Hours: Mondays 3-5pm in Room 214",
    "Dr. Allison Johnson",
    "CHEM 1A. General Chemistry. 4 units.",
    "Fall 2026 - Section 003",
    "Prerequisite: MATH 1A.",
    "Students may miss two lectures without penalty.",
    "Café résumé naïve",
    "AP Chem1stry — J0hnson 4O%",
    "   ",
    "",
    "!!!",
    "MATH101",
    "MATH-1010",
    "Biology vs AP Biology",
    "x" * 400,
]


def main() -> None:
    checkpoint = Path(DEFAULT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"),
                                             local_files_only=True)
    engine = classify_engine(str(checkpoint), device="cpu")
    context = classify_engine("artifacts/field-classifier-context", device="cpu")
    max_length = int(getattr(engine, "max_length", MAX_LENGTH))

    tokens = []
    for text in TEXTS:
        encoded = tokenizer(text, truncation=True, max_length=max_length)
        tokens.append({"text": text, "ids": encoded["input_ids"]})

    sections = []
    for name in sorted(FIXTURES.glob("*.txt")):
        cleaned, _ = clean_html(name.read_text(encoding="utf-8"))
        detected = detect_document_sections(cleaned)
        # The app ships both models and uses `classify_section`'s fallback, so the
        # parity target is the fused decision, not the heading-only one.
        typed = type_sections(cleaned, engine, context)
        sections.append({
            "fixture": name.name,
            "headings": [{"heading": item["text"], "label": item["label"],
                          "confidence": round(float(item["confidence"]), 6),
                          "band": confidence_band(float(item["confidence"]))}
                         for item in typed],
            "detected": len(detected),
        })

    payload = {
        "checkpoint": str(checkpoint),
        "max_length": max_length,
        "temperature": float(getattr(engine, "temperature", 1.0)),
        "labels": list(engine.labels),
        "tokens": tokens,
        "sections": sections,
        "note": ("sections use the same heading -> heading+value fallback the app "
                 "implements (threshold 0.8), so label and probability are directly "
                 "comparable"),
    }
    target = FIXTURES / "parity.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(target), "tokens": len(tokens),
                      "fixtures": len(sections), "max_length": max_length}))


if __name__ == "__main__":
    main()
