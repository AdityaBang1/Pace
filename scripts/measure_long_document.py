"""What actually happens when the document is much longer than the input window.

The reader takes 127 tokens. A real syllabus is 10-40 k characters, so the honest
question is not "does it fit" but "what does the pipeline do with the rest, and how
long does it take". This measures one real long document end to end:

* the whole text truncated to the window (the naive path),
* the section segmentation this repo already has (`detect_document_sections`),
  with each heading and each body window classified separately,
* coverage (how many sections get a non-``other`` label), the number of distinct
  fields found, wall-clock latency and throughput.

    python scripts/measure_long_document.py --checkpoint artifacts/jev-field-v2
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from pace.field_corpus import clean_html
from pace.long_document import split_document


def longest_documents(path: Path, limit: int = 3) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text, _ = clean_html(record.get("text") or "")
            if len(text) < 8000:
                continue
            rows.append({"id": record.get("id"), "school": record.get("school"),
                         "title": record.get("title") or record.get("course_title"),
                         "text": text})
    rows.sort(key=lambda item: -len(item["text"]))
    return rows[:limit]


def analyse(engine, document: dict, max_body: int = 400) -> dict:
    text = document["text"]
    start = time.perf_counter()
    naive = engine.classify(text[:max_body], top_k=1)
    sections = split_document(text)
    found: list[dict] = []
    for section in sections:
        result = engine.classify(section.text[:max_body], top_k=1)
        entry = {"heading": (section.heading or section.text[:80]),
                 "method": section.method, "field": result["label"],
                 "percent": result["percent"]}
        found.append(entry)
    elapsed = time.perf_counter() - start
    covered = sum(1 for item in found if item["field"] != "other")
    agree = [item for item in found if item.get("body_field")]
    return {
        "document_id": document["id"],
        "school": document["school"],
        "title": document["title"],
        "characters": len(text),
        "naive_truncated_read": {"field": naive["label"], "percent": naive["percent"]},
        "sections_found": len(found),
        "sections_classified": covered,
        "coverage": covered / len(found) if found else 0.0,
        "methods": sorted({item["method"] for item in found}),
        "distinct_fields": len({item["field"] for item in found}),
        "heading_body_agreement": None,
        "top_fields": Counter(item["field"] for item in found).most_common(8),
        "seconds": round(elapsed, 2),
        "sections_per_second": round(len(found) / elapsed, 1) if elapsed else None,
        "sections": found[:40],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Long-document behaviour of the reader")
    parser.add_argument("--checkpoint", default="artifacts/jev-field-v2")
    parser.add_argument("--ingested",
                        default="data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl")
    parser.add_argument("--documents", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/transfer-eval/long_document.json")
    args = parser.parse_args(argv)

    from pace.jev_model import JevEngine

    engine = JevEngine(args.checkpoint, device=args.device)
    documents = longest_documents(Path(args.ingested), args.documents)
    results = [analyse(engine, document) for document in documents]
    summary = {
        "checkpoint": args.checkpoint,
        "window_tokens": engine.max_state,
        "documents": [{key: value for key, value in item.items() if key != "sections"}
                      for item in results],
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["documents"], indent=2))


if __name__ == "__main__":
    main()
