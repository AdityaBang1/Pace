"""End-to-end demo: segment real syllabus documents and classify every field.

This is the pipeline a user actually runs: take a document the institution
published, find its sections, and ask the model what each one is. Nothing is
rewritten -- every prediction carries the exact source text it was made from,
so a person can check it against the original.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .field_corpus import clean_html, detect_document_sections
from .field_model import FIELD_TASKS, FieldEngine, classify_section

_INGESTED = Path("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl")


def load_syllabi(path: str | Path = _INGESTED, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record["clean_text"], _ = clean_html(record.get("text") or "")
            rows.append(record)
            if limit and len(rows) >= limit:
                break
    return rows


def analyze_syllabus(record: dict[str, Any], engine: FieldEngine,
                     max_body_chars: int = 400,
                     context_engine: FieldEngine | None = None,
                     threshold: float = 0.8) -> dict[str, Any]:
    """Segment one syllabus and classify each section heading and body.

    With ``context_engine`` the recommended deployment recipe is used: the
    heading is classified first, and the heading+value model only takes over
    when the heading's calibrated confidence is below ``threshold``.
    """
    text = record["clean_text"]
    sections = detect_document_sections(text)
    items: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        body_start = section["end"]
        body_end = sections[index + 1]["start"] if index + 1 < len(sections) else len(text)
        body = re.sub(r"\s+", " ", text[body_start:body_end]).strip()
        heading = section["text"]
        inline = section.get("inline") or ""
        if context_engine is not None:
            decision = classify_section(heading, body, engine, context_engine,
                                        threshold=threshold, max_chars=max_body_chars)
            heading_result = engine.classify(heading, top_k=3)
            predicted_label = decision["label"]
            predicted_task = decision["task"]
            percent = decision["percent"]
        else:
            heading_result = engine.classify(heading, top_k=3)
            decision = None
            predicted_label = heading_result["label"]
            predicted_task = heading_result["task"]
            percent = heading_result["percent"]
        entry = {
            "section_index": index,
            "source_text": heading,
            "predicted_field": predicted_label,
            "predicted_task": predicted_task,
            "percent": percent,
            "alternatives": heading_result["top"][1:],
        }
        if decision is not None:
            entry["decision_source"] = decision["source"]
            entry["fallback"] = decision["fallback"]
        if inline:
            inline_result = engine.classify(inline, top_k=3)
            entry["inline_value"] = inline
            entry["inline_field"] = inline_result["label"]
            entry["inline_percent"] = inline_result["percent"]
        if len(body) >= 40:
            body_result = engine.classify(body[:max_body_chars], top_k=3)
            entry["body_excerpt"] = body[:160]
            entry["body_field"] = body_result["label"]
            entry["body_percent"] = body_result["percent"]
            entry["heading_body_agree"] = body_result["label"] == predicted_label
        items.append(entry)
    covered = sum(1 for item in items if item["predicted_field"] != "other")
    agree = sum(1 for item in items if item.get("heading_body_agree"))
    comparable = sum(1 for item in items if "heading_body_agree" in item)
    return {
        "document_id": record.get("id"),
        "school": record.get("school"),
        "document_title": record.get("title") or record.get("course_title"),
        "sections_found": len(items),
        "sections_classified": covered,
        "heading_body_agreement": (agree / comparable) if comparable else None,
        "sections": items,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify real syllabus documents end to end")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--context-checkpoint", default=None,
                        help="Heading+value model; enables the heading-first cascade")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--ingested", default=str(_INGESTED))
    parser.add_argument("--limit", type=int, default=3, help="How many documents to process")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/field-ood/syllabus_demo.json")
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = FieldEngine(args.checkpoint, device=device)
    context_engine = None
    if args.context_checkpoint and Path(args.context_checkpoint).exists():
        context_engine = FieldEngine(args.context_checkpoint, device=device)
    syllabi = load_syllabi(args.ingested, limit=args.limit)
    documents = [analyze_syllabus(record, engine, context_engine=context_engine,
                                  threshold=args.threshold) for record in syllabi]

    fallbacks = sum(1 for d in documents for s in d["sections"] if s.get("fallback"))
    report = {
        "checkpoint": str(args.checkpoint),
        "context_checkpoint": str(args.context_checkpoint) if context_engine else None,
        "device": device,
        "documents": documents,
        "summary": {
            "documents": len(documents),
            "sections_found": sum(d["sections_found"] for d in documents),
            "sections_classified": sum(d["sections_classified"] for d in documents),
            "heading_body_agreement": (
                sum(d["heading_body_agreement"] * sum(1 for s in d["sections"]
                                                      if "heading_body_agree" in s)
                    for d in documents if d["heading_body_agreement"] is not None)
                / max(1, sum(sum(1 for s in d["sections"] if "heading_body_agree" in s)
                             for d in documents))),
            "fallbacks": fallbacks,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    for document in documents:
        print(f"\n== {document['school']} :: {document['document_title']} "
              f"({document['sections_found']} sections) ==")
        for item in document["sections"]:
            body = f" | body={item.get('body_field')}"
            agree = "" if item.get("heading_body_agree") is None else (
                " agree" if item["heading_body_agree"] else " DISAGREE")
            source = "" if not item.get("fallback") else " [fallback]"
            print(f"  {item['percent']:5.1f}%  {item['predicted_field']:<22}{body}{agree}{source}"
                  f"  <- {item['source_text'][:60]}")


if __name__ == "__main__":
    main()
