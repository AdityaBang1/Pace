"""Evaluate the transcript path end to end: model type, model entities, parser records.

The reference spans for the entity comparison are produced by the deterministic
transcript parser, which is a separate code path from the neural tagger: every span
it reports is copied from the source text with character offsets. The comparison
therefore measures how well the neural tagger agrees with a rule-based reading of the
same document, not how well it agrees with itself.

The record-level numbers (rows, credits, sections, GPA) come from the parser and are
reported as checks against what the document itself states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .transcript_extract import parse_transcript, summarize

PARSER_ENTITY_TYPES = (
    "STUDENT", "SCHOOL", "SUBJECT_AREA", "DATE", "GRADE_LEVEL",
    "COURSE", "MARK", "CREDIT", "SCORE", "PERFORMANCE_LEVEL", "GPA",
)


def gold_entities_from_parser(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the parser's evidence offsets into a typed span reference set."""
    entities: list[dict[str, Any]] = []

    def add(entity_type: str, span: dict[str, Any] | None) -> None:
        if not span or span.get("start") is None or span.get("end") is None:
            return
        if span["start"] >= span["end"]:
            return
        entities.append({"type": entity_type, "start": int(span["start"]), "end": int(span["end"])})

    header = parsed.get("header", {})
    add("STUDENT", header.get("student_name"))
    add("SCHOOL", header.get("school_name"))
    for section in parsed.get("sections", []):
        add("SUBJECT_AREA", section)
    for record in parsed.get("course_records", []):
        evidence = record["evidence"]
        add("DATE", evidence.get("date"))
        add("GRADE_LEVEL", evidence.get("grade_level"))
        add("COURSE", evidence.get("title"))
        add("MARK", evidence.get("mark"))
        add("CREDIT", evidence.get("credit"))
    for record in parsed.get("assessment_records", []):
        evidence = record["evidence"]
        add("DATE", evidence.get("date"))
        add("GRADE_LEVEL", evidence.get("grade_level"))
        add("COURSE", evidence.get("title"))
        add("SCORE", evidence.get("score"))
        # The performance level value is the level plus its label; the field label
        # ("Performance Level:") is a label, not the entity.
        add("PERFORMANCE_LEVEL", evidence.get("performance_level_value"))
    standing = parsed.get("academic_standing", {})
    add("DATE", standing.get("as_of"))
    add("CREDIT", standing.get("total_credits"))
    add("GPA", standing.get("gpa"))
    return entities


def span_scores(gold: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact-span precision, recall and F1, overall and per entity type."""
    gold_set = {(item["type"], item["start"], item["end"]) for item in gold}
    predicted_set = {(item["type"], item["start"], item["end"]) for item in predicted}
    true_positive = len(gold_set & predicted_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)
    per_type: dict[str, Any] = {}
    for entity_type in sorted({item["type"] for item in gold} | {item["type"] for item in predicted}):
        gold_type = {item for item in gold_set if item[0] == entity_type}
        predicted_type = {item for item in predicted_set if item[0] == entity_type}
        hits = len(gold_type & predicted_type)
        per_type[entity_type] = {
            "support": len(gold_type),
            "predicted": len(predicted_type),
            "true_positive": hits,
            "precision": hits / len(predicted_type) if predicted_type else 0.0,
            "recall": hits / len(gold_type) if gold_type else 0.0,
            "f1": (2 * hits / (len(gold_type) + len(predicted_type))) if (gold_type or predicted_type) else 0.0,
        }
    return {
        "gold_spans": len(gold_set),
        "predicted_spans": len(predicted_set),
        "true_positive_spans": true_positive,
        "false_positive_spans": false_positive,
        "false_negative_spans": false_negative,
        "precision": true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0,
        "recall": true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0,
        "f1": (2 * true_positive / (2 * true_positive + false_positive + false_negative))
        if (true_positive + false_positive + false_negative) else 0.0,
        "per_entity_type": per_type,
    }


def evaluate_transcript(text: str, checkpoint: str | Path = "artifacts/synthetic-multitask",
                        device: str | None = None) -> dict[str, Any]:
    """Run the neural transcript path and the deterministic parser over one document."""
    from .synthetic_doc_model import SyntheticDocumentEngine

    engine = SyntheticDocumentEngine(checkpoint, device=device)
    result = engine.analyze(text)
    parsed = parse_transcript(text)

    model_entities = [
        {"type": entity["type"], "start": entity["start"], "end": entity["end"]}
        for entity in result.get("entities", [])
        if entity.get("type") in PARSER_ENTITY_TYPES
    ]
    gold = gold_entities_from_parser(parsed)
    document_type = result.get("document_type", {})
    transcript_block = result.get("transcript", {})
    summary = summarize(parsed)
    counts: dict[str, int] = {}
    for entity in result.get("entities", []):
        counts[entity["type"]] = counts.get(entity["type"], 0) + 1
    examples: dict[str, list[str]] = {}
    for entity in result.get("entities", []):
        bucket = examples.setdefault(entity["type"], [])
        if len(bucket) < 4:
            bucket.append(entity.get("source_text", entity.get("text", "")))
    scores = span_scores(gold, model_entities)
    return {
        "checkpoint": str(checkpoint),
        "model_scope": result.get("model_scope"),
        "document_type": {
            "label": document_type.get("label"),
            "confidence": document_type.get("confidence"),
            "probabilities": document_type.get("probabilities"),
        },
        "transcript_triggered_by": transcript_block.get("triggered_by"),
        "model_entity_scores": scores,
        "model_entity_counts_by_type": dict(sorted(counts.items())),
        "model_entity_examples": examples,
        "parser_summary": summary,
        "parser_checks": parsed.get("checks", {}),
        "model_entity_count": len(result.get("entities", [])),
        "headline": {
            "document_type_label": document_type.get("label"),
            "entity_span_f1": scores["f1"],
            "entity_span_precision": scores["precision"],
            "entity_span_recall": scores["recall"],
            "course_records": summary.get("course_records"),
            "assessment_records": summary.get("assessment_records"),
            "subject_areas": len(summary.get("sections", [])),
            "computed_total_credits": summary.get("computed_total_credits"),
            "reported_total_credits": summary.get("reported_total_credits"),
            "credit_sum_matches_reported_total": summary.get("credit_sum_matches_reported_total"),
            "gpa": (summary.get("gpa") or {}).get("value") if summary.get("gpa") else None,
        },
        "note": (
            "Entity reference spans come from the deterministic parser (independent of the "
            "neural tagger). Record-level results are the parser's, with the document's own "
            "reported totals used as the consistency check."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the transcript path on one document.")
    parser.add_argument("path")
    parser.add_argument("--checkpoint", default="artifacts/synthetic-multitask")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    text = Path(args.path).read_text(encoding="utf-8")
    report = evaluate_transcript(text, args.checkpoint, args.device)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
