"""Does the field vocabulary cover a transcript?

A syllabus is prose with headings, so the field classifier fits it directly. A
transcript is a table: a date, a subject level, a course title, a mark and a
credit value on every row. This audit takes the parser's own evidence spans —
the exact characters each entity was read from — and asks the classifier what
field each one is. Where the classifier has no sensible answer, the vocabulary
is missing a field, which is a fact about the label set rather than the model.

The sample transcripts in this repo are synthetic mocks, so this measures
coverage of the vocabulary, **not** accuracy on real student records.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .field_model import classify_section
from .transcript_extract import parse_transcript

# What a reader would call each parsed entity, in the classifier's vocabulary.
# ``None`` means the vocabulary has no field for it at all.
EXPECTED = {
    "title": "course_title",
    "credit": "credits",
    "mark": "grades",
    "grade_level": "course_level",
    "date": None,
}

# The column header a person reading the page would see above each cell. A
# transcript is a table, so this is the heading the heading-aware model expects.
COLUMN_HEADER = {
    "title": "COURSE TITLE",
    "credit": "CREDIT",
    "mark": "GRADE",
    "grade_level": "LEVEL",
    "date": "DATE",
}


def audit(transcript_text: str, engine, context_engine=None,
          threshold: float = 0.8) -> dict[str, Any]:
    parsed = parse_transcript(transcript_text)
    records = parsed.get("course_records") or []
    per_field: dict[str, dict[str, Any]] = {}
    uncovered: Counter = Counter()

    for record in records:
        for field_name, expected_label in EXPECTED.items():
            evidence = (record.get("evidence") or {}).get(field_name)
            if not evidence or not evidence.get("text"):
                continue
            text = evidence["text"]
            prediction = engine.classify(text, top_k=3)
            entry = per_field.setdefault(field_name, {
                "expected_label": expected_label,
                "column_header": COLUMN_HEADER.get(field_name),
                "rows": 0,
                "agreements": 0,
                "with_header_agreements": 0,
                "with_header_predictions": Counter(),
                "predictions": Counter(),
                "examples": [],
            })
            entry["rows"] += 1
            entry["predictions"][prediction["label"]] += 1
            if expected_label is None:
                uncovered[prediction["label"]] += 1
            elif prediction["label"] == expected_label:
                entry["agreements"] += 1
            if context_engine is not None:
                decision = classify_section(COLUMN_HEADER[field_name], text, engine,
                                            context_engine, threshold=threshold)
                entry["with_header_predictions"][decision["label"]] += 1
                if expected_label is not None and decision["label"] == expected_label:
                    entry["with_header_agreements"] += 1
            if len(entry["examples"]) < 5:
                entry["examples"].append({
                    "text": text,
                    "predicted": prediction["label"],
                    "percent": prediction["percent"],
                })

    summary = {}
    for field_name, entry in per_field.items():
        rows = entry["rows"]
        summary[field_name] = {
            "expected_label": entry["expected_label"],
            "column_header": entry["column_header"],
            "rows": rows,
            "agreement": (entry["agreements"] / rows) if rows else None,
            "with_header_agreement": ((entry["with_header_agreements"] / rows)
                                      if rows and context_engine is not None else None),
            "top_predictions": entry["predictions"].most_common(4),
            "with_header_predictions": (entry["with_header_predictions"].most_common(4)
                                        if context_engine is not None else None),
            "examples": entry["examples"],
        }
    gaps = [name for name, entry in EXPECTED.items() if entry is None]
    return {
        "course_records": len(records),
        "fields": summary,
        "vocabulary_gaps": gaps,
        "gap_predictions": uncovered.most_common(6),
        "note": ("Synthetic mock transcript: this measures vocabulary coverage, "
                 "not accuracy on real student records."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit field-type coverage on a transcript")
    parser.add_argument("path", nargs="?", default="data/samples/mock_transcript_lepid_rowwise.txt")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--context-checkpoint", default=None,
                        help="Enables the column-header test (needs the heading+value model)")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/field-ood/transcript_coverage.json")
    args = parser.parse_args(argv)

    import torch

    from .field_model import FieldEngine

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = FieldEngine(args.checkpoint, device=device)
    context_engine = None
    if args.context_checkpoint and Path(args.context_checkpoint).exists():
        context_engine = FieldEngine(args.context_checkpoint, device=device)
    report = audit(Path(args.path).read_text(encoding="utf-8"), engine,
                   context_engine=context_engine, threshold=args.threshold)
    report["checkpoint"] = str(args.checkpoint)
    report["context_checkpoint"] = str(args.context_checkpoint) if context_engine else None
    report["source"] = str(args.path)

    print(f"== transcript coverage ({report['course_records']} course records) ==")
    header = "with header" if context_engine is not None else ""
    print(f"{'entity':<14}{'expected':<18}{'rows':>6}{'bare':>8}{header:>13}  top predictions")
    for field_name, entry in report["fields"].items():
        agreement = "-" if entry["agreement"] is None else f"{entry['agreement']:.3f}"
        with_header = entry["with_header_agreement"]
        header_cell = "-" if with_header is None else f"{with_header:.3f}"
        expected = entry["expected_label"] or "(no field exists)"
        top = ", ".join(f"{label} {count}" for label, count in entry["top_predictions"][:2])
        print(f"{field_name:<14}{expected:<18}{entry['rows']:>6}{agreement:>8}"
              f"{header_cell:>13}  {top}")
    if report["vocabulary_gaps"]:
        print(f"\nvocabulary gaps: {', '.join(report['vocabulary_gaps'])}")
        print(f"  what the model says instead: {report['gap_predictions']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
