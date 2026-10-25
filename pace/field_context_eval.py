"""Does the heading, or the value, decide the field?

For every real syllabus in the corpus, the section heading itself says which
field the section holds. That gives a pseudo-ground truth that does not depend
on the model at all: the heading is matched against a fixed vocabulary of real
section names written by universities.

Three inputs are then compared against that truth:

``heading``
    just the heading line, e.g. "Office Hours"
``body``
    just the text under it, e.g. "Wednesdays 3-4 PM"
``heading + body``
    the heading and the text together, e.g. "Office Hours: Wednesdays 3-4 PM"

If the body alone is much weaker than the heading, then the transferable signal
is the *context* a value sits in, and a field extractor should read the
document structure rather than classify loose values.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .field_corpus import clean_html, detect_document_sections
from .field_model import FieldEngine

SYLLABUS_FILES = [
    "data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl",
    "data/raw/hf/INT-Business-Syllabus/documents.jsonl",
]


def collect_sections(limit: int | None = None) -> list[dict[str, Any]]:
    """Every labelled section of every real syllabus, with heading and body."""
    sections: list[dict[str, Any]] = []
    for path in SYLLABUS_FILES:
        source_path = Path(path)
        if not source_path.exists():
            continue
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text, _ = clean_html(record.get("text") or "")
                found = detect_document_sections(text)
                for position, section in enumerate(found):
                    if section["label"] is None:
                        continue
                    body_start = section["end"]
                    body_end = (found[position + 1]["start"]
                                if position + 1 < len(found) else len(text))
                    body = re.sub(r"\s+", " ", text[body_start:body_end]).strip()
                    inline = section.get("inline") or ""
                    if inline:
                        body = f"{inline} {body}".strip()
                    if len(body) < 40:
                        continue
                    sections.append({
                        "document_id": str(record.get("id") or ""),
                        "school": record.get("school"),
                        "heading": section["text"],
                        "body": body[:600],
                        "truth": section["label"],
                    })
                if limit and len(sections) >= limit:
                    return sections
    return sections


def evaluate(engine: FieldEngine, sections: list[dict[str, Any]], variant: str,
             max_body_chars: int = 400) -> dict[str, Any]:
    """Score one input variant against the heading-derived truth."""
    texts = []
    for section in sections:
        if variant == "heading":
            texts.append(section["heading"])
        elif variant == "body":
            texts.append(section["body"][:max_body_chars])
        else:
            texts.append(f"{section['heading']}: {section['body'][:max_body_chars]}")
    predicted = []
    for row in engine.classify_many(texts):
        predicted.append(row["label"])
    truth = [section["truth"] for section in sections]
    per_label = defaultdict(list)
    for actual, predicted_label in zip(truth, predicted):
        per_label[actual].append(predicted_label == actual)
    return {
        "variant": variant,
        "rows": len(sections),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "per_label_accuracy": {label: float(np.mean(hits)) for label, hits in
                               sorted(per_label.items())},
        "per_label_support": {label: len(hits) for label, hits in sorted(per_label.items())},
    }


def cascade(bare: FieldEngine, context: FieldEngine, sections: list[dict[str, Any]],
            threshold: float) -> dict[str, Any]:
    """Classify the heading; fall back to heading+body when it is unsure.

    The heading carries the field identity, so it is the right first guess.
    When the heading is unfamiliar the model says so with a low calibrated
    percentage, and the context model -- which reads the heading together with
    the text under it -- takes over.
    """
    headings = [section["heading"] for section in sections]
    combined = [f"{section['heading']}: {section['body'][:400]}" for section in sections]
    heading_rows = bare.classify_many(headings)
    context_rows = context.classify_many(combined)
    truth = [section["truth"] for section in sections]
    predictions = []
    fallbacks = 0
    for heading_row, context_row, actual in zip(heading_rows, context_rows, truth):
        if heading_row["confidence"] >= threshold:
            predictions.append(heading_row["label"])
        else:
            predictions.append(context_row["label"])
            fallbacks += 1
    return {
        "variant": f"heading-first (threshold {threshold})",
        "threshold": threshold,
        "rows": len(sections),
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / max(1, len(sections)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Heading vs body vs heading+body")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--context-checkpoint", default="artifacts/field-classifier-context")
    parser.add_argument("--label", default="bare-value model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="artifacts/field-ood/context_eval.json")
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = FieldEngine(args.checkpoint, device=device)
    sections = collect_sections(args.limit)
    results = [evaluate(engine, sections, variant)
               for variant in ("heading", "body", "heading + body")]

    cascades: list[dict[str, Any]] = []
    context_engine = None
    if args.context_checkpoint and Path(args.context_checkpoint).exists():
        context_engine = FieldEngine(args.context_checkpoint, device=device)
        for threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
            cascades.append(cascade(engine, context_engine, sections, threshold))

    report = {
        "model": args.label,
        "checkpoint": str(args.checkpoint),
        "context_checkpoint": str(args.context_checkpoint) if context_engine else None,
        "device": device,
        "documents": len({section["document_id"] for section in sections}),
        "sections": len(sections),
        "results": results,
        "cascade": cascades,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "cascade"}},
                     indent=2))
    for result in results:
        print(f"  {result['variant']:<16} rows={result['rows']:<6} "
              f"accuracy={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}")
    for entry in cascades:
        print(f"  {entry['variant']:<34} fallback={entry['fallback_rate']:.1%} "
              f"accuracy={entry['accuracy']:.4f}  macro_f1={entry['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
