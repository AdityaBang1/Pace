"""Header reliance: the same rows scored with and without their column header.

Every training row is written ``"<column>: <value>"``, and the column often names
the field. A checkpoint can therefore score ~0.998 accuracy while reading almost
nothing but the prefix. This script measures both numbers on the same rows, and
per source, so a checkpoint that only reads headers cannot hide behind one
IID figure.

    python scripts/measure_header_reliance.py --checkpoint artifacts/jev-field-v3 \
        --output artifacts/jev-field-v3/header_reliance.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pace.field_train import load_jsonl
from pace.jev_eval import evaluate_rows
from pace.jev_model import JevEngine


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Header vs bare-value scoring")
    parser.add_argument("--checkpoint", default="artifacts/jev-field-v3")
    parser.add_argument("--rows", default="data/processed/field_mix/test.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.rows)
    bare_rows = [dict(row, text=row["bare_text"]) for row in rows
                 if (row.get("bare_text") or "").strip()]
    engine = JevEngine(args.checkpoint, device=args.device)
    with_header = evaluate_rows(engine, rows)
    bare = evaluate_rows(engine, bare_rows)

    per_source = {}
    by_source: dict[str, list[dict]] = {}
    for row in bare_rows:
        by_source.setdefault(row["source"], []).append(row)
    for source, items in sorted(by_source.items()):
        if len(items) < 50:
            per_source[source] = {"rows": len(items), "skipped": "fewer than 50 rows"}
            continue
        scored = evaluate_rows(engine, items)
        per_source[source] = {"rows": len(items), "accuracy": scored["accuracy"],
                              "macro_f1": scored["macro_f1"]}

    report = {
        "checkpoint": str(args.checkpoint),
        "rows": len(rows),
        "bare_rows": len(bare_rows),
        "with_header": {"accuracy": with_header["accuracy"],
                        "macro_f1": with_header["macro_f1"],
                        "ece_10bin": with_header["ece_10bin"]},
        "bare_value": {"accuracy": bare["accuracy"], "macro_f1": bare["macro_f1"],
                       "ece_10bin": bare["ece_10bin"]},
        "accuracy_drop_without_header": with_header["accuracy"] - bare["accuracy"],
        "macro_f1_drop_without_header": with_header["macro_f1"] - bare["macro_f1"],
        "per_source_bare": per_source,
        "top_confusions_bare": bare["top_confusions"][:15],
        "label_name_in_text_share": round(
            sum(1 for row in rows
                if row["label"].replace("_", " ") in row["text"].lower()) / max(len(rows), 1), 4),
        "sources": dict(Counter(row["source"] for row in rows)),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
