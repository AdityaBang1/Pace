"""Calibrated abstention: what does the reader's confidence actually buy off-distribution?

A percentage is only useful if a threshold on it means something. This script
sweeps confidence thresholds on four kinds of data the reader was never tuned on
-- bare values, new syllabus documents, district cells, and a publisher whose
entire corpus was held out of training -- and reports coverage, precision, the
Clopper-Pearson lower bound on that precision, and the operating point that
reaches a target.

    python scripts/measure_abstention.py --checkpoint artifacts/jev-field-v4 \
        --source-folds artifacts/source-holdout-hd --output artifacts/jev-field-v4/abstention.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pace.field_calibration import operating_point, threshold_sweep
from pace.field_train import load_jsonl
from pace.jev_model import JevEngine

THRESHOLDS = [round(0.50 + 0.02 * index, 2) for index in range(25)]


def _score(engine: JevEngine, rows: list[dict], batch_size: int = 256) -> dict[str, Any]:
    texts = [row["text"] for row in rows]
    targets = [row["label"] for row in rows]
    labels = list(engine.labels)
    index = {label: position for position, label in enumerate(labels)}
    keep = [(text, index[target]) for text, target in zip(texts, targets) if target in index]
    predictions: list[int] = []
    confidences: list[float] = []
    for start in range(0, len(keep), batch_size):
        batch = keep[start:start + batch_size]
        for row in engine.probabilities([text for text, _ in batch]):
            ranking = sorted(row.items(), key=lambda item: item[1], reverse=True)
            predictions.append(index[ranking[0][0]])
            confidences.append(ranking[0][1])
    truth = np.asarray([target for _, target in keep], dtype=int)
    guess = np.asarray(predictions, dtype=int)
    confidence = np.asarray(confidences, dtype=float)
    correct = guess == truth
    sweep = threshold_sweep(confidence, correct, THRESHOLDS)
    return {
        "rows": len(truth),
        "skipped_outside_vocabulary": len(rows) - len(keep),
        "accuracy": float(correct.mean()) if len(correct) else None,
        "mean_confidence": float(confidence.mean()) if len(confidence) else None,
        "operating_points": {str(target): operating_point(sweep, target)
                             for target in (0.90, 0.95, 0.99)},
        "sweep": sweep,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Precision-coverage for the field reader")
    parser.add_argument("--checkpoint", default="artifacts/jev-field-v4")
    parser.add_argument("--mix-root", default="data/processed/field_mix")
    parser.add_argument("--syllabus-docs", default="data/processed/transfer/syllabus-docs/test.jsonl")
    parser.add_argument("--sis", default="data/processed/sis/sis_holdout.jsonl")
    parser.add_argument("--source-folds", default="artifacts/source-holdout-hd",
                        help="Leave-one-publisher-out folds; each is scored on its held-out rows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    mix = Path(args.mix_root)
    sets: dict[str, dict[str, Any]] = {}

    engine = JevEngine(args.checkpoint, device=args.device)
    test_rows = load_jsonl(mix / "test.jsonl")
    sets["field_with_header"] = _score(engine, test_rows)
    sets["field_bare_values"] = _score(engine,
                                       [dict(row, text=row["bare_text"]) for row in test_rows
                                        if (row.get("bare_text") or "").strip()])
    if Path(args.syllabus_docs).exists():
        sets["syllabus_documents_unseen"] = _score(engine, load_jsonl(args.syllabus_docs))
    if Path(args.sis).exists():
        sets["sis_cells_holdout"] = _score(engine, load_jsonl(args.sis))

    folds_root = Path(args.source_folds)
    if folds_root.exists():
        pool = sum((load_jsonl(mix / f"{split}.jsonl")
                    for split in ("train", "validation", "test")), [])
        by_source: dict[str, list[dict]] = {}
        for row in pool:
            by_source.setdefault(row["source"], []).append(row)
        for fold_dir in sorted(folds_root.glob("fold-*")):
            if not (fold_dir / "model.pt").exists():
                continue
            report = json.loads((fold_dir / "fold_source.json").read_text(encoding="utf-8")) \
                if (fold_dir / "fold_source.json").exists() else {}
            source = report.get("source")
            if not source:
                # The fold order is fixed in scripts/build_source_holdout.py.
                order = ["anordkvist/gu-course-syllabus", "chaseharmon/MIT_Course_Catalog",
                         "USC/USC-Course-Catalog", "Dev-the-dev91/syllabus-cleaned",
                         "alecwatts/umich-course-descriptions"]
                position = int(fold_dir.name.split("-")[1])
                source = order[position] if position < len(order) else None
            if not source or source not in by_source:
                continue
            held_out = by_source[source]
            fold_engine = JevEngine(fold_dir, device=args.device)
            sets[f"held_out_publisher::{source}"] = _score(fold_engine, held_out)

    summary = {
        "checkpoint": str(args.checkpoint),
        "thresholds": THRESHOLDS,
        "sets": sets,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")

    print(f"{'set':52}{'rows':>8}{'acc':>8}{'cov@.90':>9}{'prec@.90':>10}"
          f"{'cov@.95':>9}{'prec@.95':>10}")
    for name, result in sets.items():
        point90 = result["operating_points"]["0.9"]
        point95 = result["operating_points"]["0.95"]
        def cell(point, key):
            return "-" if point.get(key) is None else f"{point[key]:.3f}"
        print(f"{name[:52]:52}{result['rows']:>8}{cell(result, 'accuracy'):>8}"
              f"{cell(point90, 'coverage'):>9}{cell(point90, 'precision'):>10}"
              f"{cell(point95, 'coverage'):>9}{cell(point95, 'precision'):>10}")


if __name__ == "__main__":
    main()
