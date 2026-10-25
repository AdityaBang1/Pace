"""Score a trained Jev checkpoint on rows of real text, one row per line.

This is the same protocol the MiniLM/bge classifiers were scored with
(``pace.field_train.evaluate_classifier``): accuracy, macro/weighted F1, the
10-bin expected calibration error, per-class F1 and the top confusions. The
difference is that the model being scored is the 10 M decision model, so the
numbers are directly comparable to ``artifacts/field-classifier-mix``.

    python -m pace.jev_eval --checkpoint artifacts/jev-field \
        --rows data/processed/field_mix/test.jsonl \
        --output artifacts/jev-field/test_metrics.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .field_train import load_jsonl
from .jev_model import JevEngine


def evaluate_rows(engine: JevEngine, rows: list[dict[str, Any]], batch_size: int = 256
                  ) -> dict[str, Any]:
    """Accuracy, F1, calibration and confusions for one set of labelled rows."""
    texts = [row["text"] for row in rows]
    targets = [row["label"] for row in rows]
    labels = list(engine.labels)
    index = {label: position for position, label in enumerate(labels)}
    kept = [(text, index[target]) for text, target in zip(texts, targets) if target in index]
    skipped = len(texts) - len(kept)
    predictions: list[int] = []
    confidences: list[float] = []
    for start in range(0, len(kept), batch_size):
        batch = kept[start:start + batch_size]
        probabilities = engine.probabilities([text for text, _ in batch])
        for row in probabilities:
            ranking = sorted(row.items(), key=lambda item: item[1], reverse=True)
            predictions.append(index[ranking[0][0]])
            confidences.append(ranking[0][1])
    targets_array = np.asarray([target for _, target in kept], dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    confidence_array = np.asarray(confidences, dtype=float)
    correct = prediction_array == targets_array
    ece = 0.0
    bins = 10
    for position in range(bins):
        low, high = position / bins, (position + 1) / bins
        mask = (confidence_array >= low) & (
            (confidence_array < high) if position < bins - 1 else (confidence_array <= high))
        if mask.any():
            ece += float(mask.mean() * abs(confidence_array[mask].mean() - correct[mask].mean()))
    present = sorted(set(targets_array.tolist()))
    report = classification_report(targets_array, prediction_array,
                                   labels=present, output_dict=True, zero_division=0)
    confusions = Counter((labels[target], labels[predicted])
                         for target, predicted in zip(targets_array, prediction_array)
                         if target != predicted)
    per_label = {label: float(f1_score(targets_array == index[label],
                                       prediction_array == index[label], zero_division=0))
                 for label in labels if any(target == index[label] for target in targets_array)}
    support = Counter(labels[target] for target in targets_array)
    model_name = (engine.checkpoint.get("base_model")
                  or engine.checkpoint.get("stage") or "jev-small")
    return {
        "checkpoint": str(engine.checkpoint.get("output", "")),
        "model": model_name,
        "rows": int(len(targets_array)),
        "skipped_outside_vocabulary": skipped,
        "accuracy": float(accuracy_score(targets_array, prediction_array)),
        "macro_f1": float(f1_score(targets_array, prediction_array, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(targets_array, prediction_array, average="weighted",
                                      zero_division=0)),
        "mean_confidence": float(confidence_array.mean()),
        "ece_10bin": ece,
        "temperature": float(engine.temperature),
        "per_class_f1": per_label,
        "per_class_support": {label: int(support[label]) for label in per_label},
        "top_confusions": [[f"{real} -> {guessed}", count]
                           for (real, guessed), count in confusions.most_common(20)],
        "classification_report": {key: value for key, value in report.items()
                                  if key in present or key in {"accuracy", "macro avg",
                                                               "weighted avg"}},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Jev-shaped field checkpoint")
    parser.add_argument("--checkpoint", default="artifacts/jev-field")
    parser.add_argument("--rows", nargs="+", default=["data/processed/field_mix/test.jsonl"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", default=None,
                        help="An encoder classifier (bge/MiniLM) to score on the same rows, "
                             "so the two numbers are directly comparable")
    args = parser.parse_args(argv)

    engine = JevEngine(args.checkpoint, device=args.device)
    results = []
    for path in args.rows:
        rows = load_jsonl(path)
        result = evaluate_rows(engine, rows, batch_size=args.batch_size)
        result["rows_file"] = path
        results.append(result)
        print(json.dumps({"rows": path, "rows_scored": result["rows"],
                          "accuracy": result["accuracy"],
                          "macro_f1": result["macro_f1"],
                          "weighted_f1": result["weighted_f1"],
                          "ece_10bin": result["ece_10bin"]}), flush=True)
    if args.baseline:
        from .field_model import FieldEngine

        baseline = FieldEngine(args.baseline, device=args.device)
        for path, ours in zip(args.rows, results):
            theirs = evaluate_rows(baseline, load_jsonl(path), batch_size=args.batch_size)
            print(json.dumps({"rows": path, "baseline": baseline.checkpoint.get("base_model"),
                              "baseline_accuracy": theirs["accuracy"],
                              "jev_accuracy": ours["accuracy"],
                              "baseline_macro_f1": theirs["macro_f1"],
                              "jev_macro_f1": ours["macro_f1"],
                              "delta_accuracy": ours["accuracy"] - theirs["accuracy"],
                              "delta_macro_f1": ours["macro_f1"] - theirs["macro_f1"]}),
                  flush=True)
            ours["baseline"] = {key: theirs[key] for key in
                                ("model", "accuracy", "macro_f1", "weighted_f1",
                                 "ece_10bin")}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = results[0] if len(results) == 1 else {"results": results}
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
