"""Score any number of field-classifier checkpoints on exactly the same rows.

Used to compare a new training recipe against a recorded baseline without the
comparison being confounded by which rows each checkpoint can score: the rows
are fixed, and a checkpoint that lacks a label in its vocabulary reports how
many rows it skipped instead of silently scoring a different set.

    python scripts/eval_field_transfer.py \
        --rows data/processed/field_mix/fold-0/test.jsonl \
        --checkpoint artifacts/field-crossinstitution-context/fold-0 \
        --checkpoint artifacts/field-mix/fold-0-pilot \
        --name baseline --name pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.field_calibration import predict_arrays  # noqa: E402
from pace.field_train import load_jsonl  # noqa: E402


def score(checkpoint: str, rows: list[dict], device: str | None = None) -> dict:
    arrays = predict_arrays(checkpoint, rows, device=device)
    labels = arrays["labels"]
    targets, prediction = arrays["targets"], arrays["prediction"]
    correct = arrays["correct"]
    confusion = {}
    for truth, guess in zip(targets, prediction):
        if truth != guess:
            key = f"{labels[int(truth)]} -> {labels[int(guess)]}"
            confusion[key] = confusion.get(key, 0) + 1
    return {
        "checkpoint": str(checkpoint),
        "rows": int(len(correct)),
        "skipped_outside_vocabulary": int(arrays["skipped"]),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, prediction, average="weighted", zero_division=0)),
        "mean_confidence": float(arrays["confidence"].mean()),
        "ece_10bin": float(np.mean(np.abs(arrays["confidence"] - correct))),
        "temperature": arrays["temperature"],
        "top_confusions": sorted(confusion.items(), key=lambda kv: -kv[1])[:12],
        "per_label_f1": {
            label: float((prediction[targets == index] == index).mean())
            for index, label in enumerate(labels)
            if (targets == index).any()
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoints on identical rows")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--name", action="append", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.rows)
    names = args.name or [Path(path).name for path in args.checkpoint]
    results = []
    for name, checkpoint in zip(names, args.checkpoint):
        result = score(checkpoint, rows, device=args.device)
        result["name"] = name
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_label_f1"},
                         indent=2), flush=True)

    print("\n== summary ==")
    print(f"{'name':<28}{'rows':>6}{'acc':>8}{'macroF1':>9}{'conf':>8}{'|conf-acc|':>12}")
    for result in results:
        print(f"{result['name']:<28}{result['rows']:>6}{result['accuracy']:>8.4f}"
              f"{result['macro_f1']:>9.4f}{result['mean_confidence']:>8.4f}"
              f"{result['ece_10bin']:>12.4f}")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": str(args.rows), "results": results},
                                   indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
