"""Compare checkpoints on identical rows, class by class.

Two models scored on different test sets cannot be compared: a change in the
data and a change in the model look the same. This runs every checkpoint over
the same rows and reports per-class F1 next to the support that produced it,
plus accuracy restricted to the long tail — the classes a balanced corpus hides.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .field_calibration import predict_arrays
from .field_train import load_jsonl

TAIL_MAX_ROWS = 200


def per_class(arrays: dict[str, Any]) -> dict[str, Any]:
    labels = arrays["labels"]
    precision, recall, f1, support = precision_recall_fscore_support(
        arrays["targets"], arrays["prediction"], labels=list(range(len(labels))),
        zero_division=0)
    table = {
        label: {
            "support": int(support[index]),
            "f1": float(f1[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
        }
        for index, label in enumerate(labels)
    }
    correct = arrays["correct"]
    tail = np.array([table[labels[target]]["support"] for target in arrays["targets"]]) <= TAIL_MAX_ROWS
    return {
        "rows": int(len(correct)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(arrays["targets"], arrays["prediction"], average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(arrays["targets"], arrays["prediction"], average="weighted",
                                      zero_division=0)),
        "tail_rows": int(tail.sum()),
        "tail_accuracy": float(correct[tail].mean()) if tail.any() else None,
        "per_class": table,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-class comparison of field checkpoints")
    parser.add_argument("--data", default="data/processed/field_uncapped/group/test.jsonl")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name=path; repeat for each model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/field-calibration/per_class_comparison.json")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.data)
    results: dict[str, Any] = {"data": str(args.data), "rows": len(rows), "models": {}}
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).name, spec
        print(json.dumps({"evaluating": name, "checkpoint": path}), flush=True)
        arrays = predict_arrays(path, rows, device=args.device)
        results["models"][name] = {"checkpoint": path, **per_class(arrays)}

    # Print the tail first: it is the part a balanced evaluation hides.
    names = list(results["models"])
    tail_labels = sorted(
        {label for model in results["models"].values() for label, entry in model["per_class"].items()
         if entry["support"] <= TAIL_MAX_ROWS},
        key=lambda label: results["models"][names[0]]["per_class"][label]["support"])
    print(f"\n{'class':<24}{'rows':>7}" + "".join(f"{name[:18]:>20}" for name in names))
    print(f"{'':<24}{'':>7}" + "".join(f"{'F1':>20}" for _ in names))
    for label in tail_labels:
        cells = []
        for name in names:
            entry = results["models"][name]["per_class"][label]
            cells.append(f"{entry['f1']:.3f}")
        support = results["models"][names[0]]["per_class"][label]["support"]
        print(f"{label:<24}{support:>7}" + "".join(f"{cell:>20}" for cell in cells))

    print(f"\n{'model':<26}{'acc':>8}{'macroF1':>9}{'tail acc':>10}{'tail rows':>11}")
    for name in names:
        model = results["models"][name]
        tail_acc = "-" if model["tail_accuracy"] is None else f"{model['tail_accuracy']:.3f}"
        print(f"{name:<26}{model['accuracy']:>8.4f}{model['macro_f1']:>9.4f}"
              f"{tail_acc:>10}{model['tail_rows']:>11,}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
