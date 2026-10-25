"""Calibration and operating-point evaluation for the field classifier.

The classifier always emits a percentage. This module answers the two questions
that percentage invites:

* **Is it true?** Reliability bins compare the mean confidence of a bin with the
  accuracy actually observed in it, and ECE is their weighted gap.
* **Where does it stop being wrong?** A threshold sweep reports how much of the
  data is accepted at each confidence, how accurate the accepted rows are, and
  how many wrong answers survive, so an operating point can be chosen from
  measurement rather than taste.

Both are computed on whatever rows are passed in, so the same code scores a
familiar test split and a university the model has never seen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import beta
from torch.utils.data import DataLoader

from .field_model import load_field_checkpoint
from .field_train import FieldRows, collate, load_jsonl, predict_logits

DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995)


def reliability_bins(confidence: np.ndarray, correct: np.ndarray,
                     bins: int = 10) -> list[dict[str, Any]]:
    """Mean confidence against observed accuracy, per equal-width bin."""
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence >= low) & ((confidence < high) if index < bins - 1 else confidence <= high)
        if not mask.any():
            rows.append({"bin": f"{low:.1f}-{high:.1f}", "rows": 0,
                         "mean_confidence": None, "accuracy": None, "gap": None})
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        rows.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "rows": int(mask.sum()),
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "gap": mean_confidence - accuracy,
        })
    return rows


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               bins: int = 10) -> float:
    total = 0.0
    for row in reliability_bins(confidence, correct, bins=bins):
        if row["rows"]:
            total += (row["rows"] / len(confidence)) * abs(row["gap"])
    return float(total)


def error_upper_bound(errors: int, accepted: int, confidence: float = 0.95) -> float | None:
    """Clopper-Pearson upper bound on the error rate among accepted rows.

    A measured precision of 99.5% on 200 rows is not evidence of 99.5% precision;
    this is the largest error rate the data still permits at the given
    confidence. It is what turns "no false positives" into a claim that can fail.
    """
    if accepted <= 0:
        return None
    if errors >= accepted:
        return 1.0
    return float(min(1.0, beta.ppf(confidence, errors + 1, accepted - errors)))


def threshold_sweep(confidence: np.ndarray, correct: np.ndarray,
                    thresholds=DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    """Coverage, precision and surviving errors at each confidence threshold.

    ``false_positive_rate`` counts accepted *wrong* answers over all rows: it is
    the fraction of the whole input that would receive a confident wrong label,
    which is the number a person deciding whether to trust the output cares about.
    """
    total = len(confidence)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        mask = confidence >= threshold
        accepted = int(mask.sum())
        errors = int((~correct & mask).sum())
        bound = error_upper_bound(errors, accepted)
        rows.append({
            "threshold": float(threshold),
            "coverage": accepted / total,
            "accepted_rows": accepted,
            "precision": float(correct[mask].mean()) if accepted else None,
            "abstained_rows": total - accepted,
            "abstention_rate": 1 - accepted / total,
            "false_positive_rate": errors / total,
            "certified_precision_lower": None if bound is None else 1 - bound,
        })
    return rows


def operating_point(sweep: list[dict[str, Any]], target_precision: float) -> dict[str, Any]:
    """Lowest threshold whose accepted rows reach ``target_precision``.

    Lowest, not highest, because a higher threshold buys precision by refusing
    more work; the useful point is the first one that clears the bar. The
    confidence bound travels with it, because observed precision on a few
    hundred rows is not a guarantee.
    """
    for row in sweep:
        if row["precision"] is not None and row["precision"] >= target_precision:
            certified = row["certified_precision_lower"]
            return {**row, "certified": bool(certified is not None
                                             and certified >= target_precision)}
    best = max((row for row in sweep if row["precision"] is not None),
               key=lambda row: row["precision"], default=None)
    return {
        "threshold": None,
        "note": f"no threshold reaches precision {target_precision}",
        "best_observed": None if best is None else {
            "threshold": best["threshold"],
            "precision": best["precision"],
            "coverage": best["coverage"],
            "certified_precision_lower": best["certified_precision_lower"],
        },
    }


def per_label_at_threshold(confidence: np.ndarray, correct: np.ndarray,
                           targets: np.ndarray, labels: list[str],
                           threshold: float) -> list[dict[str, Any]]:
    mask = confidence >= threshold
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        present = targets == index
        accepted = present & mask
        rows.append({
            "label": label,
            "rows": int(present.sum()),
            "accepted": int(accepted.sum()),
            "coverage": float(accepted.sum() / present.sum()) if present.any() else None,
            "precision": float(correct[accepted].mean()) if accepted.any() else None,
        })
    rows.sort(key=lambda row: (row["precision"] is not None, -(row["precision"] or 0)))
    return rows


def predict_arrays(directory: str | Path, rows: list[dict[str, Any]],
                   device: str | None = None) -> dict[str, Any]:
    """Run one checkpoint over rows and return per-row confidence and correctness."""
    model, checkpoint, tokenizer = load_field_checkpoint(directory)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)
    labels = list(checkpoint["labels"])
    index_of = {label: index for index, label in enumerate(labels)}
    usable = [row for row in rows if row["label"] in index_of]
    dataset = FieldRows(usable, labels, tokenizer, int(checkpoint.get("max_length", 192)))
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id or 0),
                        num_workers=0)
    logits, targets = predict_logits(model, loader, device_obj, torch.bfloat16)
    temperature = float(checkpoint.get("temperature", 1.0))
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    confidence, prediction = probs.max(dim=-1)
    confidence = confidence.numpy()
    prediction = prediction.numpy()
    del model
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "labels": labels,
        "temperature": temperature,
        "confidence": confidence,
        "prediction": prediction,
        "targets": targets,
        "correct": prediction == targets,
        "skipped": len(rows) - len(usable),
    }


def summarise(confidence: np.ndarray, correct: np.ndarray, targets: np.ndarray,
              labels: list[str], target_precision: float) -> dict[str, Any]:
    sweep = threshold_sweep(confidence, correct)
    point = operating_point(sweep, target_precision)
    threshold = point["threshold"]
    return {
        "rows": len(correct),
        "labels": len(labels),
        "accuracy": float(correct.mean()),
        "ece_10bin": expected_calibration_error(confidence, correct),
        "mean_confidence": float(confidence.mean()),
        "overconfidence": float(confidence.mean() - correct.mean()),
        "reliability": reliability_bins(confidence, correct),
        "sweep": sweep,
        "target_precision": target_precision,
        "operating_point": point,
        "per_label": per_label_at_threshold(confidence, correct, targets, labels,
                                            threshold if threshold is not None else 1.01),
    }


def evaluate_checkpoint(directory: str | Path, rows: list[dict[str, Any]],
                        target_precision: float = 0.99,
                        device: str | None = None) -> dict[str, Any]:
    arrays = predict_arrays(directory, rows, device=device)
    result = summarise(arrays["confidence"], arrays["correct"], arrays["targets"],
                       arrays["labels"], target_precision)
    result.update({
        "checkpoint": str(directory),
        "skipped_rows_outside_vocabulary": arrays["skipped"],
        "temperature": arrays["temperature"],
    })
    return result


def evaluate_folds(corpus: str | Path, folds_root: str | Path,
                   fold_plan: str | Path, seed: int = 20261019, folds: int = 3,
                   target_precision: float = 0.99,
                   device: str | None = None) -> dict[str, Any]:
    """Calibration on held-out universities, reusing each fold's own checkpoint.

    Pooling matters: the test sets are disjoint, so concatenating them gives a
    single estimate over every real syllabus from a school the model never saw,
    rather than one small fold's opinion.
    """
    from .field_splits import cross_institution_folds

    rows = load_jsonl(corpus)
    plan_path = Path(fold_plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    fold_defs = cross_institution_folds(rows, seed=seed, folds=folds, school_plan=plan)

    confidences: list[np.ndarray] = []
    corrects: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    labels: list[str] | None = None
    per_fold: list[dict[str, Any]] = []
    for fold in fold_defs:
        directory = Path(folds_root) / f"fold-{fold['fold']}"
        if not (directory / "model.pt").exists():
            continue
        arrays = predict_arrays(directory, fold["test"], device=device)
        if labels is None:
            labels = arrays["labels"]
        elif labels != arrays["labels"]:
            raise SystemExit("fold checkpoints disagree on the label vocabulary")
        confidences.append(arrays["confidence"])
        corrects.append(arrays["correct"])
        targets_all.append(arrays["targets"])
        per_fold.append({
            "fold": fold["fold"],
            "holdout_schools": fold["holdout_schools"],
            "rows": len(arrays["correct"]),
            "accuracy": float(arrays["correct"].mean()),
            "mean_confidence": float(arrays["confidence"].mean()),
            "ece_10bin": expected_calibration_error(arrays["confidence"], arrays["correct"]),
        })
    if not confidences:
        raise SystemExit(f"no fold checkpoints under {folds_root}")

    confidence = np.concatenate(confidences)
    correct = np.concatenate(corrects)
    targets = np.concatenate(targets_all)
    pooled = summarise(confidence, correct, targets, labels or [], target_precision)
    return {
        "folds_root": str(folds_root),
        "corpus": str(corpus),
        "per_fold": per_fold,
        "pooled": pooled,
        "rows": pooled["rows"],
    }


def print_report(result: dict[str, Any], target_precision: float) -> None:
    print(f"rows {result['rows']:,} | accuracy {result['accuracy']:.4f} | "
          f"mean confidence {result['mean_confidence']:.4f} | "
          f"ECE {result['ece_10bin']:.4f}")
    print("\nreliability (confidence bin -> observed accuracy)")
    print(f"{'bin':<10}{'rows':>8}{'conf':>8}{'acc':>8}{'gap':>8}")
    for row in result["reliability"]:
        if not row["rows"]:
            continue
        print(f"{row['bin']:<10}{row['rows']:>8,}{row['mean_confidence']:>8.3f}"
              f"{row['accuracy']:>8.3f}{row['gap']:>+8.3f}")
    print("\nthreshold sweep")
    print(f"{'thr':>6}{'coverage':>10}{'precision':>11}{'certified>=':>13}{'abstain':>9}{'FPR':>8}")
    for row in result["sweep"]:
        precision = "-" if row["precision"] is None else f"{row['precision']:.4f}"
        certified = row.get("certified_precision_lower")
        certified_text = "-" if certified is None else f"{certified:.4f}"
        print(f"{row['threshold']:>6.3f}{row['coverage']:>10.3f}{precision:>11}"
              f"{certified_text:>13}{row['abstention_rate']:>9.3f}"
              f"{row['false_positive_rate']:>8.4f}")
    point = result["operating_point"]
    if point.get("threshold") is not None:
        verdict = "certified" if point.get("certified") else "NOT certified by this sample"
        print(f"\noperating point for precision >= {target_precision}: "
              f"threshold {point['threshold']:.3f} -> coverage {point['coverage']:.3f}, "
              f"precision {point['precision']:.4f}, FPR {point['false_positive_rate']:.4f} "
              f"({verdict}) "
              f"[95% lower bound {point['certified_precision_lower']}]")
    else:
        print(f"\nno threshold reaches precision {target_precision}")
        best = point.get("best_observed")
        if best:
            print(f"best observed: threshold {best['threshold']:.3f} -> precision "
                  f"{best['precision']:.4f} at coverage {best['coverage']:.3f}; "
                  f"95% lower bound {best['certified_precision_lower']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure whether the field classifier's percentages are true")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--data", default="data/processed/field/group/test.jsonl",
                        help="JSONL rows with text and label")
    parser.add_argument("--folds-root", default=None,
                        help="Evaluate per-fold checkpoints on their held-out universities")
    parser.add_argument("--fold-plan", default="data/processed/field/fold_plan.json")
    parser.add_argument("--corpus", default="data/processed/field/corpus.jsonl",
                        help="Corpus used to reconstruct the held-out folds")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--label", default=None, help="Name for this evaluation, written to the output")
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="JSON output path (optional)")
    args = parser.parse_args(argv)

    if args.folds_root:
        result = evaluate_folds(args.corpus, args.folds_root, args.fold_plan,
                                seed=args.seed, folds=args.folds,
                                target_precision=args.target_precision,
                                device=args.device)
        result["name"] = args.label or Path(args.folds_root).name
        print(f"== {result['name']} ==")
        print(f"{len(result['per_fold'])} folds, {result['rows']:,} held-out rows")
        for row in result["per_fold"]:
            print(f"  fold {row['fold']}: rows {row['rows']:>4} acc {row['accuracy']:.4f} "
                  f"conf {row['mean_confidence']:.3f} ECE {row['ece_10bin']:.3f}  "
                  f"{', '.join(row['holdout_schools'])}")
        print()
        print_report(result["pooled"], args.target_precision)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"\nwrote {output}")
        return

    rows = load_jsonl(args.data)
    result = evaluate_checkpoint(args.checkpoint, rows,
                                 target_precision=args.target_precision,
                                 device=args.device)
    result["data"] = str(args.data)
    result["name"] = args.label or Path(args.data).stem

    print(f"== {result['name']} ==")
    print(f"checkpoint {result['checkpoint']}")
    print_report(result, args.target_precision)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
