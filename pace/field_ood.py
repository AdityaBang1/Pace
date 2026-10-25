"""Out-of-distribution evaluation for the real-text field classifier.

Two levels of honesty:

``per_source``
    The model trained on every source, evaluated separately on the held-out
    groups of each source. Each source's test rows are documents the model has
    never seen, but the model has seen the other sources' layouts.

``leave_one_source_out``
    A whole dataset is removed from training and then evaluated. This is the
    strict test of whether the model learned the field or one institution's
    column layout: the target schema is genuinely unseen.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .field_model import FIELD_TASKS
from .field_train import (FieldRows, LengthGroupedBatches, collate, evaluate_logits,
                          fit_temperature, load_jsonl, oversample_rare, predict_logits,
                          train_one)

SOURCE_NAMES = {
    "USC": "USC course catalog",
    "anordkvist": "Gothenburg syllabus fields",
    "Dev-the-dev91": "multi-institution syllabi",
    "twangodev": "UW-Madison course guide",
    "chaseharmon": "MIT course catalog",
    "syllabus-finetune": "span-annotated syllabi",
    "iristun": "CU-DocBench (not a field value)",
    "alecwatts": "UMich course pages",
    "INT-Business-Syllabus": "UF business syllabi",
}


def _source_of(row: dict[str, Any]) -> str:
    return str(row.get("source_key") or str(row.get("source", "")).split("/")[0])


def per_source_evaluation(checkpoint_dir: str | Path, test_rows: list[dict[str, Any]],
                          output: Path) -> dict[str, Any]:
    """Evaluate one trained checkpoint on the held-out groups of each source."""
    from .field_model import load_field_checkpoint

    model, checkpoint, tokenizer = load_field_checkpoint(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labels = list(checkpoint["labels"])
    pad_id = tokenizer.pad_token_id or 0
    temperature = float(checkpoint.get("temperature", 1.0))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        grouped[_source_of(row)].append(row)

    results: dict[str, Any] = {}
    for source, rows in sorted(grouped.items()):
        dataset = FieldRows(rows, labels, tokenizer, int(checkpoint.get("max_length", 192)))
        loader = DataLoader(dataset, batch_size=128, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id), num_workers=0)
        logits, targets = predict_logits(model, loader, device, torch.bfloat16)
        metrics = evaluate_logits(logits, targets, labels, temperature)
        metrics.pop("_confidence")
        metrics.pop("_prediction")
        results[source] = {
            "source_name": SOURCE_NAMES.get(source, source),
            "rows": len(rows),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class_f1": metrics["per_class_f1"],
            "per_class_support": metrics["per_class_support"],
            "top_confusions": metrics["top_confusions"],
        }
    return results


def leave_one_source_out(corpus: list[dict[str, Any]], source: str, output_root: Path,
                         base_model: str, epochs: int, seed: int = 20261019) -> dict[str, Any]:
    """Train without one source, then evaluate on that source."""
    train_pool = [row for row in corpus if _source_of(row) != source]
    target = [row for row in corpus if _source_of(row) == source]
    if not train_pool or not target:
        raise ValueError(f"source {source!r} has no train or no target rows")

    groups = sorted({row["group_id"] for row in train_pool})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    cut = max(1, int(len(groups) * 0.9))
    validation_groups = {groups[i] for i in order[:cut]}
    train_rows = [row for row in train_pool if row["group_id"] not in validation_groups]
    validation_rows = [row for row in train_pool if row["group_id"] in validation_groups]
    if not validation_rows:
        validation_rows = train_rows[: max(1, len(train_rows) // 20)]

    output = output_root / f"holdout-{source}"
    result = train_one(seed, train_rows, validation_rows, target, output,
                       base_model=base_model, epochs=epochs, verbose=True)
    evaluation = evaluate_rows(output, target, Counter(result["train_label_counts"]))
    return {
        "held_out_source": source,
        "held_out_source_name": SOURCE_NAMES.get(source, source),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "target_rows": len(target),
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "weighted_f1": evaluation["weighted_f1"],
        "support_decomposition": evaluation["support_decomposition"],
        "per_class_f1": evaluation["per_class_f1"],
        "per_class_support": evaluation["per_class_support"],
        "top_confusions": evaluation["top_confusions"],
        "labels": result["labels"],
    }


def _support_decomposition(logits: torch.Tensor, targets: np.ndarray,
                           labels: list[str], train_counts: Counter,
                           temperature: float) -> dict[str, Any]:
    """Split the target rows by whether their label had any training support.

    Holding out a source can leave a field type with *zero* training examples,
    which no model could predict. Reporting one blended number hides that, so
    the report separates the rows the model could in principle learn from the
    rows it could not.
    """
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    prediction = probs.argmax(dim=-1).numpy()
    supported = np.array([train_counts.get(labels[int(t)], 0) > 0 for t in targets])
    correct = (prediction == targets)
    out: dict[str, Any] = {}
    for name, mask in (("supported", supported), ("unsupported", ~supported)):
        count = int(mask.sum())
        out[name] = {
            "rows": count,
            "accuracy": float(correct[mask].mean()) if count else None,
        }
    unseen_labels = sorted({labels[int(t)] for t in targets if not train_counts.get(labels[int(t)], 0)})
    out["labels_without_training_support"] = unseen_labels
    out["unseen_label_rows"] = int(sum(train_counts.get(label, 0) == 0 for label in unseen_labels))
    return out


def evaluate_rows(directory: str | Path, rows: list[dict[str, Any]],
                  train_counts: Counter) -> dict[str, Any]:
    """Evaluate one checkpoint on rows, with the label-support decomposition."""
    from .field_model import load_field_checkpoint

    model, checkpoint, tokenizer = load_field_checkpoint(directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labels = list(checkpoint["labels"])
    pad_id = tokenizer.pad_token_id or 0
    dataset = FieldRows(rows, labels, tokenizer, int(checkpoint.get("max_length", 192)))
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id), num_workers=0)
    logits, targets = predict_logits(model, loader, device, torch.bfloat16)
    temperature = float(checkpoint.get("temperature", 1.0))
    metrics = evaluate_logits(logits, targets, labels, temperature)
    metrics.pop("_confidence")
    metrics.pop("_prediction")
    result = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "per_class_support": metrics["per_class_support"],
        "top_confusions": metrics["top_confusions"],
        "support_decomposition": _support_decomposition(logits, targets, labels,
                                                       train_counts, temperature),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def analyze_holdouts(output_root: Path, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-evaluate every saved holdout checkpoint with the support decomposition."""
    results: list[dict[str, Any]] = []
    for directory in sorted(output_root.glob("holdout-*")):
        source = directory.name[len("holdout-"):]
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_counts = Counter(saved.get("train_label_counts") or {})
        target = [row for row in corpus if _source_of(row) == source]
        if not target:
            continue
        print(json.dumps({"analyzing_holdout": source}), flush=True)
        evaluation = evaluate_rows(directory, target, train_counts)
        results.append({
            "held_out_source": source,
            "held_out_source_name": SOURCE_NAMES.get(source, source),
            "target_rows": len(target),
            "train_rows": saved.get("train_rows"),
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "weighted_f1": evaluation["weighted_f1"],
            "support_decomposition": evaluation["support_decomposition"],
            "per_class_f1": evaluation["per_class_f1"],
            "per_class_support": evaluation["per_class_support"],
            "top_confusions": evaluation["top_confusions"],
        })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Out-of-distribution field classifier evaluation")
    parser.add_argument("--corpus", default="data/processed/field/corpus.jsonl")
    parser.add_argument("--test", default="data/processed/field/group/test.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--output", default="artifacts/field-ood")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--holdout-sources", nargs="*", default=[
        "USC", "anordkvist", "twangodev", "Dev-the-dev91"])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Re-evaluate saved holdout checkpoints without retraining")
    args = parser.parse_args(argv)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    test_rows = load_jsonl(args.test)

    report: dict[str, Any] = {"per_source": per_source_evaluation(args.checkpoint, test_rows,
                                                                   output_root)}
    corpus = load_jsonl(args.corpus)
    if args.analyze_only:
        report["leave_one_source_out"] = analyze_holdouts(output_root, corpus)
    elif not args.skip_training:
        report["leave_one_source_out"] = []
        for source in args.holdout_sources:
            print(json.dumps({"leave_one_source_out": source}), flush=True)
            report["leave_one_source_out"].append(
                leave_one_source_out(corpus, source, output_root, args.base_model, args.epochs))
    (output_root / "ood_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({
        "per_source": {k: {"rows": v["rows"], "accuracy": round(v["accuracy"], 4),
                           "macro_f1": round(v["macro_f1"], 4)}
                       for k, v in report["per_source"].items()},
        "leave_one_source_out": [
            {"held_out_source": r["held_out_source"], "rows": r["target_rows"],
             "accuracy": round(r["accuracy"], 4), "macro_f1": round(r["macro_f1"], 4),
             "supported_accuracy": (round(r["support_decomposition"]["supported"]["accuracy"], 4)
                                    if r["support_decomposition"]["supported"]["accuracy"]
                                    is not None else None),
             "unsupported_rows": r["support_decomposition"]["unsupported"]["rows"]}
            for r in report.get("leave_one_source_out", [])],
    }, indent=2))


if __name__ == "__main__":
    main()
