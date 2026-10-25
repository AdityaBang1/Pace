"""Strict exact-text-disjoint evaluation of USC catalog field models.

Subject-held-out rows can still share identical cell text across courses. This evaluator
removes normalized strings seen in train/validation and scores each remaining test string
once, giving a more honest estimate of generalization to novel cell values.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoTokenizer

from .model import MAX_LENGTH, PaceClassifier
from .real_data import normalize_value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strict_novel_text_test(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_keys = {normalize_value(row["text"]) for row in train_rows}
    validation_keys = {normalize_value(row["text"]) for row in validation_rows}
    prior_keys = train_keys | validation_keys

    candidates = [row for row in test_rows if normalize_value(row["text"]) not in prior_keys]
    # One test instance per exact normalized text; label conflicts should have been
    # removed before this task, but fail loudly if that invariant is violated.
    by_text: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = normalize_value(row["text"])
        previous = by_text.get(key)
        if previous is not None and previous["label"] != row["label"]:
            raise ValueError(f"Conflicting labels remain for exact text {row['text']!r}")
        by_text.setdefault(key, row)
    selected = list(by_text.values())
    selected.sort(key=lambda row: (row.get("group_id", ""), normalize_value(row["text"])))

    counts = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows_original": len(test_rows),
        "unique_texts_in_train": len(train_keys),
        "unique_texts_in_validation": len(validation_keys),
        "test_rows_seen_in_train_or_validation_removed": len(test_rows) - len(candidates),
        "duplicate_test_rows_removed": len(candidates) - len(selected),
        "strict_test_unique_novel_text_rows": len(selected),
        "strict_test_subjects": len({row.get("group_id") for row in selected}),
    }
    return selected, counts


def _bootstrap_subject_accuracy(
    rows: list[dict[str, Any]],
    predictions: list[str],
    iterations: int = 2000,
    seed: int = 20261019,
) -> dict[str, float]:
    by_subject: dict[str, list[bool]] = defaultdict(list)
    for row, pred in zip(rows, predictions):
        by_subject[str(row.get("group_id", "unknown"))].append(pred == row["label"])
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        return {"subjects": len(subjects), "accuracy_ci95_low": 0.0, "accuracy_ci95_high": 1.0}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        chosen = [rng.choice(subjects) for _ in subjects]
        outcomes = [ok for subject in chosen for ok in by_subject[subject]]
        samples.append(sum(outcomes) / len(outcomes))
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return {
        "subjects": len(subjects),
        "accuracy_ci95_low": float(low),
        "accuracy_ci95_high": float(high),
    }


def _evaluate_sparse(rows: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    model = joblib.load(model_path)
    texts = [row["text"] for row in rows]
    return {"predictions": model.predict(texts).tolist(), "model": "char_tfidf_logistic"}


def _evaluate_neural(rows: list[dict[str, Any]], model_dir: Path, batch_size: int = 128) -> dict[str, Any]:
    checkpoint_path = model_dir / "model.pt"
    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(meta["labels"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / meta["tokenizer_dir"]), local_files_only=True)
    model = PaceClassifier(
        meta["base_model"],
        config_dir=str(model_dir / meta["encoder_config"]),
        num_classes=len(labels),
    )
    model.load_state_dict(meta["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                padding="max_length",
                truncation=True,
                max_length=int(meta.get("max_length", MAX_LENGTH)),
                return_tensors="pt",
            )
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            predictions.extend(labels[int(i)] for i in logits.argmax(-1).cpu().tolist())
    return {"predictions": predictions, "model": "finetuned_encoder", "device": str(device)}


def _score(rows: list[dict[str, Any]], predicted: list[str]) -> dict[str, Any]:
    truth = [row["label"] for row in rows]
    labels = sorted(set(truth) | set(predicted))
    errors = [
        {
            "subject": row.get("group_id"),
            "text": row["text"],
            "true": row["label"],
            "predicted": pred,
        }
        for row, pred in zip(rows, predicted)
        if row["label"] != pred
    ]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "errors": len(errors),
        "classification_report": classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "subject_bootstrap_accuracy_95ci": _bootstrap_subject_accuracy(rows, predicted),
        "error_examples": errors,
    }


def evaluate(output_path: str | Path = "artifacts/usc-real/strict_novel_text_evaluation.json") -> dict[str, Any]:
    root = Path("artifacts")
    baseline_dir = root / "usc-real"
    train_rows = _read_jsonl(baseline_dir / "train.jsonl")
    validation_rows = _read_jsonl(baseline_dir / "validation.jsonl")
    test_rows = _read_jsonl(baseline_dir / "test.jsonl")
    strict_rows, split_audit = strict_novel_text_test(train_rows, validation_rows, test_rows)
    if not strict_rows:
        raise ValueError("Strict novel-text test is empty")

    results = {}
    sparse = _evaluate_sparse(strict_rows, baseline_dir / "usc_char_tfidf.joblib")
    results["char_tfidf_logistic"] = _score(strict_rows, sparse["predictions"])
    for model_name in ("usc-miniLM", "usc-deberta-xsmall"):
        neural = _evaluate_neural(strict_rows, root / model_name)
        results[model_name] = {
            **_score(strict_rows, neural["predictions"]),
            "device": neural["device"],
        }

    payload = {
        "task": "USC natural catalog-cell field classification",
        "protocol": (
            "subject-held-out test subjects; test strings normalized for whitespace/case; "
            "remove any test string present in train or validation; score one row per remaining exact test string"
        ),
        "important_limit": (
            "This is still one scraped USC Spring 2024 source and uses naturally available field labels. "
            "It does not measure OCR, arbitrary syllabus extraction, other institutions, or independently human-annotated semantics."
        ),
        "split_audit": split_audit,
        "models": results,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run strict exact-text-disjoint USC field model evaluation.")
    parser.add_argument("--output", default="artifacts/usc-real/strict_novel_text_evaluation.json")
    args = parser.parse_args(argv)
    result = evaluate(args.output)
    print(json.dumps({"split_audit": result["split_audit"], "models": {
        name: {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "errors": metrics["errors"],
            "subject_bootstrap_accuracy_95ci": metrics["subject_bootstrap_accuracy_95ci"],
        }
        for name, metrics in result["models"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
