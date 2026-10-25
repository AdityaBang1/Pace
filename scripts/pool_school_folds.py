"""Score the grouped school-holdout folds as one pooled test set.

Each fold's model only ever scores schools it never saw, and every school is
tested exactly once, so concatenating the folds gives one estimate over all real
syllabus rows rather than three small opinions.

Two numbers are reported because this corpus repeats generic cell values across
schools ("Attendance", "3", "Lecture"). The pooled score counts every row; the
clean score drops test rows whose exact text also appears in that fold's training
set, which is the strictest reading of "never seen".

    python scripts/pool_school_folds.py --folds 3 --output artifacts/transfer-eval/school-holdout.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & ((confidence < high) if index < bins - 1
                                      else (confidence <= high))
        if mask.any():
            total += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return total


#: Coarse families. The 42-field menu splits work into four classes whose
#: boundaries the upstream syllabus extractor does not draw (a weekly lecture
#: topic, a reading, and a graded task all arrive labelled ``assignments``). If
#: the model is right about the family but wrong inside it, the taxonomy is the
#: problem rather than the reading.
FAMILIES: dict[str, str] = {
    "course_title": "identity", "course_code": "identity",
    "course_description": "identity", "program": "identity",
    "course_level": "identity", "class_type": "identity",
    "class_section": "identity", "capacity": "identity",
    "instructor": "people", "office_hours": "people",
    "contact_information": "people",
    "meeting_time": "when", "meeting_days": "when", "term": "when",
    "session_dates": "when", "course_schedule": "when", "exam_schedule": "when",
    "due_date": "when", "drop_add_deadline": "when",
    "assignments": "work", "weekly_readings": "work", "assessment": "work",
    "materials": "work",
    "grading_policy": "grading", "grades": "grading",
    "entry_requirements": "rules", "restrictions": "rules",
    "prerequisites": "rules", "attendance_policy": "rules",
    "academic_integrity": "rules", "course_policies": "rules",
    "disability_services": "rules",
    "credits": "admin", "confirmation": "admin", "teaching_form": "admin",
    "course_modality": "admin", "course_evaluation": "admin",
    "learning_outcomes": "admin", "course_content": "admin",
    "additional_information": "admin", "other": "admin",
}


def _family_accuracy(labels_true: list[str], labels_pred: list[str]) -> dict:
    pairs = [(FAMILIES.get(t, "admin"), FAMILIES.get(p, "admin"))
             for t, p in zip(labels_true, labels_pred)]
    if not pairs:
        return {"rows": 0, "accuracy": None, "per_family": {}}
    hits = sum(1 for t, p in pairs if t == p)
    per_family = {}
    for family in sorted({t for t, _ in pairs}):
        rows = [(t, p) for t, p in pairs if t == family]
        per_family[family] = {"rows": len(rows),
                              "accuracy": sum(1 for t, p in rows if t == p) / len(rows)}
    return {"rows": len(pairs), "accuracy": hits / len(pairs), "per_family": per_family}


def _summary(labels_true: list[str], labels_pred: list[str],
             confidence: np.ndarray, vocabulary: list[str]) -> dict:
    truth = np.asarray(labels_true)
    prediction = np.asarray(labels_pred)
    correct = truth == prediction
    present = sorted(set(labels_true))
    per_class = {label: float(f1_score(truth == label, prediction == label, zero_division=0))
                 for label in present}
    support = Counter(labels_true)
    confusions = Counter((t, p) for t, p in zip(labels_true, labels_pred) if t != p)
    return {
        "rows": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0,
                                   labels=present)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "mean_confidence": float(confidence.mean()),
        "ece_10bin": _ece(confidence, correct),
        "classes": len(present),
        "per_class_f1": per_class,
        "per_class_support": {label: int(support[label]) for label in present},
        "top_confusions": [[f"{real} -> {guessed}", count]
                           for (real, guessed), count in confusions.most_common(15)],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Pool the school-holdout folds")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--data", default="data/processed/transfer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/transfer-eval/school-holdout.json")
    args = parser.parse_args(argv)

    from pace.field_train import load_jsonl
    from pace.jev_model import JevEngine

    truths: list[str] = []
    predictions: list[str] = []
    confidences: list[float] = []
    clean_mask: list[bool] = []
    per_fold: list[dict] = []
    vocabulary: list[str] = []

    for index in range(args.folds):
        checkpoint = Path(args.root) / f"school-fold-{index}"
        test_path = Path(args.data) / f"school-fold-{index}" / "test.jsonl"
        train_path = Path(args.data) / f"school-fold-{index}" / "train.jsonl"
        if not (checkpoint / "model.pt").exists():
            print(json.dumps({"skipped": str(checkpoint)}))
            continue
        rows = load_jsonl(test_path)
        seen = {_normal(row["text"]) for row in load_jsonl(train_path)}
        engine = JevEngine(checkpoint, device=args.device or "cpu")
        vocabulary = engine.labels
        index_of = {label: position for position, label in enumerate(engine.labels)}
        kept = [row for row in rows if row["label"] in index_of]
        outputs = engine.classify_many([row["text"] for row in kept], top_k=1)
        fold_truth = [row["label"] for row in kept]
        fold_pred = [item["label"] for item in outputs]
        fold_conf = np.asarray([item["confidence"] for item in outputs])
        truths.extend(fold_truth)
        predictions.extend(fold_pred)
        confidences.extend(fold_conf.tolist())
        clean_mask.extend(_normal(row["text"]) not in seen for row in kept)
        per_fold.append({
            "fold": index,
            "holdout_schools": json.loads((Path(args.data) / f"school-fold-{index}"
                                           / "fold.json").read_text(encoding="utf-8"))["holdout_schools"],
            **_summary(fold_truth, fold_pred, fold_conf, engine.labels),
        })
        print(json.dumps({"fold": index, "rows": len(kept),
                          "accuracy": per_fold[-1]["accuracy"],
                          "macro_f1": per_fold[-1]["macro_f1"]}), flush=True)

    confidence = np.asarray(confidences)
    mask = np.asarray(clean_mask)
    pooled = _summary(truths, predictions, confidence, vocabulary)
    clean = _summary([t for t, keep in zip(truths, clean_mask) if keep],
                     [p for p, keep in zip(predictions, clean_mask) if keep],
                     confidence[mask], vocabulary)
    pooled["family"] = _family_accuracy(truths, predictions)
    clean["family"] = _family_accuracy(
        [t for t, keep in zip(truths, clean_mask) if keep],
        [p for p, keep in zip(predictions, clean_mask) if keep])
    result = {
        "folds": len(per_fold),
        "per_fold": per_fold,
        "pooled": pooled,
        "clean": clean,
        "clean_rows": int(mask.sum()),
        "generic_value_overlap_rows": int((~mask).sum()),
        "mean_fold_accuracy": float(np.mean([fold["accuracy"] for fold in per_fold])),
        "std_fold_accuracy": float(np.std([fold["accuracy"] for fold in per_fold])),
        "mean_fold_macro_f1": float(np.mean([fold["macro_f1"] for fold in per_fold])),
        "std_fold_macro_f1": float(np.std([fold["macro_f1"] for fold in per_fold])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": {key: pooled[key] for key in
                                 ("rows", "accuracy", "macro_f1", "ece_10bin", "classes")},
                      "clean": {key: clean[key] for key in
                                ("rows", "accuracy", "macro_f1", "ece_10bin")},
                      "mean_fold_macro_f1": result["mean_fold_macro_f1"],
                      "std_fold_macro_f1": result["std_fold_macro_f1"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
