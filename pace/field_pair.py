"""Paired comparison of the bare-value and context-aware field classifiers.

The two corpora are not the same size -- the heading-aware corpus removes
duplicate (heading, value) pairs -- so comparing their headline test numbers
directly would confound "which model is better" with "which rows were scored".

This script removes that confound: for each fold it takes the rows both
corpora agree on, and scores both checkpoints on exactly those rows.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .field_ood import evaluate_rows
from .field_train import load_jsonl


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Identify a row by the *value* it carries, not by the composed text.

    The context corpus rewrites ``text`` to ``"Heading: value"`` and rehashes
    it, so the two corpora can only be lined up through the original value,
    which the context row keeps as ``bare_text``.
    """
    value = row.get("bare_text") or row["text"]
    return (row["group_id"], value, row["label"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired bare vs context comparison")
    parser.add_argument("--bare-corpus", default="data/processed/field/corpus.jsonl")
    parser.add_argument("--context-corpus", default="data/processed/field_context/corpus.jsonl")
    parser.add_argument("--bare-folds", default="artifacts/field-crossinstitution")
    parser.add_argument("--context-folds", default="artifacts/field-crossinstitution-context")
    parser.add_argument("--output", default="artifacts/field-ood/paired_comparison.json")
    args = parser.parse_args(argv)

    bare_rows = load_jsonl(args.bare_corpus)
    context_rows = load_jsonl(args.context_corpus)
    context_by_key = {_key(row): row for row in context_rows}

    from .field_splits import cross_institution_folds

    plan_path = Path("data/processed/field/fold_plan.json")
    school_plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    folds = cross_institution_folds(bare_rows, folds=len(school_plan or []) or 3,
                                    school_plan=school_plan)

    results: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = Path(args.bare_folds) / f"fold-{fold['fold']}"
        context_dir = Path(args.context_folds) / f"fold-{fold['fold']}"
        bare_metrics_path = fold_dir / "metrics.json"
        context_metrics_path = context_dir / "metrics.json"
        if not bare_metrics_path.exists() or not context_metrics_path.exists():
            continue
        bare_saved = json.loads(bare_metrics_path.read_text(encoding="utf-8"))
        context_saved = json.loads(context_metrics_path.read_text(encoding="utf-8"))

        heldout_docs = set(fold["holdout_documents"])
        bare_test = [row for row in bare_rows if row["group_id"] in heldout_docs]
        shared = [row for row in bare_test if _key(row) in context_by_key]
        shared_context = [context_by_key[_key(row)] for row in shared]
        if not shared:
            continue

        # A checkpoint can only score labels it was trained on; drop anything
        # outside its own vocabulary rather than letting the loader raise.
        bare_labels = set(bare_saved["train_label_counts"])
        context_labels = set(context_saved["train_label_counts"])
        shared = [row for row in shared if row["label"] in bare_labels
                  and row["label"] in context_labels]
        shared_context = [context_by_key[_key(row)] for row in shared]
        if not shared:
            continue

        bare_eval = evaluate_rows(fold_dir, shared, Counter(bare_saved["train_label_counts"]))
        context_eval = evaluate_rows(context_dir, shared_context,
                                     Counter(context_saved["train_label_counts"]))
        results.append({
            "fold": fold["fold"],
            "holdout_schools": fold["holdout_schools"],
            "bare_test_rows": len(bare_test),
            "context_test_rows": int(context_saved.get("test_rows", 0) or 0),
            "paired_rows": len(shared),
            "bare_accuracy": bare_eval["accuracy"],
            "bare_macro_f1": bare_eval["macro_f1"],
            "context_accuracy": context_eval["accuracy"],
            "context_macro_f1": context_eval["macro_f1"],
        })
        print(json.dumps(results[-1]), flush=True)

    if results:
        summary = {
            "folds": len(results),
            "paired_rows": sum(r["paired_rows"] for r in results),
            "mean_bare_accuracy": sum(r["bare_accuracy"] for r in results) / len(results),
            "mean_context_accuracy": sum(r["context_accuracy"] for r in results) / len(results),
            "mean_bare_macro_f1": sum(r["bare_macro_f1"] for r in results) / len(results),
            "mean_context_macro_f1": sum(r["context_macro_f1"] for r in results) / len(results),
            "results": results,
        }
    else:
        summary = {"folds": 0, "results": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
