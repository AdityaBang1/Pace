"""Cross-institution evaluation: train on universities, test on others.

The question this answers is the one a user actually cares about: *if this
model meets a syllabus from a university it has never seen, does it still work?*

Each fold holds out a disjoint set of universities, trains its own model on
the rest, and evaluates on the held-out universities' documents. The test sets
are disjoint, so pooling them gives a single estimate over many real syllabi
instead of one small sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .field_splits import cross_institution_folds
from .field_train import load_jsonl, train_one


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-institution field classifier evaluation")
    parser.add_argument("--corpus", default="data/processed/field/corpus.jsonl")
    parser.add_argument("--splits", default=None,
                        help="Directory holding prebuilt fold splits (overrides --corpus)")
    parser.add_argument("--fold-plan", default="data/processed/field/fold_plan.json",
                        help="JSON file mapping fold index to held-out universities")
    parser.add_argument("--write-fold-plan", action="store_true",
                        help="Write the fold plan for this corpus and use it")
    parser.add_argument("--output", default="artifacts/field-crossinstitution")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.corpus)
    plan_path = Path(args.fold_plan)
    school_plan = None
    if plan_path.exists() and not args.write_fold_plan:
        school_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    folds = cross_institution_folds(rows, seed=args.seed, folds=args.folds,
                                    school_plan=school_plan)
    if args.write_fold_plan:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps([fold["holdout_schools"] for fold in folds], indent=2)
                             + "\n", encoding="utf-8")
        print(json.dumps({"fold_plan": str(plan_path),
                          "folds": [fold["holdout_schools"] for fold in folds]}, indent=2))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for fold in folds:
        print(json.dumps({"fold": fold["fold"], "holdout_schools": fold["holdout_schools"]}),
              flush=True)
        output = output_root / f"fold-{fold['fold']}"
        kwargs = {"base_model": args.base_model} if args.base_model else {}
        trained = train_one(args.seed, fold["train"], fold["validation"], fold["test"], output,
                            epochs=args.epochs, **kwargs)
        results.append({
            "fold": fold["fold"],
            "holdout_schools": fold["holdout_schools"],
            "holdout_documents": len(fold["holdout_documents"]),
            "train_rows": trained["train_rows_before_oversampling"],
            "validation_rows": trained["validation_rows"],
            "test_rows": trained["test_rows"],
            "test_accuracy": trained["test"]["accuracy"],
            "test_macro_f1": trained["test"]["macro_f1"],
            "test_weighted_f1": trained["test"]["weighted_f1"],
            "test_ece_10bin": trained["test"]["ece_10bin"],
            "test_per_class_f1": trained["test"]["per_class_f1"],
            "test_per_class_support": trained["test"]["per_class_support"],
            "test_top_confusions": trained["test"]["top_confusions"],
        })

    pooled_rows = sum(r["test_rows"] for r in results)
    summary = {
        "folds": len(results),
        "pooled_test_rows": pooled_rows,
        "mean_test_accuracy": float(np.mean([r["test_accuracy"] for r in results])),
        "std_test_accuracy": float(np.std([r["test_accuracy"] for r in results])),
        "mean_test_macro_f1": float(np.mean([r["test_macro_f1"] for r in results])),
        "std_test_macro_f1": float(np.std([r["test_macro_f1"] for r in results])),
        "results": results,
    }
    (output_root / "cross_institution_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    for entry in results:
        print(f"  fold {entry['fold']}: holdout={entry['holdout_schools']} "
              f"rows={entry['test_rows']} acc={entry['test_accuracy']:.4f} "
              f"macroF1={entry['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
