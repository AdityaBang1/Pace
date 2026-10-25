"""Leave-one-source-out: does the reader transfer to an institution it never saw?

The school-holdout benchmark keeps a corpus but changes the school inside it. This
one is harder: an entire *publisher* is removed -- a Swedish university's syllabus
form, MIT's catalog, USC's catalog, Michigan's course descriptions -- and the
model is asked to read that publisher's cells with only the other institutions'
examples. Nothing about the held-out layout, its column names or its vocabulary
exists in the training data.

Labels the remaining sources do not use stay in the menu (the menu is made of
descriptions, not trained classes), and those are reported separately: predicting
a field from its description alone is a different claim from transferring a field
that was trained on other publishers.

    python scripts/build_source_holdout.py --sources 4 --train-rows 40000 \
        --output artifacts/source-holdout
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from pace.field_train import load_jsonl
from pace.jev_eval import evaluate_rows
from pace.jev_model import JevEngine
from pace.jev_train import train_choice

# The largest and most stylistically distinct publishers, so a fold is a real
# change of document form rather than a change of subject area.
DEFAULT_SOURCES = [
    "anordkvist/gu-course-syllabus",
    "chaseharmon/MIT_Course_Catalog",
    "USC/USC-Course-Catalog",
    "Dev-the-dev91/syllabus-cleaned",
    "alecwatts/umich-course-descriptions",
]


def _sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Stratified sample by label, so a small source is not swallowed."""
    if len(rows) <= limit:
        return rows
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = random.Random(seed)
    per_label = max(1, limit // max(len(by_label), 1))
    kept: list[dict] = []
    leftovers: list[dict] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        kept.extend(items[:per_label])
        leftovers.extend(items[per_label:])
    # A rare label cannot fill its quota, so top up from what is left rather than
    # training on a fraction of the budget.
    if len(kept) < limit and leftovers:
        rng.shuffle(leftovers)
        kept.extend(leftovers[: limit - len(kept)])
    rng.shuffle(kept)
    return kept[:limit]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Leave-one-source-out evaluation")
    parser.add_argument("--mix-root", default="data/processed/field_mix")
    parser.add_argument("--encoder", default="artifacts/jev-field-pretrain",
                        help="MLM-only encoder; no labelled source is baked in")
    parser.add_argument("--output", default="artifacts/source-holdout")
    parser.add_argument("--sources", type=int, default=len(DEFAULT_SOURCES))
    parser.add_argument("--source-list", nargs="*", default=None)
    parser.add_argument("--train-rows", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--option-dropout", type=float, default=0.15)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--header-probability", type=float, default=0.5,
                        help="Share of training rows that keep their column header; the rest "
                             "train on the bare value, as in the v4 recipe")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    mix = Path(args.mix_root)
    train_pool = load_jsonl(mix / "train.jsonl")
    validation_pool = load_jsonl(mix / "validation.jsonl")
    test_pool = load_jsonl(mix / "test.jsonl")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(Path(args.encoder) / "tokenizer.json"))

    sources = args.source_list or DEFAULT_SOURCES[: args.sources]
    report: list[dict] = []
    for index, source in enumerate(sources):
        held_out = [row for row in train_pool + validation_pool + test_pool
                    if row["source"] == source]
        if not held_out:
            print(json.dumps({"skipped": source}), flush=True)
            continue
        rest_train = [row for row in train_pool if row["source"] != source]
        rest_validation = [row for row in validation_pool if row["source"] != source]
        if not rest_validation:
            rest_validation = _sample(rest_train, 4000, 7 + index)
        train_rows = _sample(rest_train, args.train_rows, 11 + index)
        if args.header_probability < 1.0:
            rng = random.Random(101 + index)
            dropped = 0
            for row in train_rows:
                bare = (row.get("bare_text") or "").strip()
                if bare and rng.random() >= args.header_probability:
                    row["text"] = bare
                    dropped += 1
            print(json.dumps({"fold": source, "train_rows_without_header": dropped}),
                  flush=True)
        seen_labels = {row["label"] for row in train_rows + rest_validation}
        unseen_labels = sorted({row["label"] for row in held_out} - seen_labels)

        print(json.dumps({"fold": source, "held_out_rows": len(held_out),
                          "train_rows": len(train_rows),
                          "labels_never_seen": unseen_labels}), flush=True)
        trained = train_choice(train_rows, rest_validation, held_out,
                               output / f"fold-{index}", tokenizer,
                               encoder_checkpoint=Path(args.encoder),
                               epochs=args.epochs, batch_size=args.batch_size,
                               lr=args.lr, patience=args.patience,
                               option_dropout=args.option_dropout,
                               sampler_power=args.sampler_power,
                               device=args.device)
        engine = JevEngine(output / f"fold-{index}", device=args.device or "cpu")
        scored = evaluate_rows(engine, held_out)
        known = [row for row in held_out if row["label"] in seen_labels]
        known_metrics = evaluate_rows(engine, known) if known else {}
        report.append({
            "held_out_source": source,
            "held_out_rows": len(held_out),
            "train_rows": len(train_rows),
            "labels_never_seen": unseen_labels,
            "labels_in_fold": sorted({row["label"] for row in held_out}),
            "accuracy": scored["accuracy"],
            "macro_f1": scored["macro_f1"],
            "weighted_f1": scored["weighted_f1"],
            "ece_10bin": scored["ece_10bin"],
            "mean_confidence": scored["mean_confidence"],
            "accuracy_on_labels_seen_in_training": known_metrics.get("accuracy"),
            "rows_with_labels_seen_in_training": len(known),
            "per_class_f1": scored["per_class_f1"],
            "per_class_support": scored["per_class_support"],
            "top_confusions": scored["top_confusions"],
            "trained_test_accuracy": trained["test"]["accuracy"],
            "trained_test_macro_f1": trained["test"]["macro_f1"],
        })

    summary = {
        "folds": len(report),
        "mean_accuracy": float(np.mean([item["accuracy"] for item in report])) if report else None,
        "mean_macro_f1": float(np.mean([item["macro_f1"] for item in report])) if report else None,
        "mean_accuracy_on_seen_labels": float(np.mean(
            [item["accuracy_on_labels_seen_in_training"] for item in report
             if item["accuracy_on_labels_seen_in_training"] is not None])) if report else None,
        "results": report,
    }
    (output / "source_holdout_report.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"},
                     indent=2), flush=True)
    for item in report:
        print(f"  {item['held_out_source']}: rows={item['held_out_rows']} "
              f"acc={item['accuracy']:.4f} macroF1={item['macro_f1']:.4f} "
              f"seen-label acc={item['accuracy_on_labels_seen_in_training']}", flush=True)


if __name__ == "__main__":
    main()
