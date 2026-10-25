"""Cross-institution evaluation for the Jev-shaped decision model.

The question is the one a user asks: *this syllabus is from a university the
model has never seen — does it still read it correctly?* For each fold, the
menu of 42 described fields is unchanged, only the documents change. That is
exactly the setting the architecture was chosen for: the choices are described
in words, so a new institution's vocabulary does not have to be a trained class.

The folds come from ``pace.field_mix`` (``data/processed/field_mix/fold-*``), the
same ones the bge/MiniLM baselines were scored on, so the numbers line up.

    python -m pace.jev_cross --encoder artifacts/jev-field \
        --folds-root data/processed/field_mix --output artifacts/jev-cross
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from .field_train import load_jsonl
from .jev_train import train_choice


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Cross-institution evaluation, Jev model")
    parser.add_argument("--encoder", default="artifacts/jev-field",
                        help="Directory holding tokenizer.json and a pretrained model.pt")
    parser.add_argument("--folds-root", default="data/processed/field_mix")
    parser.add_argument("--output", default="artifacts/jev-cross")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-ids", type=int, nargs="*", default=None,
                        help="Only these fold indices (default: all of --folds)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--option-dropout", type=float, default=0.15)
    parser.add_argument("--sampler-power", type=float, default=0.0,
                        help="Class-balanced sampling, as in the main fine-tune")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    encoder = Path(args.encoder)
    tokenizer = Tokenizer.from_file(str(encoder / "tokenizer.json"))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for fold in (args.fold_ids if args.fold_ids is not None else range(args.folds)):
        directory = Path(args.folds_root) / f"fold-{fold}"
        if not directory.exists():
            print(json.dumps({"skipped": str(directory)}), flush=True)
            continue
        stats = json.loads((directory / "corpus_stats.json").read_text(encoding="utf-8")) \
            if (directory / "corpus_stats.json").exists() else {}
        held_out = stats.get("holdout_schools", [])
        print(json.dumps({"fold": fold, "holdout_schools": held_out}), flush=True)
        trained = train_choice(load_jsonl(directory / "train.jsonl"),
                               load_jsonl(directory / "validation.jsonl"),
                               load_jsonl(directory / "test.jsonl"),
                               output_root / f"fold-{fold}", tokenizer,
                               encoder_checkpoint=encoder, epochs=args.epochs,
                               batch_size=args.batch_size, lr=args.lr,
                               patience=args.patience, option_dropout=args.option_dropout,
                               sampler_power=args.sampler_power,
                               arch={key: value for key, value in (("hidden", args.hidden),
                                                                  ("ffn", args.ffn))
                                     if value is not None},
                               device=args.device)
        results.append({
            "fold": fold,
            "holdout_schools": held_out,
            "test_rows": trained["test_rows"],
            "test_accuracy": trained["test"]["accuracy"],
            "test_macro_f1": trained["test"]["macro_f1"],
            "test_weighted_f1": trained["test"]["weighted_f1"],
            "test_ece_10bin": trained["test"]["ece_10bin"],
            "test_per_class_f1": trained["test"].get("per_class_f1"),
            "test_per_class_support": trained["test"].get("per_class_support"),
            "test_top_confusions": trained["test"].get("top_confusions"),
        })

    if not results:
        print(json.dumps({"folds": 0}), flush=True)
        return
    summary = {
        "folds": len(results),
        "pooled_test_rows": int(sum(item["test_rows"] for item in results)),
        "mean_test_accuracy": float(np.mean([item["test_accuracy"] for item in results])),
        "std_test_accuracy": float(np.std([item["test_accuracy"] for item in results])),
        "mean_test_macro_f1": float(np.mean([item["test_macro_f1"] for item in results])),
        "std_test_macro_f1": float(np.std([item["test_macro_f1"] for item in results])),
        "results": results,
    }
    (output_root / "cross_institution_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"},
                     indent=2))
    for item in results:
        print(f"  fold {item['fold']}: holdout={item['holdout_schools']} "
              f"rows={item['test_rows']} acc={item['test_accuracy']:.4f} "
              f"macroF1={item['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
