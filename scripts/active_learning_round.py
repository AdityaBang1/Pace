"""One active-learning round: spend labels where the reader is least sure.

The pool is the validation split, which the checkpoint being probed did not train
on. Rows are ranked by predictive entropy (and, as a control, by random order at
the same budget), the top slice is appended to the training set, and the result is
written next to the original split so training can proceed with the normal
``pace.jev_train`` command. Nothing here decides whether active learning helped —
that needs the retrain and the held-out score, and both arms are printed so they
can be compared.

    python scripts/active_learning_round.py --checkpoint artifacts/jev-field-v4 \
        --budget 1500 --output data/processed/field_mix_al
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from pace.field_train import load_jsonl
from pace.jev_model import JevEngine


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Select the most uncertain rows to label")
    parser.add_argument("--checkpoint", default="artifacts/jev-field-v4")
    parser.add_argument("--pool", default="data/processed/field_mix/validation.jsonl")
    parser.add_argument("--train", default="data/processed/field_mix_dropout/train.jsonl")
    parser.add_argument("--validation", default="data/processed/field_mix/validation.jsonl")
    parser.add_argument("--test", default="data/processed/field_mix/test.jsonl")
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="data/processed/field_mix_al")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)

    pool = load_jsonl(args.pool)
    engine = JevEngine(args.checkpoint, device=args.device)
    probabilities = engine.probabilities([row["text"] for row in pool])
    entropy = []
    for row in probabilities:
        values = np.asarray(list(row.values()), dtype=float)
        values = values / max(values.sum(), 1e-9)
        entropy.append(float(-(values * np.log(np.clip(values, 1e-12, None))).sum()))
    order = np.argsort(entropy)[::-1]

    uncertain = [pool[index] for index in order[: args.budget]]
    rng = random.Random(args.seed)
    control = list(pool)
    rng.shuffle(control)
    control = control[: args.budget]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = load_jsonl(args.train)
    for name, extra in (("al", uncertain), ("random", control)):
        rows = base + [dict(row, selected_by=name) for row in extra]
        with (output / f"train-{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("validation", "test"):
        source = Path(args.validation if split == "validation" else args.test)
        (output / f"{split}.jsonl").write_text(source.read_text(encoding="utf-8"),
                                               encoding="utf-8")

    summary = {
        "checkpoint": str(args.checkpoint),
        "pool_rows": len(pool),
        "budget": args.budget,
        "mean_entropy_selected": float(np.mean([entropy[index] for index in order[: args.budget]])),
        "mean_entropy_pool": float(np.mean(entropy)),
        "selected_label_counts": {
            label: sum(1 for row in uncertain if row["label"] == label)
            for label in sorted({row["label"] for row in uncertain})
        },
        "train_rows_al": len(base) + len(uncertain),
        "train_rows_random": len(base) + len(control),
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
