"""Training rows with and without the field's own header.

Every row in the mixed corpus is written ``"<source column>: <value>"``, and the
source column usually names the field ("Class Name: Literature and Popular
Culture"). A model trained on that alone reads the prefix and never has to learn
what the value means: the same checkpoint scores 0.998 with the header and 0.899
on the bare value.

This script writes a training split in which every row appears twice -- once as
published, once with only the value -- so the text tower has to learn both the
header-conditioned and the semantic path. Validation and test are untouched, so
the header-conditioned numbers stay comparable with earlier checkpoints and the
bare-value number is measured separately.

    python scripts/build_header_dropout.py --output data/processed/field_mix_dropout
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pace.field_train import load_jsonl


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Duplicate training rows without their header")
    parser.add_argument("--mix-root", default="data/processed/field_mix")
    parser.add_argument("--output", default="data/processed/field_mix_dropout")
    parser.add_argument("--header-probability", type=float, default=0.5,
                        help="Share of the duplicated corpus that keeps the header")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)

    mix = Path(args.mix_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        rows = load_jsonl(mix / f"{split}.jsonl")
        written: list[dict] = []
        bare_share = 1.0 - args.header_probability
        for row in rows:
            bare = (row.get("bare_text") or "").strip()
            keeps_header = rng.random() < args.header_probability or not bare
            if split != "train":
                # Validation and test stay as published: they are the reference
                # measurement, not a training signal.
                written.append(row)
                continue
            copy = dict(row)
            if not keeps_header and bare:
                copy["text"] = bare
                copy["header_dropped"] = True
            written.append(copy)
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in written:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {"rows": len(written),
                        "without_header": sum(1 for r in written if r.get("header_dropped")),
                        "labels": len({r["label"] for r in written})}
        print(json.dumps({split: stats[split],
                          "bare_share_requested": round(bare_share, 2)}), flush=True)

    (output / "manifest.json").write_text(json.dumps({
        "source": str(mix),
        "header_probability": args.header_probability,
        "seed": args.seed,
        "rows_by_label": dict(Counter(r["label"] for r in load_jsonl(output / "train.jsonl"))),
        "splits": stats,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
