"""Grouped school-holdout folds: every institution is tested exactly once.

The document fold elsewhere holds out whole syllabus *documents* while keeping
their schools in training, so it answers "can it read another page from a school
it knows?". This one answers the product question: "can it read a school it has
never seen?". Schools are grouped, never split, and every school lands in exactly
one test fold, so the three folds pool back into all 1,638 real syllabus rows.

Two leakage rules, both measured afterwards by :func:`leakage_report`:

* a held-out school's rows are removed from training across *all* families, and
* any composed row carrying that school's document id is removed too, because a
  composed row re-pairs a value with a heading and would leak the vocabulary.

    python scripts/build_school_folds.py --folds 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pace.transfer import (CORPUS, MIX, institution_of, leakage_report,  # noqa: E402
                           load_jsonl)

SYLLABUS = "Dev-the-dev91/syllabus-cleaned"
OUT = Path("data/processed/transfer")


def _num(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def syllabus_schools(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") != SYLLABUS:
            continue
        if row.get("family") not in (None, "natural"):
            continue
        grouped[institution_of(row)].append(row)
    return dict(grouped)


def group_schools(grouped: dict[str, list[dict]], folds: int) -> list[list[str]]:
    """Balance the folds by row count, largest school first."""
    ordered = sorted(grouped, key=lambda name: -len(grouped[name]))
    buckets: list[list[str]] = [[] for _ in range(folds)]
    sizes = [0] * folds
    for name in ordered:
        target = sizes.index(min(sizes))
        buckets[target].append(name)
        sizes[target] += len(grouped[name])
    return buckets


def _document_id(row: dict) -> str:
    source_id = str(row.get("source_id") or "")
    if source_id.startswith("syllabus:"):
        return source_id
    return str((row.get("meta") or {}).get("document_id") or "")


def build(index: int, schools: list[str], grouped: dict[str, list[dict]],
          seed: int) -> dict:
    out = OUT / f"school-fold-{index}"
    test = [row for name in schools for row in grouped[name]]
    heldout_documents = {_document_id(row) for row in test} - {""}

    def keep(row: dict) -> bool:
        if row.get("source") == SYLLABUS and institution_of(row) in schools:
            return False
        if _document_id(row) in heldout_documents:
            return False
        return True

    train = [row for row in load_jsonl(MIX / "train.jsonl") if keep(row)]
    validation = [row for row in load_jsonl(MIX / "validation.jsonl") if keep(row)]
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "fold": index,
        "holdout_schools": sorted(schools),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_labels": dict(Counter(row["label"] for row in test)),
        "train_institutions": len({institution_of(row) for row in train}),
        "leakage": leakage_report(train, test),
        "seed": seed,
    }
    (out / "fold.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Grouped school-holdout folds")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)

    rows = load_jsonl(CORPUS)
    grouped = syllabus_schools(rows)
    buckets = group_schools(grouped, args.folds)
    summaries = [build(index, bucket, grouped, args.seed)
                 for index, bucket in enumerate(buckets)]
    pooled = sum(item["test_rows"] for item in summaries)
    print(json.dumps({"folds": len(summaries), "schools": len(grouped),
                      "pooled_test_rows": pooled,
                      "per_fold": [{key: item[key] for key in
                                    ("fold", "holdout_schools", "test_rows",
                                     "leakage")} for item in summaries]}, indent=2))


if __name__ == "__main__":
    main()
