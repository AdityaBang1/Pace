"""Grow the labelled corpus without moving a single existing test row.

Every number this project has reported for the field classifier was earned on
``data/processed/field_context`` and its three split files. Adding new sources
must not reshuffle those splits: if the 7,571 test rows changed, the old and new
models would no longer be comparable, and the comparison is the whole point.

So the extension is additive by construction:

* the base corpus and its ``train``/``validation``/``test`` group assignment are
  read verbatim and copied through;
* new rows are appended only when their normalised text does not already carry a
  *different* label in the base corpus, so a new source can never make an old
  row ambiguous and drop it from the test set;
* every new group is assigned to ``train``, which is what a newly downloaded
  source is: more to learn from, nothing re-scored.

The one thing that is deliberately new is the *school* a new source belongs to,
because the cross-institution folds key off it. Columbia is now a university the
fold builder can hold out entirely, which is a stronger unseen-institution test
than 1,295 rows drawn from thirteen schools that all appear in training.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .data import save_jsonl
from .field_context import build_context_corpus
from .field_corpus import FIELD_TYPES, normalize_value

SPLIT_NAMES = ("train", "validation", "test")

#: How many new rows one field may contribute. A new source can dwarf the corpus
#: -- Columbia alone is 248 k of 340 k rows, and 169 k of those are section ids --
#: and a model trained on that learns one registrar's layout, which is the exact
#: failure the cross-institution folds measure. The cap keeps the *coverage* the
#: new source buys (Columbia prints 123 distinct day strings, 1,841 clock spans,
#: 1,360 rooms: all under the cap) and drops the bulk repetition that would
#: otherwise make one university most of the training set.
DEFAULT_MAX_PER_LABEL = 2500


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extend_corpus(base_rows: list[dict[str, Any]], extra_rows: Iterable[dict[str, Any]],
                  max_per_label: int | None = DEFAULT_MAX_PER_LABEL,
                  seed: int = 20261019) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append new rows whose text does not contradict a base label.

    A row is dropped when its normalised text already appears in the base corpus
    under another label (the base label stays authoritative), when the text
    appears twice in the new sources under different labels, or when its label is
    not in the taxonomy. Everything else is kept, de-duplicated by
    ``(text, label)`` so a repeated catalog value is one example rather than
    eighty thousand.
    """
    base_labels: dict[str, set[str]] = {}
    for row in base_rows:
        base_labels.setdefault(normalize_value(row["text"]), set()).add(row["label"])

    extra_labels: dict[str, set[str]] = {}
    for row in extra_rows:
        extra_labels.setdefault(normalize_value(row["text"]), set()).add(row["label"])
    contradictory = {text for text, labels in extra_labels.items() if len(labels) > 1}

    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stats = Counter()
    for row in extra_rows:
        label = row.get("label")
        text = normalize_value(row.get("text") or "")
        if not text or not label:
            stats["empty"] += 1
            continue
        if label not in FIELD_TYPES:
            stats["unknown_label"] += 1
            continue
        if base_labels.get(text, {label}) - {label}:
            stats["contradicts_base"] += 1
            continue
        if text in contradictory:
            stats["ambiguous_within_new"] += 1
            continue
        key = (text, label)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        # The heading pools group by source, exactly as the base builder does.
        row["source_key"] = str(row.get("source") or "").split("/")[0]
        kept.append(row)
    # New rows get the same treatment the base corpus got: the heading a value
    # was published under, then ``heading: value``. Deduplication happens here
    # too, on the composed text.
    kept, context_stats = build_context_corpus(kept)
    if max_per_label is not None:
        rng = random.Random(seed)
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in kept:
            by_label.setdefault(row["label"], []).append(row)
        capped: list[dict[str, Any]] = []
        for label in sorted(by_label):
            items = by_label[label]
            if len(items) > max_per_label:
                rng.shuffle(items)
                stats["capped_per_label"] += len(items) - max_per_label
                items = items[:max_per_label]
            capped.extend(items)
        kept = capped
    rows = [*base_rows, *kept]
    report = {
        "base_rows": len(base_rows),
        "new_rows_considered": len(kept) + stats["duplicate"] + stats["contradicts_base"]
        + stats["ambiguous_within_new"] + stats["unknown_label"] + stats["empty"],
        "new_rows_kept": len(kept),
        "dropped": {key: value for key, value in sorted(stats.items()) if value},
        "rows": len(rows),
        "base_by_school": dict(sorted(Counter(_school(row) for row in base_rows).items())),
        "new_by_school": dict(sorted(Counter(_school(row) for row in kept).items())),
        "new_by_label": dict(sorted(Counter(row["label"] for row in kept).items())),
        "new_groups": len({row["group_id"] for row in kept}),
        "new_rows_without_heading": context_stats["rows_without_heading"],
        "max_per_label": max_per_label,
    }
    return rows, report


def _school(row: dict[str, Any]) -> str:
    return str((row.get("meta") or {}).get("school") or "")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Extend the field corpus, keeping the splits")
    parser.add_argument("--base", default="data/processed/field_context/corpus.jsonl")
    parser.add_argument("--base-split-dir", default="data/processed/field_context")
    parser.add_argument("--output", default="data/processed/field_context_ext")
    parser.add_argument("--columbia", default="data/raw/columbia")
    parser.add_argument("--coursera", default="data/raw/coursera/coursera_course_2024.csv")
    parser.add_argument("--sis", nargs="*", default=None,
                        help="Infinite Campus / OneRoster exports to import as extra rows")
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args(argv)

    from .field_sources_ext import load_columbia_examples, load_coursera_examples

    base_rows = _read_jsonl(args.base)
    extra = [*load_columbia_examples(args.columbia), *load_coursera_examples(args.coursera)]
    if args.sis:
        from .sis_import import load_sis_examples

        extra.extend(load_sis_examples(args.sis))
    rows, report = extend_corpus(base_rows, extra)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(output / "corpus.jsonl", rows)
    new_groups = {row["group_id"] for row in rows} - {
        row["group_id"] for row in base_rows}
    for name in SPLIT_NAMES:
        part = _read_jsonl(Path(args.base_split_dir) / f"{name}.jsonl")
        if name == "train":
            # New groups are unlabelled-in-split material: they belong to train.
            part = [*part, *[row for row in rows if row["group_id"] in new_groups]]
        save_jsonl(output / f"{name}.jsonl", part)
        report[f"{name}_rows"] = len(part)
    report["output"] = str(output)
    (output / "corpus_stats.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"new_by_label", "base_by_school"}}, indent=2))


if __name__ == "__main__":
    main()
