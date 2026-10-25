"""Build the real field corpus on disk and write every evaluation split.

Three split views are produced, each answering a different question:

``group``
    Whole groups (course subject, course, document) are held out, so no
    document contributes to both train and test. This is the in-distribution
    number.

``cross_institution``
    Entire universities are held out of training. This is the number that says
    whether the model learned the field or learned one school's layout.

``by_source``
    One file per real source, so a model trained on one schema can be tested on
    a schema it has never seen.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import save_jsonl
from .field_corpus import build_field_corpus


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(path, rows)


def _split_by_group(rows: list[dict[str, Any]], seed: int,
                    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
                    ) -> dict[str, list[dict[str, Any]]]:
    """Hold out whole groups while guaranteeing every field type is learnable.

    Field types are long-tailed, so a purely random group split can leave a rare
    type (two examples of a drop deadline) entirely in the test split, where it
    is unlearnable. Rare types are therefore pinned into the training split and
    the rest is split at random.
    """
    groups = sorted({row["group_id"] for row in rows})
    label_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label_groups[row["label"]].add(row["group_id"])
    rng = random.Random(seed)
    rng.shuffle(groups)
    assignment: dict[str, str] = {}
    for label, label_group_set in label_groups.items():
        if len(label_group_set) <= 5:
            assignment[min(label_group_set)] = "train"
    remaining = [g for g in groups if g not in assignment]
    first = round(len(remaining) * fractions[0])
    second = round(len(remaining) * (fractions[0] + fractions[1]))
    assignment.update({g: "train" for g in remaining[:first]})
    assignment.update({g: "validation" for g in remaining[first:second]})
    assignment.update({g: "test" for g in remaining[second:]})
    # Safety net: any type left without a training group moves one group over.
    for label, label_group_set in label_groups.items():
        if not any(assignment.get(g) == "train" for g in label_group_set):
            assignment[min(label_group_set)] = "train"
    return {name: [row for row in rows if assignment.get(row["group_id"]) == name]
            for name in ("train", "validation", "test")}


def choose_holdout_schools(rows: list[dict[str, Any]], target_fraction: float = 0.2,
                           min_documents: int = 3) -> list[str]:
    """Pick whole universities to hold out, largest first, covering a share of docs."""
    docs_by_school: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        school = (row.get("meta") or {}).get("school")
        if school:
            docs_by_school[school].add(row["group_id"])
    eligible = sorted(
        ((school, len(docs)) for school, docs in docs_by_school.items() if len(docs) >= min_documents),
        key=lambda item: (-item[1], item[0]),
    )
    total_docs = sum(len(docs) for docs in docs_by_school.values())
    target = max(1, int(total_docs * target_fraction))
    chosen: list[str] = []
    covered = 0
    for school, count in eligible:
        if covered >= target:
            break
        chosen.append(school)
        covered += count
    if len(chosen) < 2 and eligible:
        chosen = [school for school, _ in eligible[:2]]
    return sorted(chosen)


def cross_institution_folds(rows: list[dict[str, Any]], seed: int = 20261019,
                            folds: int = 3,
                            school_plan: list[list[str]] | None = None
                            ) -> list[dict[str, Any]]:
    """Split the universities into disjoint held-out folds.

    One held-out pair gives a test set of a few hundred rows, which is too
    small to trust. Several disjoint folds each train their own model on the
    remaining universities, and their test sets are disjoint, so the pooled
    result covers many more real syllabi while every model still sees a
    realistic amount of training variety.

    ``school_plan`` fixes which universities go into which fold. It must be
    passed when two different corpora are compared, so that both are measured
    on exactly the same held-out universities.
    """
    syllabus_rows = [row for row in rows if row["group_id"].startswith("syllabus-doc:")]
    docs_by_school: dict[str, set[str]] = defaultdict(set)
    for row in syllabus_rows:
        school = (row.get("meta") or {}).get("school")
        if school:
            docs_by_school[school].add(row["group_id"])
    if school_plan is not None:
        assigned = [list(group) for group in school_plan]
    else:
        schools = sorted(docs_by_school, key=lambda s: (-len(docs_by_school[s]), s))
        if len(schools) < folds * 2:
            raise ValueError("not enough universities to build disjoint folds")
        # Deal the universities round-robin so every fold gets a similar mix of
        # large and small universities.
        assigned = [[] for _ in range(folds)]
        for index, school in enumerate(schools):
            assigned[index % folds].append(school)

    result: list[dict[str, Any]] = []
    for index, holdout in enumerate(assigned):
        heldout_docs = {doc for school in holdout for doc in docs_by_school[school]}
        train_rows = [row for row in rows if row["group_id"] not in heldout_docs]
        test_rows = [row for row in rows if row["group_id"] in heldout_docs]
        # The validation slice comes from the *training* universities. Holding a
        # university out means it is out for good: selecting the epoch on it
        # would quietly tune on the test set.
        train_groups = sorted({row["group_id"] for row in train_rows})
        rng = random.Random(seed + index)
        rng.shuffle(train_groups)
        cut = max(1, round(len(train_groups) * 0.04))
        validation_groups = set(train_groups[:cut])
        validation_rows = [row for row in train_rows if row["group_id"] in validation_groups]
        fit_rows = [row for row in train_rows if row["group_id"] not in validation_groups]
        result.append({
            "fold": index,
            "holdout_schools": holdout,
            "holdout_documents": sorted(heldout_docs),
            "train": fit_rows,
            "validation": validation_rows,
            "test": test_rows,
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the real field corpus and its splits")
    parser.add_argument("--output", default="data/processed/field")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--cap-per-label", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    corpus_path = output / "corpus.jsonl"
    if corpus_path.exists() and not args.rebuild:
        rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stats = json.loads((output / "corpus_stats.json").read_text(encoding="utf-8"))
    else:
        built = build_field_corpus(cap_per_label=args.cap_per_label, seed=args.seed)
        rows = built["rows"]
        stats = built["stats"]
        stats["ambiguous_examples"] = {k: v for k, v in list(built["ambiguous"].items())[:50]}
        _write(rows, corpus_path)
        (output / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # ---- view 1: group-wise split over every source ------------------------
    group_split = _split_by_group(rows, args.seed)
    for name, part in group_split.items():
        _write(part, output / "group" / f"{name}.jsonl")

    # ---- view 2: cross-institution split ----------------------------------
    syllabus_rows = [r for r in rows if r["group_id"].startswith("syllabus-doc:")]
    holdout = choose_holdout_schools(syllabus_rows)
    heldout_docs = {r["group_id"] for r in syllabus_rows
                    if (r.get("meta") or {}).get("school") in holdout}
    train_rows = [r for r in rows if r["group_id"] not in heldout_docs]
    remaining = [r for r in rows if r["group_id"] in heldout_docs]
    # Split the held-out universities into validation and test by document.
    heldout_docs_sorted = sorted(heldout_docs)
    rng = random.Random(args.seed)
    rng.shuffle(heldout_docs_sorted)
    cut = max(1, round(len(heldout_docs_sorted) * 0.5))
    validation_docs = set(heldout_docs_sorted[:cut])
    test_docs = set(heldout_docs_sorted[cut:])
    cross = {
        "train": train_rows,
        "validation": [r for r in remaining if r["group_id"] in validation_docs],
        "test": [r for r in remaining if r["group_id"] in test_docs],
    }
    for name, part in cross.items():
        _write(part, output / "cross_institution" / f"{name}.jsonl")

    # ---- view 3: one file per source --------------------------------------
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_key"]].append(row)
    for source, part in by_source.items():
        _write(part, output / "by_source" / f"{source}.jsonl")

    summary = {
        "rows": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "by_source": {k: len(v) for k, v in sorted(by_source.items())},
        "by_label": dict(sorted(Counter(r["label"] for r in rows).items())),
        "group_split": {k: len(v) for k, v in group_split.items()},
        "group_split_groups": {k: len({r["group_id"] for r in v}) for k, v in group_split.items()},
        "cross_institution": {k: len(v) for k, v in cross.items()},
        "holdout_schools": holdout,
        "holdout_documents": sorted(heldout_docs),
        "corpus_stats": stats,
    }
    (output / "splits_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"corpus_stats", "holdout_documents"}},
                     indent=2))


if __name__ == "__main__":
    main()
