"""Institution-holdout benchmark: the number this project should be judged on.

A random row split answers "can the model read another page like the ones it
trained on?". The product question is different: "can it read a school it has
never seen?". This module builds that benchmark honestly.

Two facts about the existing data shape the design.

* Only ``Dev-the-dev91/syllabus-cleaned`` carries a school per row. Every other
  source is one institution, so the institution of a row is its source unless
  the row says otherwise.
* Composed families (``cross``, ``heading``, ``bare``, ``short``, ``sentence``,
  ``schedule``, ``value_diversity``) re-pair a value with a heading borrowed from
  somewhere in the corpus. That is a legitimate augmentation for training, but it
  must never appear in a held-out evaluation: it leaks the heading vocabulary of
  the training institutions.

So evaluation rows are **natural only** -- a value as a real document printed it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CORPUS = Path("data/processed/field/corpus.jsonl")
MIX = Path("data/processed/field_mix_ext")

#: Sources that are one institution each. ``syllabus-cleaned`` is absent because
#: its rows carry their own school in ``meta.school``.
SINGLE_INSTITUTION: dict[str, str] = {
    "twangodev/uwcourses": "University of Washington",
    "anordkvist/gu-course-syllabus": "University of Gothenburg",
    "USC/USC-Course-Catalog": "University of Southern California",
    "chaseharmon/MIT_Course_Catalog": "MIT (catalog)",
    "Coursera/Coursera-Catalog": "Coursera",
    "syllabus-finetune": "SyllabusEntityCorpus",
    "iristun/CU-DocBench": "CU-DocBench",
    "alecwatts/umich-course-descriptions": "University of Michigan (catalog)",
}

#: Below this many natural rows a held-out institution cannot support a
#: per-class number, so it is pooled instead of being its own fold.
MIN_FOLD_ROWS = 200


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def institution_of(row: dict[str, Any]) -> str:
    """The institution a row came from, without trusting the compose families."""
    school = ((row.get("meta") or {}).get("school")
              or row.get("school") or "").strip()
    if school:
        return school
    return SINGLE_INSTITUTION.get(row.get("source", ""), row.get("source", "unknown"))


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _fingerprint(text: str) -> str:
    """Exact-content identity, so a duplicate cannot be counted as two rows."""
    return hashlib.sha256(_normal(text).encode("utf-8")).hexdigest()


def _shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", _normal(text))
    if len(words) < width:
        return {tuple(words)}
    return {tuple(words[index:index + width]) for index in range(len(words) - width + 1)}


def near_duplicate_rate(reference: Iterable[str], target: Iterable[str],
                        threshold: float = 0.8) -> float:
    """Fraction of target texts almost entirely contained in some reference text."""
    reference = [text for text in reference if text]
    target = [text for text in target if text]
    if not target:
        return 0.0
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    buckets = [_shingles(text) for text in reference]
    for position, shingles in enumerate(buckets):
        for shingle in shingles:
            index[shingle].append(position)
    hits = 0
    for text in target:
        shingles = _shingles(text)
        if not shingles:
            continue
        overlap: Counter[int] = Counter()
        for shingle in shingles:
            for position in index.get(shingle, ()):  # at most a handful
                overlap[position] += 1
        if overlap:
            best = max(overlap.values()) / len(shingles)
            if best >= threshold:
                hits += 1
    return hits / len(target)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def natural_by_institution(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("family") not in (None, "natural"):
            continue
        grouped[institution_of(row)].append(row)
    return dict(grouped)


def fold_plan(rows: list[dict[str, Any]], minimum: int = MIN_FOLD_ROWS) -> dict[str, Any]:
    """Institutions large enough to be held out on their own, plus the pool."""
    grouped = natural_by_institution(rows)
    sizes = {name: len(items) for name, items in grouped.items()}
    folds = [{"name": _slug(name), "institution": name, "test_rows": size,
              "labels": len({row["label"] for row in grouped[name]})}
             for name, size in sorted(sizes.items(), key=lambda item: -item[1])
             if size >= minimum]
    small = {name: size for name, size in sizes.items() if size < minimum}
    return {"institutions": len(sizes), "folds": folds,
            "small_institutions": small,
            "pooled_rows": sum(small.values())}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:40]


def materialize(fold_institution: str, out: Path, mix_root: Path = MIX,
                seed: int = 20261024,
                holdout_source: str | None = None) -> dict[str, Any]:
    """Write train/validation/test for one held-out institution.

    The held-out institution's rows are removed from *every* training family,
    including the composed ones, so no composed row can carry its phrasing.
    Validation is a slice of the remaining institutions, not of the held-out one:
    a temperature fitted on the test institution would be a leak.
    """
    train = load_jsonl(mix_root / "train.jsonl")
    validation = load_jsonl(mix_root / "validation.jsonl")
    if holdout_source:
        # A whole source is the cleanest fold: every school in it is unseen, and
        # every composed row borrowed from it is removed too.
        test = [row for row in load_jsonl(CORPUS)
                if row.get("source") == holdout_source
                and row.get("family") in (None, "natural")]
    else:
        test = [row for row in load_jsonl(CORPUS)
                if institution_of(row) == fold_institution
                and row.get("family") in (None, "natural")]

    def keep(row: dict[str, Any]) -> bool:
        if holdout_source:
            return row.get("source") != holdout_source
        return institution_of(row) != fold_institution

    train = [row for row in train if keep(row)]
    validation = [row for row in validation if keep(row)]
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "heldout_institution": fold_institution,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_labels": dict(Counter(row["label"] for row in test)),
        "train_institutions": len({institution_of(row) for row in train}),
        "test_institutions": len({institution_of(row) for row in test}),
        "leakage": leakage_report(train, test),
        "seed": seed,
    }
    (out / "fold.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def materialize_documents(out: Path, fraction: float = 0.3, seed: int = 20261024,
                          mix_root: Path = MIX) -> dict[str, Any]:
    """Hold out whole syllabus *documents*.

    The source-level fold removes the syllabus family from training entirely and
    measures a different question (can the model read a family it never saw?).
    This fold keeps the family and holds out documents from it, so a training
    recipe can be compared without either side having seen the test text.
    """
    def bucket(document: str) -> int:
        digest = hashlib.sha256(f"{seed}:{document}".encode()).hexdigest()
        return int(digest[:8], 16) % 100

    def document_id(row: dict[str, Any]) -> str:
        return str((row.get("meta") or {}).get("document_id")
                   or row.get("group_id", ""))

    corpus = [row for row in load_jsonl(CORPUS)
              if row.get("source") == "Dev-the-dev91/syllabus-cleaned"]
    heldout = {document_id(row) for row in corpus
               if bucket(document_id(row)) < int(fraction * 100)}
    test = [row for row in corpus if document_id(row) in heldout]

    def keep(row: dict[str, Any]) -> bool:
        return document_id(row) not in heldout

    train = [row for row in load_jsonl(mix_root / "train.jsonl") if keep(row)]
    validation = [row for row in load_jsonl(mix_root / "validation.jsonl") if keep(row)]
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "heldout_documents": len(heldout),
        "train_rows": len(train), "validation_rows": len(validation),
        "test_rows": len(test),
        "train_syllabus_rows": sum(1 for row in train
                                   if row.get("source") == "Dev-the-dev91/syllabus-cleaned"),
        "test_labels": dict(Counter(row["label"] for row in test)),
        "leakage": leakage_report(train, test),
        "seed": seed, "fraction": fraction,
    }
    (out / "fold.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def leakage_report(train: list[dict[str, Any]], test: list[dict[str, Any]],
                   sample: int = 400) -> dict[str, Any]:
    """How much of the test set the training set has already seen."""
    train_prints = {_fingerprint(row["text"]) for row in train}
    exact = sum(1 for row in test if _fingerprint(row["text"]) in train_prints)
    train_headings = {_normal(row.get("heading", "")) for row in train} - {""}
    test_headings = {_normal(row.get("heading", "")) for row in test} - {""}
    sample_rows = test[:sample]
    return {
        "test_rows": len(test),
        "exact_text_overlap": exact,
        "exact_text_overlap_rate": round(exact / max(1, len(test)), 4),
        "heading_overlap": len(test_headings & train_headings),
        "test_headings": len(test_headings),
        "heading_overlap_rate": round(len(test_headings & train_headings)
                                      / max(1, len(test_headings)), 4),
        "near_duplicate_rate_sample": round(near_duplicate_rate(
            [row["text"] for row in train][:20000], [row["text"] for row in sample_rows]), 4),
        "sampled": len(sample_rows),
    }


def diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset diversity, so row count is never mistaken for coverage."""
    grouped = natural_by_institution(rows)
    labels = Counter(row["label"] for row in rows)
    lengths = sorted(len(row["text"].split()) for row in rows)
    prints = Counter(_fingerprint(row["text"]) for row in rows)
    return {
        "rows": len(rows),
        "institutions": len(grouped),
        "institution_sizes": {name: len(items) for name, items in
                              sorted(grouped.items(), key=lambda item: -len(item[1]))},
        "labels": len(labels),
        "rarest_labels": [[label, count] for label, count in labels.most_common()[-10:]],
        "exact_duplicates": sum(count - 1 for count in prints.values() if count > 1),
        "duplicate_rate": round(sum(count - 1 for count in prints.values() if count > 1)
                                / max(1, len(rows)), 4),
        "words_p50": lengths[len(lengths) // 2] if lengths else 0,
        "words_p90": lengths[int(len(lengths) * 0.9)] if lengths else 0,
        "words_max": lengths[-1] if lengths else 0,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Institution-holdout benchmark")
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--materialize", default=None,
                        help="Institution to hold out, exactly as it is named in the data")
    parser.add_argument("--holdout-source", default=None,
                        help="Hold out every row from this source id instead of one institution")
    parser.add_argument("--holdout-documents", action="store_true",
                        help="Hold out whole syllabus documents (recipe comparison)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.corpus)
    if args.holdout_documents:
        out = Path(args.out or "data/processed/transfer/syllabus-docs")
        print(json.dumps(materialize_documents(out), indent=2))
        return
    if args.materialize or args.holdout_source:
        name = args.holdout_source or args.materialize
        out = Path(args.out or f"data/processed/transfer/{_slug(name)}")
        print(json.dumps(materialize(name or "", out, holdout_source=args.holdout_source),
                         indent=2))
        return
    result = {"diversity": diversity(rows), "folds": fold_plan(rows)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
