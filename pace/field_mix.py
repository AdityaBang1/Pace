"""Build a training corpus that teaches the *field*, not an institution's layout.

The recorded limitation of the reference checkpoint is that a model trained on
real catalog and syllabus text learns each source's layout: hold one university
out and accuracy on it collapses from 0.99 to 0.57-0.70. The reason is visible
in the data -- ``Units: 3.0`` only ever appears under Gothenburg's heading in
training, so the model memorises the heading *and* the source together.

This module composes training examples that break that pairing while keeping
every word real:

``natural``
    the row exactly as the institution printed it (heading: value).

``cross``
    a heading from a *different* source printed above the same labelled value,
    e.g. USC's ``Class Name`` above a Gothenburg course title. The heading is
    real text from a real document, the value is real text from a real
    document; only the pairing is new, and the label is guaranteed by the
    label of the value. A model that reads the heading instead of the source
    still gets the right answer, and one that memorised "long prose =
    description" is contradicted.

``heading``
    the heading alone, which is the strongest transferable signal measured on
    real syllabi (0.881 accuracy from the heading only).

``short``
    the heading with only the opening of the value. Long bodies dominate the
    encoder and pull the prediction away from the heading (naive heading+body
    concatenation measured *worse* than the heading alone), so the same label
    is also shown with a truncated body, where the heading has to carry it.

``bare``
    the value alone, kept at a low rate because it is genuinely ambiguous and
    the deployed reader always has a heading.

``value_diversity``
    values the 6,000-per-type cap excluded from the reference corpus, printed
    under their real source heading. The cap kept the classes trainable but
    also narrowed the vocabulary a value can have; a 12-layer encoder can use
    the extra real values without the tail being drowned out.

``sentence``
    a single sentence from a real syllabus section, under that section's real
    heading. There are only ~1,300 labelled syllabus sections, which is the
    scarcest and most transferable part of the corpus; splitting each long
    body into its sentences gives the model many more examples of what
    "credit hours" or "office hours" content actually reads like, instead of
    one long body that outvotes its heading.

``schedule``
    a real day string, a real clock span and a real room, printed together the
    way a printed schedule prints them (``TR 4:00pm - 5:30pm``). The measured
    failure this family exists for: with no training row that starts with a day
    token, bare ``MWF 9:00-9:50 AM`` was read as a course code. Every component
    is a real catalog cell; only the combination is new, and it is labelled by
    what the combination *is* -- a meeting time.

Nothing composed here is ever placed in validation or test: those stay exactly
the real rows, so the reported numbers are earned on real documents.

    python -m pace.field_mix                 # corpus for the full model
    python -m pace.field_mix --fold 0        # train set without fold 0's schools
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .data import save_jsonl
from .field_context import _heading_from_source_field

#: How many rows of each family a single field type may contribute. Field types
#: are wildly unbalanced (84k UW rows, 2 attendance policies), and a corpus
#: dominated by ``course_title`` teaches a model that every short line is a
#: title. The cap is per family so the cross-source signal is not drowned by the
#: source that happens to have the most rows.
DEFAULT_CAPS = {"natural": 6000, "cross": 8000, "heading": 2500, "bare": 1200,
                "value_diversity": 6000, "short": 6000, "sentence": 8000,
                "schedule": 6000}

#: Characters of a value kept in the ``short`` family. The deployed reader
#: passes at most 400 characters to the model, and the failure the extra family
#: teaches against is a long body outvoting its own heading.
SHORT_VALUE_CHARS = 120

_SPACE = re.compile(r"\s+")
_PUNCT_ONLY = re.compile(r"^[\W_]+$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u2018\"'])")
#: Headings that name a container rather than a field; see ``has_real_heading``.
_PLACEHOLDER_HEADINGS = {"section body", "section text", "body", "passage"}
#: A day string as a registrar prints it: MWF, TR, TTh, R, SU.
_DAY_VALUE = re.compile(r"^[MTWRFSUmtwrfSu]{1,6}$")


def _clean(text: str) -> str:
    return _SPACE.sub(" ", (text or "").strip())


def _usable_heading(heading: str) -> bool:
    text = _clean(heading)
    if not text or len(text) > 80 or _PUNCT_ONLY.match(text):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    return True


def _compose(heading: str, value: str) -> str:
    """Print a value under a heading the way ``classify_section`` will feed it."""
    heading, value = _clean(heading), _clean(value)
    if not value:
        return heading
    if value.casefold().startswith(heading.casefold()):
        return value
    return f"{heading}: {value}"


def _school(row: dict[str, Any]) -> str:
    return str((row.get("meta") or {}).get("school") or "")


def _row(text: str, base: dict[str, Any], family: str, label: str) -> dict[str, Any]:
    record = dict(base)
    record["text"] = text
    record["label"] = label
    record["family"] = family
    record.setdefault("provenance", "")
    record["provenance"] = f"{record['provenance']} [{family}]".strip()
    return record


def _heading_pools(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Unique real headings per field type, per source."""
    pools: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        heading = _clean(row.get("heading") or "")
        if not _usable_heading(heading):
            continue
        key = (row["label"], row["source_key"], heading)
        if key in seen:
            continue
        seen.add(key)
        pools[row["label"]][row["source_key"]].append(heading)
    return pools


def _split_groups(split_dir: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        path = split_dir / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing split file {path}")
        with path.open(encoding="utf-8") as handle:
            groups[name] = {json.loads(line)["group_id"] for line in handle if line.strip()}
    return groups


def _balanced(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    """Keep at most ``cap`` rows per field type; drop nothing if under the cap."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label in sorted(by_label):
        items = by_label[label]
        if len(items) > cap:
            rng.shuffle(items)
            items = items[:cap]
        out.extend(items)
    rng.shuffle(out)
    return out


def build_mix(context_rows: list[dict[str, Any]], split_dir: Path, seed: int = 20261019,
              holdout_schools: Iterable[str] = (), caps: dict[str, int] | None = None,
              cross_rate: float = 0.35, bare_rate: float = 0.10, short_rate: float = 0.35,
              extra_values: str | Path | None = None,
              ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    caps = {**DEFAULT_CAPS, **(caps or {})}
    holdout = set(holdout_schools)
    groups = _split_groups(split_dir)
    rng = random.Random(seed)

    def is_train(row: dict[str, Any]) -> bool:
        return row["group_id"] in groups["train"] and _school(row) not in holdout

    # A placeholder is not a heading. Syllabus bodies whose real heading was
    # lost carry ``section body: <paragraph>`` and inherit the label of a heading
    # that is nowhere in the text -- 63 of materials' 185 rows are this, as are
    # 45 of office_hours' and 8 of attendance_policy's 16. Training on them
    # teaches "generic paragraph = that thin field", which is exactly the
    # confusion the long tail shows. They stay in validation and test (the
    # reported numbers are unchanged); they are not training material.
    def has_real_heading(row: dict[str, Any]) -> bool:
        return _clean(row.get("heading") or "").casefold() not in _PLACEHOLDER_HEADINGS

    fit_rows = [row for row in context_rows if is_train(row) and has_real_heading(row)]
    # Headings may only be borrowed from documents the model is allowed to see.
    pools = _heading_pools(fit_rows)

    natural: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    heading_only: list[dict[str, Any]] = []
    short: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    bare: list[dict[str, Any]] = []
    seen_text: set[tuple[str, str]] = set()

    def add(bucket: list[dict[str, Any]], text: str, base: dict[str, Any],
            family: str, label: str) -> None:
        key = (text.casefold(), label)
        if key in seen_text or not text.strip():
            return
        seen_text.add(key)
        bucket.append(_row(text, base, family, label))

    heading_seen: set[tuple[str, str]] = set()
    for row in fit_rows:
        label, value = row["label"], _clean(row.get("bare_text") or "")
        own_heading = _clean(row.get("heading") or "")
        if not value:
            continue
        add(natural, _compose(own_heading, value), row, "natural", label)

        if _usable_heading(own_heading) and (label, own_heading) not in heading_seen:
            heading_seen.add((label, own_heading))
            add(heading_only, own_heading, row, "heading", label)

        if rng.random() < cross_rate:
            donors = [source for source, names in pools.get(label, {}).items()
                      if source != row["source_key"] and names]
            if donors:
                donor = rng.choice(donors)
                borrowed = rng.choice(pools[label][donor])
                add(cross, _compose(borrowed, value), row, "cross", label)

        if rng.random() < short_rate and _usable_heading(own_heading):
            add(short, _compose(own_heading, value[:SHORT_VALUE_CHARS]), row, "short", label)

        if row["group_id"].startswith("syllabus-doc:") and _usable_heading(own_heading):
            parts = [part.strip() for part in _SENTENCE.split(value)
                     if 6 <= len(part.split()) <= 40]
            for part in parts[:3]:
                add(sentences, _compose(own_heading, part), row, "sentence", label)

        if rng.random() < bare_rate:
            add(bare, value, row, "bare", label)

    schedule: list[dict[str, Any]] = []
    # Pools are the bare printed values, not ``heading: value``, because a
    # schedule prints the day and the clock alone.
    days_pool = sorted({_clean(row.get("bare_text") or "") for row in fit_rows
                        if row["label"] == "meeting_days"
                        and _DAY_VALUE.match(_clean(row.get("bare_text") or ""))})
    times_pool = sorted({_clean(row.get("bare_text") or "") for row in fit_rows
                         if row["label"] == "meeting_time"
                         and 4 <= len(_clean(row.get("bare_text") or "")) <= 40
                         and re.search(r"\d", _clean(row.get("bare_text") or ""))})
    rooms_pool = sorted({_clean(row.get("bare_text") or "") for row in fit_rows
                         if row["label"] == "class_meeting_room"
                         and 2 <= len(_clean(row.get("bare_text") or "")) <= 40})
    if days_pool and times_pool:
        rng3 = random.Random(seed + 11)
        base_row = next(row for row in fit_rows if row["label"] == "meeting_time")
        forms = 0
        while forms < caps["schedule"]:
            days, span = rng3.choice(days_pool), rng3.choice(times_pool)
            room = rng3.choice(rooms_pool) if rooms_pool and rng3.random() < 0.4 else ""
            text = f"{days} {span}" + (f", {room}" if room else "")
            before = len(schedule)
            add(schedule, text, base_row, "schedule", "meeting_time")
            if len(schedule) == before and forms > 4 * caps["schedule"]:
                break
            forms += 1
        # The days on their own are a different field: "MWF" is not a time.
        for days in days_pool:
            add(schedule, days, base_row, "schedule", "meeting_days")

    diversity: list[dict[str, Any]] = []
    if extra_values is not None:
        covered = {(row["label"], _clean(row.get("bare_text") or row["text"]).casefold())
                   for row in context_rows}
        forbidden = groups["validation"] | groups["test"]
        candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        with Path(extra_values).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                label, value = row.get("label"), _clean(row.get("text") or "")
                heading = _heading_from_source_field(row.get("source_field", ""))
                if (not label or not value or row["group_id"] in forbidden
                        or _school(row) in holdout or not _usable_heading(heading)):
                    continue
                if (label, value.casefold()) in covered:
                    continue
                candidates[(label, row.get("source_key", ""))].append((row, heading))
        rng2 = random.Random(seed + 7)
        for key in sorted(candidates):
            items = candidates[key]
            rng2.shuffle(items)
            for row, heading in items[:caps["value_diversity"]]:
                add(diversity, _compose(heading, row["text"]), row, "value_diversity", row["label"])

    train = []
    for family, rows, cap_key in (("natural", natural, "natural"), ("cross", cross, "cross"),
                                  ("heading", heading_only, "heading"), ("bare", bare, "bare"),
                                  ("short", short, "short"),
                                  ("sentence", sentences, "sentence"),
                                  ("schedule", schedule, "schedule"),
                                  ("value_diversity", diversity, "value_diversity")):
        kept = _balanced(rows, caps[cap_key], seed + hash(family) % 1000)
        print(json.dumps({"family": family, "built": len(rows), "kept": len(kept)}), flush=True)
        train.extend(kept)
    rng.shuffle(train)

    if holdout:
        # The transfer run: every held-out school is a test document, and the
        # model's own validation slice comes from the schools it may see.
        test = [row for row in context_rows if _school(row) in holdout]
        validation = [row for row in context_rows
                      if row["group_id"] in groups["validation"] and _school(row) not in holdout]
    else:
        validation = [row for row in context_rows if row["group_id"] in groups["validation"]]
        test = [row for row in context_rows if row["group_id"] in groups["test"]]

    splits = {"train": train, "validation": validation, "test": test}
    stats = {
        "train_rows": len(train),
        "train_by_family": dict(Counter(row["family"] for row in train)),
        "train_by_label": dict(sorted(Counter(row["label"] for row in train).items())),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_by_label": dict(sorted(Counter(row["label"] for row in test).items())),
        "holdout_schools": sorted(holdout),
        "holdout_test_schools": dict(sorted(Counter(
            _school(row) for row in test if _school(row) in holdout).items())),
        "caps": caps,
        "cross_rate": cross_rate,
        "bare_rate": bare_rate,
        "short_rate": short_rate,
        "headings_borrowed": len(heading_only),
        "schedule_days": len(days_pool),
        "schedule_times": len(times_pool),
    }
    return splits, stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the semantics-transferring field corpus")
    parser.add_argument("--context", default="data/processed/field_context/corpus.jsonl")
    parser.add_argument("--split-dir", default="data/processed/field_context")
    parser.add_argument("--output", default="data/processed/field_mix")
    parser.add_argument("--fold", type=int, default=None,
                        help="Hold out this cross-institution fold's schools entirely")
    parser.add_argument("--fold-plan", default="data/processed/field/fold_plan.json")
    parser.add_argument("--extra-values", default=None,
                        help="Uncapped corpus to draw additional real values from")
    parser.add_argument("--short-rate", type=float, default=0.35,
                        help="Share of rows also shown with a truncated value")
    parser.add_argument("--value-diversity-cap", type=int, default=6000,
                        help="Values kept per field type from the uncapped corpus")
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args(argv)

    with Path(args.context).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    holdout: list[str] = []
    output = Path(args.output)
    if args.fold is not None:
        plan = json.loads(Path(args.fold_plan).read_text(encoding="utf-8"))
        holdout = list(plan[args.fold])
        output = output / f"fold-{args.fold}"

    splits, stats = build_mix(rows, Path(args.split_dir), seed=args.seed,
                              holdout_schools=holdout, extra_values=args.extra_values,
                              short_rate=args.short_rate,
                              caps={"value_diversity": args.value_diversity_cap})
    output.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        save_jsonl(output / f"{name}.jsonl", part)
    (output / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in {"train_by_family", "train_by_label", "test_by_label"}}, indent=2))


if __name__ == "__main__":
    main()
