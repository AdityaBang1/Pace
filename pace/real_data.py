"""Task examples from real course catalog rows, preserving natural field values."""
from __future__ import annotations
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from .data import document_hash, save_jsonl

FIELD_MAP = {
    "Class Name": "course_title", "Catalogue": "course_description",
    "Prerequisites": "prerequisites", "Restrictions": "restrictions",
    "Units": "credit_units", "Time": "meeting_time", "Days": "meeting_days",
    "Class Type": "class_type",
}
MISSING = {"", "not available", "n/a", "none", "null", "nan"}

def normalize_value(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()

def load_usc_examples(path: str | Path) -> list[dict[str, Any]]:
    """Map natural CSV cell values to field labels without fabricating text segments."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            class_id = (row.get("Class ID") or "").strip()
            match = re.match(r"([A-Za-z]+)\s*[- ]?\s*\d", class_id)
            subject_group = (match.group(1).upper() if match else (row.get("Program Code") or "UNKNOWN").strip().upper())
            source_id = f"usc-2024-spring:{index}:{class_id}"
            for source_field, label in FIELD_MAP.items():
                text = (row.get(source_field) or "").strip()
                if text.casefold() in MISSING:
                    continue
                if source_field == "Class Name": text = re.sub(r"^\s*[^:]{2,16}:\s*", "", text).strip()
                if source_field == "Prerequisites": text = re.sub(r"^\s*Prerequisite:\s*", "", text, flags=re.I).strip()
                if source_field == "Restrictions": text = re.sub(r"^\s*Restriction:\s*", "", text, flags=re.I).strip()
                if not text: continue
                rows.append({"text": text, "task": "field", "label": label,
                             "group_id": f"usc-subject:{subject_group}", "source_id": source_id,
                             "source_field": source_field, "source": "USC/USC-Course-Catalog",
                             "provenance": "2024 Spring term dataset; original CSV cell value",
                             "text_sha256": document_hash(text)})
    return rows

def exclude_intrinsically_ambiguous(rows):
    """Exclude text-only labels whose normalized exact value has conflicting field labels."""
    labels_by_text=defaultdict(set)
    for row in rows: labels_by_text[normalize_value(row['text'])].add(row['label'])
    ambiguous={text:sorted(labels) for text,labels in labels_by_text.items() if len(labels)>1}
    clean=[row for row in rows if normalize_value(row['text']) not in ambiguous]
    excluded=[row for row in rows if normalize_value(row['text']) in ambiguous]
    return clean,ambiguous,excluded

def split_real_groups(rows: list[dict[str, Any]], seed: int = 20261019,
                      fractions=(0.7, 0.15, 0.15), fixed_assignment: dict[str, str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Hold out complete subject prefixes to prevent repeated course leakage."""
    import random
    if abs(sum(fractions) - 1) > 1e-8 or any(x <= 0 for x in fractions):
        raise ValueError("split fractions must be positive and sum to one")
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < 3: raise ValueError("at least three distinct subject groups are required")
    if fixed_assignment is not None:
        missing = set(groups) - set(fixed_assignment)
        if missing: raise ValueError(f"fixed assignment missing groups: {sorted(missing)}")
        if any(fixed_assignment[g] not in {"train", "validation", "test"} for g in groups):
            raise ValueError("fixed assignment values must be train, validation, or test")
        assigned = {g: fixed_assignment[g] for g in groups}
    else:
        random.Random(seed).shuffle(groups)
        first, second = round(len(groups)*fractions[0]), round(len(groups)*(fractions[0]+fractions[1]))
        assigned = {group: "train" for group in groups[:first]}
        assigned.update({group: "validation" for group in groups[first:second]})
        assigned.update({group: "test" for group in groups[second:]})
    return {name: [row for row in rows if assigned[row["group_id"]] == name]
            for name in ("train", "validation", "test")}

def write_real_splits(rows, output: str | Path, seed: int = 20261019):
    splits = split_real_groups(rows, seed)
    for name, part in splits.items(): save_jsonl(Path(output) / f"{name}.jsonl", part)
    return splits
