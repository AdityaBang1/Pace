"""Build a context-aware corpus: the heading a value sits under, plus the value.

The bare-value task is only partly well posed. "3.0 units" is *credits* only if
you know it came from a credits field; "Lecture" is *class_type* only in a
class-type column. A real reader never sees a value alone -- it sees the
heading above it. This module rebuilds the corpus so every example carries the
natural heading that introduces it, exactly as the institution wrote it:

    "Class Name: Intro to Biology"
    "Units: 3.0 units"
    "Office Hours: Mon 2-4pm"

Nothing is paraphrased. The heading comes from the source document's own
structure (a section heading line) or from the catalog column the value was
published in.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import document_hash, save_jsonl
from .field_corpus import clean_html, detect_document_sections

SYLLABUS_SOURCES = {"Dev-the-dev91/syllabus-cleaned", "INT-Business-Syllabus"}

#: Columns whose source name is a machine qualifier rather than a real heading.
_QUALIFIER = re.compile(r"\s*\((body|as written)\)\s*$", re.I)


def _heading_from_source_field(source_field: str) -> str:
    text = _QUALIFIER.sub("", source_field or "").strip()
    return text


def _syllabus_heading_index() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Index every real syllabus section body by its real heading.

    Returns two lookups: a hash of the body text to the heading, and a
    per-document list of (body, heading) pairs so a *windowed* body can be
    traced back to the section it was cut from.
    """
    by_hash: dict[str, str] = {}
    by_document: dict[str, list[tuple[str, str]]] = defaultdict(list)
    files = [
        Path("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl"),
        Path("data/raw/hf/INT-Business-Syllabus/documents.jsonl"),
    ]
    for path in files:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                document_id = str(record.get("id") or "")
                raw = record.get("text") or ""
                text, _ = clean_html(raw)
                sections = detect_document_sections(text)
                for position, section in enumerate(sections):
                    if section["label"] is None:
                        continue
                    heading = section["text"]
                    body_start = section["end"]
                    body_end = (sections[position + 1]["start"]
                                if position + 1 < len(sections) else len(text))
                    body = re.sub(r"\s+", " ", text[body_start:body_end]).strip()
                    inline = section.get("inline") or ""
                    if inline:
                        body = f"{inline} {body}".strip()
                    if len(body) < 40:
                        continue
                    by_hash[document_hash(body[:2400])] = heading
                    by_hash.setdefault(document_hash(heading), heading)
                    by_document[document_id].append((body, heading))
    return by_hash, by_document


def _heading_for_syllabus_row(row: dict[str, Any], by_hash: dict[str, str],
                              by_document: dict[str, list[tuple[str, str]]]) -> str:
    """Find the heading that introduces this corpus row."""
    heading = by_hash.get(row.get("text_sha256") or "", "")
    if heading:
        return heading
    heading = by_hash.get(document_hash(row["text"]), "")
    if heading:
        return heading
    source_id = str(row.get("source_id") or "")
    document_id = source_id.split(":", 1)[1] if ":" in source_id else ""
    for body, candidate in by_document.get(document_id, []):
        if row["text"] and body.startswith(row["text"][:120]):
            return candidate
    return ""


def build_context_corpus(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a natural heading to every corpus row."""
    by_hash, by_document = _syllabus_heading_index()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    missing_heading = 0
    for row in rows:
        heading = ""
        if row["source"] in SYLLABUS_SOURCES:
            heading = _heading_for_syllabus_row(row, by_hash, by_document)
        if not heading:
            heading = _heading_from_source_field(row.get("source_field", ""))
        if not heading:
            missing_heading += 1
            continue
        text = row["text"]
        if text.casefold().startswith(heading.casefold()):
            context_text = text
        else:
            context_text = f"{heading}: {text}"
        key = (context_text, row["label"])
        if key in seen:
            continue
        seen.add(key)
        record = dict(row)
        record["text"] = context_text
        record["text_sha256"] = document_hash(context_text)
        record["heading"] = heading
        record["bare_text"] = row["text"]
        record["provenance"] = f"{row.get('provenance', '')} (with its natural heading)".strip()
        out.append(record)
    stats = {
        "rows": len(out),
        "rows_without_heading": missing_heading,
        "by_source": dict(Counter(row["source"] for row in out)),
        "by_label": dict(Counter(row["label"] for row in out)),
        "groups": len({row["group_id"] for row in out}),
    }
    return out, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the heading-aware field corpus")
    parser.add_argument("--corpus", default="data/processed/field/corpus.jsonl")
    parser.add_argument("--output", default="data/processed/field_context")
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    context_rows, stats = build_context_corpus(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(output / "corpus.jsonl", context_rows)
    (output / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    from .field_splits import _split_by_group

    splits = _split_by_group(context_rows, args.seed)
    for name, part in splits.items():
        save_jsonl(output / f"{name}.jsonl", part)
    (output / "splits_summary.json").write_text(json.dumps(
        {name: len(part) for name, part in splits.items()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": stats["rows"], "rows_without_heading": stats["rows_without_heading"],
                      "by_source": stats["by_source"],
                      "splits": {name: len(part) for name, part in splits.items()}}, indent=2))


if __name__ == "__main__":
    main()
