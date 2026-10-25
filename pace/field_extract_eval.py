"""Score the extractor on institutions it was never tuned on.

The gold here is not hand-written and not model-generated: it is the human
labelled section spans published with the syllabus corpus, one label per real
document fragment. A value counts as correct when the text it was read from
overlaps the gold span for the field it claims — so a plausible number lifted
from the wrong section is a miss, not a near miss.

Documents from the schools used while developing the rules are excluded and the
exclusion is reported, because evaluating on your own debugging set is how a
report lies to itself.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .field_calibration import error_upper_bound
from .field_extract import extract_record
from .field_ledger import suppressed_values, summarize as summarize_losses
from .field_gold import (CONFIDENCE, FLOOR_PRECISION, TARGET_PRECISION, by_field as gold_by_field,
                         load_gold, operating_point, recall as gold_recall,
                         reliability as gold_reliability, score_document, sweep as gold_sweep)

# Extracted value -> the gold labels that can justify it.
TARGETS = {
    "course_title": ["course_title"],
    "course_code": ["course_code"],
    "credits": ["credits"],
    "instructor": ["instructor", "contact_information"],
    "grading": ["grading_policy", "grades"],
    "meetings": ["meeting_time", "meeting_days", "class_meeting_room"],
    "office_hours": ["office_hours"],
    "attendance_policy": ["attendance_policy"],
    "academic_integrity": ["academic_integrity"],
    "disability_services": ["disability_services"],
    "materials": ["materials"],
}

DEFAULT_DEVELOPMENT_SCHOOLS = [
    # Rules were written and debugged on documents from these two schools.
    "University of California, Berkeley",
    "Georgia Institute of Technology",
]

# The held-out documents are split by institution, not by row. Every fix is
# written and measured on DEV. TEST is scored once, after the fixes are frozen,
# and the number it produces is the one reported. The split is part of the
# protocol, not a result: it is written to data/processed/field/extract_split.json
# before any fix, so the reader can check the test institutions were never in the
# development loop.
DEV_SCHOOLS = [
    "Stanford University",
    "Carnegie Mellon University",
    "The University of Texas at Austin",
    "University of California, Los Angeles",
    "University of Wisconsin–Madison",
    "Pennsylvania State University",
    "Purdue University",
    "University of Illinois Urbana-Champaign",
    "University of Michigan",
]
TEST_SCHOOLS = [
    "Massachusetts Institute of Technology",
    "The Ohio State University",
    "University of North Carolina at Chapel Hill",
]
# One test-school document was opened while diagnosing an earlier failure, so the
# test set drops it rather than pretending it is untouched.
TEST_EXCLUDED_DOCUMENTS = {
    "870b9308c514391c907eb01509dabe2a6ec6af2987941e9f8ed2d8f3d12efcb4":
        "inspected while debugging the previous evaluation",
    "283596966bf7b29dc51d836bed5815b282474a31cee2622c446de56d1e1f13e2":
        "a line from this document was printed while assembling an earlier "
        "hand-check list, so it is not untouched",
}


def split_of(school: str | None) -> str | None:
    if school in TEST_SCHOOLS:
        return "test"
    if school in DEV_SCHOOLS:
        return "dev"
    return None


def normalise(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _overlaps(evidence: str, gold: str) -> bool:
    """True when one normalised string contains the other, ignoring noise."""
    left, right = normalise(evidence), normalise(gold)
    if len(left) < 4 or len(right) < 4:
        return False
    return left in right or right in left


def emitted_values(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a record into (field, evidence-text, value, confidence) rows."""
    values: list[dict[str, Any]] = []
    for name, entry in (record.get("course") or {}).items():
        values.append({"field": name, "text": entry["value"], "confidence": entry["confidence"],
                       "evidence": entry["evidence"]["text"]})
    for item in record.get("grading") or []:
        values.append({"field": "grading", "text": item["category"], "weight": item["weight"],
                       "confidence": item["confidence"], "evidence": item["evidence"]["text"]})
    for item in record.get("meetings") or []:
        text = " ".join(str(part) for part in (item.get("days"), item.get("start_time"),
                                               item.get("end_time")) if part)
        values.append({"field": "meetings", "text": text,
                       "days": item.get("days"), "start_time": item.get("start_time"),
                       "end_time": item.get("end_time"), "room": item.get("room"),
                       "confidence": item["confidence"], "evidence": item["evidence"]["text"]})
    for name, entry in (record.get("policies") or {}).items():
        values.append({"field": name, "text": entry["value"], "confidence": entry["confidence"],
                       "evidence": entry["evidence"]["text"]})
    return values


def _percent_rows(gold_rows: list[dict[str, Any]]) -> list[str]:
    """Grading gold rows that carry a number, i.e. the ones a weight can match."""
    return [str(row["text"]) for row in gold_rows
            if row["label"] in TARGETS["grading"] and "%" in str(row["text"])]


def _weight_in(weight: Any, text: str) -> bool:
    return re.search(rf"\b{int(weight)}\s*%", text) is not None


def evaluate_document(record: dict[str, Any], gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_by_label: dict[str, list[str]] = defaultdict(list)
    for row in gold_rows:
        gold_by_label[row["label"]].append(row["text"])
    percent_rows = _percent_rows(gold_rows)
    judged: list[dict[str, Any]] = []
    for value in emitted_values(record):
        # A grading category is scored on the weight it carries: the published gold
        # for this field is a list of percentages, so overlap with a category name
        # is not something that gold can answer.
        if value["field"] == "grading":
            if not percent_rows:
                judged.append({**value, "judged": False, "matched": False})
                continue
            matched = any(_weight_in(value["weight"], text) for text in percent_rows)
            judged.append({**value, "judged": True, "matched": matched})
            continue
        labels = TARGETS.get(value["field"], [])
        candidates = [text for label in labels for text in gold_by_label.get(label, [])]
        if not candidates:
            judged.append({**value, "judged": False, "matched": False})
            continue
        matched = any(_overlaps(value["evidence"], text) or _overlaps(value["text"], text)
                      for text in candidates)
        judged.append({**value, "judged": True, "matched": matched})
    return judged


def recall_by_field(gold_rows: list[dict[str, Any]], judged: list[dict[str, Any]],
                    record: dict[str, Any] | None = None) -> dict[str, tuple[int, int]]:
    """Gold rows with at least one matching extracted value, per field."""
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    weights = [row["weight"] for row in (record or {}).get("grading") or []]
    for row in gold_rows:
        labels = [label for label, targets in TARGETS.items() if row["label"] in targets]
        if not labels:
            continue
        text = str(row["text"])
        if "grading" in labels and "%" in text:
            # One weight-shaped gold row is one slot; a category list of prose is not.
            hits["grading"][1] += 1
            hits["grading"][0] += int(any(_weight_in(weight, text) for weight in weights))
            labels = [label for label in labels if label != "grading"]
            if not labels:
                continue
        covered = any(item["matched"] and item["field"] in labels for item in judged)
        for label in labels:
            hits[label][1] += 1
            if covered:
                hits[label][0] += 1
    return {field: (hit, total) for field, (hit, total) in hits.items()}


DAY_NAMES = {
    "mo": "Monday", "mon": "Monday", "monday": "Monday",
    "tu": "Tuesday", "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
    "we": "Wednesday", "wed": "Wednesday", "wednesday": "Wednesday",
    "th": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
    "thursday": "Thursday",
    "fr": "Friday", "fri": "Friday", "friday": "Friday",
    "sa": "Saturday", "sat": "Saturday", "saturday": "Saturday",
    "su": "Sunday", "sun": "Sunday", "sunday": "Sunday",
}


def canonical_days(days: Any) -> tuple:
    return tuple(sorted({DAY_NAMES.get(str(day).strip().lower(), str(day)) for day in (days or [])}))


def meeting_key(item: dict[str, Any]) -> tuple:
    return canonical_days(item.get("days")), item.get("start") or item.get("start_time"), \
        item.get("end") or item.get("end_time")


def room_key(value: Any) -> str:
    """Room comparison ignores spacing: "1351Chemistry" is "1351 Chemistry"."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def meeting_matches(gold: dict[str, Any], emitted: dict[str, Any]) -> bool:
    if meeting_key(gold) != meeting_key(emitted):
        return False
    gold_room = room_key(gold.get("room"))
    if not gold_room:
        return True
    room = room_key(emitted.get("room"))
    return bool(room) and (gold_room in room or room in gold_room)


def same_number(got: Any, expected: Any) -> bool:
    try:
        return abs(float(got) - float(expected)) < 1e-6
    except (TypeError, ValueError):
        return normalise(got) == normalise(expected)


def score_hand_gold(records: dict[str, dict[str, Any]],
                    hand: dict[str, Any]) -> dict[str, Any]:
    """Exact-slot scoring on the documents a person read field by field.

    Reported both whole and in parts, because "0 of 11 meetings" hides whether
    the days, the clock times or the room is what went missing.
    """
    meeting_gold = meeting_exact = meeting_emitted = meeting_emitted_matched = 0
    day_gold = day_ok = time_gold = time_ok = room_gold = room_ok = 0
    credit_gold = credit_correct = 0
    scored_documents = 0
    failures: list[dict[str, Any]] = []
    for document in hand["documents"]:
        record = records.get(document["id"])
        if record is None:
            continue
        scored_documents += 1
        emitted = record.get("meetings") or []
        gold_meetings = document.get("meetings") or []
        for gold in gold_meetings:
            meeting_gold += 1
            aligned = [item for item in emitted
                       if canonical_days(item.get("days")) == canonical_days(gold.get("days"))]
            if any(meeting_matches(gold, item) for item in emitted):
                meeting_exact += 1
            else:
                failures.append({"id": document["id"][:8], "field": "meetings",
                                 "expected": gold, "got": emitted})
            day_gold += 1
            day_ok += int(bool(aligned))
            if gold.get("start"):
                time_gold += 1
                time_ok += int(any(meeting_key(gold)[1:] == meeting_key(item)[1:] for item in aligned))
            if gold.get("room"):
                room_gold += 1
                room_ok += int(any(meeting_matches(gold, item) for item in aligned))
        for item in emitted:
            meeting_emitted += 1
            if any(meeting_matches(gold, item) for gold in gold_meetings):
                meeting_emitted_matched += 1
        if "credits" in document:
            credit_gold += 1
            value = (record.get("course") or {}).get("credits")
            got = value["value"] if value else None
            if same_number(got, document["credits"]):
                credit_correct += 1
            else:
                failures.append({"id": document["id"][:8], "field": "credits",
                                 "expected": document["credits"], "got": got})
    def ratio(hit: int, total: int) -> float | None:
        return hit / total if total else None

    return {
        "documents": scored_documents,
        "meetings": {
            "gold_slots": meeting_gold,
            "exact": meeting_exact,
            "accuracy": ratio(meeting_exact, meeting_gold),
            "days_correct": day_ok,
            "days_scored": day_gold,
            "times_correct": time_ok,
            "times_scored": time_gold,
            "rooms_correct": room_ok,
            "rooms_scored": room_gold,
            "emitted": meeting_emitted,
            "emitted_matched": meeting_emitted_matched,
            "precision": ratio(meeting_emitted_matched, meeting_emitted),
        },
        "credits": {
            "gold_documents": credit_gold,
            "correct": credit_correct,
            "accuracy": ratio(credit_correct, credit_gold),
        },
        "failures": failures,
    }


def _gold_set_summary(field_gold: str, extra_documents: str | None) -> dict[str, Any]:
    """Where the value-level gold documents came from, school by school."""
    payload = json.loads(Path(field_gold).read_text(encoding="utf-8"))
    hosts: dict[str, Counter] = defaultdict(Counter)
    if extra_documents and Path(extra_documents).exists():
        with Path(extra_documents).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                host = urlsplit(record.get("url") or "").netloc
                if host:
                    hosts[record["school"]][host] += 1
    by_school: dict[str, dict[str, Any]] = {}
    for document in payload["documents"]:
        entry = by_school.setdefault(document["school"], {"documents": 0, "listing": ""})
        entry["documents"] += 1
    for school, entry in by_school.items():
        if hosts.get(school):
            entry["listing"] = f"`{hosts[school].most_common(1)[0][0]}` course listings"
    return {"documents": len(payload["documents"]), "schools": sorted(by_school),
            "by_school": dict(sorted(by_school.items()))}


def write_split(ingested: str, fold_plan: str, path: Path,
                extra_documents: str | None = None,
                field_gold: str | None = None) -> dict[str, Any]:
    """Record the development/test split before any fix is tuned against it."""
    holdout = {school for fold in json.loads(Path(fold_plan).read_text(encoding="utf-8"))
               for school in fold}
    counts: dict[str, Counter] = {"dev": Counter(), "test": Counter()}
    excluded: list[dict[str, str]] = []
    with Path(ingested).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            school = record.get("school")
            if school not in holdout or school in DEFAULT_DEVELOPMENT_SCHOOLS:
                continue
            part = split_of(school)
            if part is None:
                continue
            if record["id"] in TEST_EXCLUDED_DOCUMENTS:
                excluded.append({"school": school, "id": record["id"],
                                 "reason": TEST_EXCLUDED_DOCUMENTS[record["id"]]})
                continue
            counts[part][school] += 1
    payload = {
        "protocol": [
            "Institutions are assigned to dev or test before any fix is written.",
            "Every rule change is measured on dev only.",
            "Test is scored once, after the rules are frozen, and that number is the one reported.",
            "Documents from the schools used to write the very first rules are excluded entirely.",
        ],
        "development_schools": sorted(DEV_SCHOOLS),
        "test_schools": sorted(TEST_SCHOOLS),
        "documents": {part: dict(sorted(counts[part].items())) for part in ("dev", "test")},
        "document_counts": {"dev": sum(counts["dev"].values()),
                            "test": sum(counts["test"].values())},
        "excluded_documents": excluded,
        "development_schools_removed_entirely": sorted(DEFAULT_DEVELOPMENT_SCHOOLS),
    }
    if field_gold and Path(field_gold).exists():
        summary = _gold_set_summary(field_gold, extra_documents)
        payload["test_gold_documents"] = summary
        payload["test_schools"] = sorted(set(payload["test_schools"]) | set(summary["schools"]))
        payload["document_counts"]["test"] += summary["documents"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["document_counts"], indent=2))
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _dev_sweep_sentence(dev: dict[str, Any]) -> str:
    """Why the default threshold is where it is, from the development sweep."""
    rows = [row for row in dev.get("sweep") or [] if row.get("precision_lower_95") is not None]
    if not rows:
        return ("No threshold on this split has a measurable lower bound, so the default "
                "stays at `min_confidence=0`.")
    default = min(rows, key=lambda row: row["threshold"])
    best = max(rows, key=lambda row: row["precision_lower_95"])
    if best["threshold"] <= default["threshold"] + 1e-9:
        return (f"No threshold on this split certifies the target. The strongest bound is "
                f"{_fmt(best['precision_lower_95'])} at `min_confidence="
                f"{best['threshold']:.3f}`, which is the shipped default "
                f"({_fmt(default['precision'])} precision at "
                f"{default['coverage']:.0%} coverage), so no setting measured here beats "
                f"keeping every value.")
    return (f"No threshold on this split certifies the target, and the best-looking "
            f"development setting (`{best['threshold']:.3f}`, bound "
            f"{_fmt(best['precision_lower_95'])}, coverage {best['coverage']:.0%}) is "
            f"inside overlapping bounds with the default's "
            f"{_fmt(default['precision_lower_95'])}; the value-level gold measures the "
            f"same setting before it is adopted, and the default stays at "
            f"`min_confidence={default['threshold']:.1f}`.")


def _coverage_table(result: dict[str, Any]) -> list[str]:
    """What was emitted against what was read and not emitted, per field."""
    coverage = result.get("coverage") or {}
    if not coverage:
        return []
    lines = ["| Field | Emitted | Suppressed | Gold items | Found | Recall |",
             "|---|---:|---:|---:|---:|---:|"]
    for field, stats in coverage.items():
        recall = stats.get("recall")
        lines.append(f"| `{field}` | {stats['emitted']} | {stats['suppressed']} "
                     f"| {stats['gold_items']} | {stats['found']} "
                     f"| {'-' if recall is None else _fmt(recall)} |")
    # Every cause, per field, not a top-two sample: a cause left out of this table
    # is a reason a reader cannot audit, and the abstentions are exactly the kind
    # of low-volume cause a "top two" cut would hide.
    by_field = result.get("coverage_reasons_by_field") or {}
    rows = [(field, reason, count)
            for field in coverage
            for reason, count in sorted((by_field.get(field) or {}).items(),
                                        key=lambda item: (-item[1], item[0]))]
    if rows:
        lines += [
            "",
            "#### Where the losses are, by cause",
            "",
            "Every suppressed value, folded to the rule that produced it; the counts for a "
            "field add up to that field's suppressed total above. A cause beginning "
            "`no course name:` is the field's own abstention: it emitted nothing, and "
            "this is the decision and the printed line it recorded.",
            "",
            "| Field | Cause | Suppressed |",
            "|---|---|---:|",
        ]
        lines += [f"| `{field}` | {reason} | {count} |" for field, reason, count in rows]
    return lines


def _gold_section(result: dict[str, Any]) -> list[str]:
    """The value-level gold block: every named field, judged or not."""
    block = result.get("field_gold")
    if not block:
        return []
    overall = block["overall"]
    lines = [
        f"## Value-level gold ({block['documents']} documents, "
        f"{len(block['schools'])} institutions)",
        "",
        "Gold here was read field by field from the raw document: every value it states, "
        "and every named field it does not state. An emitted value on a field the document "
        "never states is a false positive, which is the failure section-span gold cannot "
        "see. Emitted values on fields this gold does not label (policy prose such as "
        "`materials`, `academic_integrity`, `disability_services`, and grading printed in "
        "points rather than percentages) are counted as unjudged, never as correct.",
        "",
        "| Field | Emitted | Judged | Correct | Precision | Gold items | Found |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, stats in block["per_field"].items():
        recall = block["recall"].get(field)
        lines.append(f"| `{field}` | {stats['emitted']} | {stats['judged']} | {stats['matched']} "
                     f"| {_fmt(stats['precision'])} | "
                     f"{'-' if recall is None else recall[1]} | "
                     f"{'-' if recall is None else recall[0]} |")
    lines += [
        "",
        f"Overall: **{overall['matched']}/{overall['judged']}** judged values correct = "
        f"precision **{_fmt(overall['precision'])}**, one-sided 95% lower bound "
        f"**{_fmt(overall['precision_lower_95'])}**; {overall['unjudged']} emitted values "
        f"had no gold either way.",
        "",
        "## Reliability: confidence against measured precision",
        "",
        "| Confidence band | Values | Correct | Precision | 95% lower bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for band in block["reliability"]:
        lines.append(f"| {band['low']:.2f}-{band['high']:.2f} | {band['values']} "
                     f"| {band['correct']} | {_fmt(band['precision'])} "
                     f"| {_fmt(band['precision_lower_95'])} |")
    filled = [band for band in block["reliability"] if band["values"]]
    if len(filled) >= 2:
        ordered = ", ".join(
            f"{band['low']:.2f}-{band['high']:.2f}: {band['correct']}/{band['values']} "
            f"= {_fmt(band['precision'])}" for band in filled)
        breaks = [(low, high) for low, high in zip(filled, filled[1:])
                  if high["precision"] < low["precision"]]
        if not breaks:
            lines += [
                "",
                f"Read the bands in order — {ordered} — and precision *rises* as the "
                "confidence rises: the percentage now orders correctness on these "
                "documents, so a threshold can select values a reader can trust.",
            ]
        else:
            detail = "; ".join(
                f"{high['low']:.2f}-{high['high']:.2f} "
                f"({high['correct']}/{high['values']} = {_fmt(high['precision'])}) sits below "
                f"{low['low']:.2f}-{low['high']:.2f} "
                f"({low['correct']}/{low['values']} = {_fmt(low['precision'])})"
                for low, high in breaks)
            involved = [band for pair in breaks for band in pair]
            thin = [band for band in involved if band["values"] < 5]
            wrong = [value for value in (block.get("values") or [])
                     if value.get("judged") and not value.get("matched")]
            counts: dict[str, int] = {}
            for value in wrong:
                label = f"`{value['field']}` at {value['confidence']:.2f}"
                counts[label] = counts.get(label, 0) + 1
            named = ", ".join(
                label + (f" x{n}" if n > 1 else "") for label, n in counts.items())
            lines += [
                "",
                f"Read the bands in order — {ordered}. The table is **not monotone**: "
                f"{detail}. "
                + ("The band(s) in that reversal hold fewer than five values each, so one "
                   "value moves the precision by 0.2 or more; no threshold can be chosen "
                   "from a difference that small."
                   if thin else
                   "Those bands hold enough values that the reversal is not only "
                   "small-sample noise. The errors the higher band keeps are "
                   + (named or "none") +
                   ", so the remaining reversal comes from values the higher "
                   "confidence cannot separate from the correct ones."),
            ]
    lines += [
        "",
        "## What a default threshold would emit",
        "",
        "Coverage is measured against the values this gold judges, so it answers \"how much "
        "of the checkable output survives at this threshold\", not \"how much of the whole "
        "record\".",
        "",
        "| `min_confidence` | Values kept | Coverage | Precision | 95% lower bound | Errors kept |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block["sweep"]:
        lines.append(f"| {row['threshold']:.3f} | {row['accepted']} | {row['coverage']:.2f} "
                     f"| {_fmt(row['precision'])} | {_fmt(row['precision_lower_95'])} "
                     f"| {row['errors']} |")
    point = block["operating_point"]
    lines.append("")
    if point["threshold"] is not None:
        lines.append(
            f"Lowest threshold whose certified precision reaches {point['target']:.2f}: "
            f"**{point['threshold']:.3f}**, keeping {point['coverage']:.0%} of judged values "
            f"at precision {_fmt(point['precision'])} "
            f"(95% lower bound {_fmt(point['precision_lower_95'])}).")
    else:
        best = point.get("best") or {}
        lines.append(
            f"No threshold certifies precision {point['target']:.2f} at 95% confidence on "
            f"this sample. The strongest bound any threshold reaches is "
            f"{_fmt(best.get('precision_lower_95'))} at `min_confidence="
            f"{best.get('threshold')}` (precision {_fmt(best.get('precision'))}, coverage "
            f"{best.get('coverage', 0):.0%}). The claim this data supports is therefore "
            f"weaker than the one asked for, and that is stated rather than rounded up.")
    lines.append("")
    return lines


def write_markdown(result: dict[str, Any], path: Path, split: str = "all") -> None:
    """A table a person can read, generated from the same run as the JSON."""
    overall = result["overall"]
    title = {"dev": "Extraction accuracy, development split",
             "test": "Extraction accuracy, test split (scored once)",
             "all": "Extraction accuracy on institutions the rules never saw"}[split]
    lines = [
        f"# {title}",
        "",
        "Generated by `python -m pace.field_extract_eval` from "
        f"`{overall['documents']}` held-out documents across "
        f"{len(overall['schools'])} institutions, at `min_confidence="
        f"{overall['min_confidence']}`.",
        "",
        "Documents from " + ", ".join(f"`{school}`" for school in overall["excluded_schools"])
        + " are excluded: the field rules were written while looking at them, so scoring "
        "there would measure memory, not reading.",
        "",
    ]
    lines += _gold_section(result)
    lines += [
        "## Section-span accuracy (the wider, coarser gold)",
        "",
        "A value counts as correct here when the line it was read from overlaps the gold "
        "span for the field it claims, so a plausible number lifted from the wrong section "
        "scores as a miss. `judged` is the count of emitted values whose field has a span "
        "in that document; `emitted` - `judged` is what this gold cannot speak to.",
        "",
        "| Field | Emitted | Judged | Matched | Precision | Recall | Gold rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, stats in result["per_field"].items():
        lines.append(f"| `{field}` | {stats['emitted']} | {stats['judged']} | {stats['matched']} "
                     f"| {_fmt(stats['precision'])} | {_fmt(stats['recall'])} "
                     f"| {stats['gold_rows']} |")
    bound = overall["error_upper_bound_95"]
    lines += [
        "",
        f"Overall: **{overall['matched']}/{overall['judged_values']}** judged values correct "
        f"= precision **{_fmt(overall['precision'])}**, one-sided 95% lower bound "
        f"**{_fmt(None if bound is None else 1 - bound)}**; "
        f"{overall['unjudged_values']} emitted values could not be judged against this gold.",
        "Recall is over gold rows, not over fields: a `grading_policy` span of prose counts "
        "against a single extracted category, so the recall column reads low for sections "
        "that are paragraphs rather than slots.",
    ]
    hand = result.get("hand_checked")
    if hand and (hand["meetings"]["gold_slots"] or hand["credits"]["gold_documents"]):
        meetings, credits = hand["meetings"], hand["credits"]
        lines += [
            "",
            "## Hand-checked meetings and credits",
            "",
            "The published span gold has no meeting-time rows in this sample, so these were "
            f"read by hand off {hand['documents']} documents of this split.",
            "",
            "| Field | Gold | Correct | Accuracy |",
            "|---|---:|---:|---:|",
        ]
        if meetings["gold_slots"]:
            lines += [
                f"| `meetings` (whole slot) | {meetings['gold_slots']} | {meetings['exact']} "
                f"| {_fmt(meetings['accuracy'])} |",
                f"| `meetings` (days) | {meetings['days_scored']} | {meetings['days_correct']} "
                f"| {_fmt(meetings['days_correct'] / meetings['days_scored'])} |",
                f"| `meetings` (time) | {meetings['times_scored']} | {meetings['times_correct']} "
                f"| {_fmt(meetings['times_correct'] / meetings['times_scored'])} |",
                f"| `meetings` (room) | {meetings['rooms_scored']} | {meetings['rooms_correct']} "
                f"| {_fmt(meetings['rooms_correct'] / meetings['rooms_scored'])} |",
            ]
        if credits["gold_documents"]:
            lines.append(f"| `credits` | {credits['gold_documents']} | {credits['correct']} "
                         f"| {_fmt(credits['accuracy'])} |")
        lines.append("")
    grounding = result.get("grounding")
    if grounding:
        lines += [
            "## Grounding",
            "",
            "A title is only accepted when the string it returns occurs verbatim in the "
            "document, so a title is never assembled from a file name or a guess.",
            "",
            "| Emitted titles | Verbatim in the document | Rate |",
            "|---:|---:|---:|",
            f"| {grounding['titles_emitted']} | {grounding['titles_grounded']} "
            f"| {grounding['rate']:.3f} |",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _sweep_table(result: dict[str, Any]) -> list[str]:
    rows = ["| `min_confidence` | Values kept | Coverage | Precision | 95% lower bound |",
            "|---:|---:|---:|---:|---:|"]
    for row in result.get("sweep") or []:
        rows.append(f"| {row['threshold']:.3f} | {row['accepted']} | {row['coverage']:.2f} "
                    f"| {_fmt(row['precision'])} | {_fmt(row['precision_lower_95'])} |")
    point = result.get("operating_point") or {}
    if point.get("threshold") is not None:
        rows += ["", f"Certified target at `min_confidence={point['threshold']:.3f}` "
                     f"(coverage {point['coverage']:.0%})."]
    elif point:
        best = point.get("best") or {}
        rows += ["", f"No threshold certifies {point.get('target', TARGET_PRECISION):.2f}; the "
                     f"strongest certified bound is "
                     f"{_fmt(best.get('precision_lower_95'))} at "
                     f"`min_confidence={best.get('threshold')}`."]
    return rows


def write_protocol_report(split: dict[str, Any], dev: dict[str, Any], test: dict[str, Any],
                          path: Path, known_failures: str | None = None) -> None:
    """The published report: the split, what tuning produced, and the one test number."""
    def field_table(result: dict[str, Any]) -> list[str]:
        rows = ["| Field | Emitted | Judged | Correct | Precision | Recall | Gold rows |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for field, stats in result["per_field"].items():
            rows.append(f"| `{field}` | {stats['emitted']} | {stats['judged']} | "
                        f"{stats['matched']} | {_fmt(stats['precision'])} | "
                        f"{_fmt(stats['recall'])} | {stats['gold_rows']} |")
        return rows

    def span_headline(result: dict[str, Any]) -> str:
        overall = result["overall"]
        bound = overall["error_upper_bound_95"]
        return (f"**{overall['matched']}/{overall['judged_values']} judged values correct = "
                f"precision {_fmt(overall['precision'])}, 95% lower bound "
                f"{_fmt(None if bound is None else 1 - bound)}** on {overall['documents']} "
                f"documents from {len(overall['schools'])} institutions; "
                f"{overall['unjudged_values']} emitted values had no span to check against.")

    def gold_headline(result: dict[str, Any]) -> str:
        block = result.get("field_gold") or {}
        overall = block.get("overall") or {}
        if not overall:
            return ""
        return (f"**{overall['matched']}/{overall['judged']} judged values correct = precision "
                f"{_fmt(overall['precision'])}, 95% lower bound "
                f"{_fmt(overall['precision_lower_95'])}** on {block['documents']} documents "
                f"from {len(block['schools'])} institutions; {overall['unjudged']} emitted "
                f"values had no gold either way.")

    record = {}
    if known_failures and Path(known_failures).exists():
        record = json.loads(Path(known_failures).read_text(encoding="utf-8"))
    rule_notes = record.get("rule_decisions") or []
    failures = record.get("failures") or []

    test_gold = test.get("field_gold") or {}
    final = test["overall"]
    lines = [
        "# Extraction accuracy: protocol, development split, and the recorded test number",
        "",
        "Generated by `python -m pace.field_extract_eval --protocol-report` from the two "
        "recorded runs. The split was written to "
        "[extract_split.json](data/processed/field/extract_split.json) before any fix was "
        "written; every rule change was measured on the development split, and the test "
        "split was scored once, at the end. Rules added in later passes were measured "
        "only on development and the artifact was not re-scored, so the table below "
        "describes that recorded run, not today's code.",
        "",
        "## The recorded test number (one touch)",
        "",
    ]
    if test_gold:
        lines += [
            gold_headline(test),
            "",
            "This is the number to quote for that run. It comes from gold that was read field by field off "
            "documents that were not in the corpus while any rule was written: two "
            "institutions are new, and every named field is judged on every document, "
            "including the fields the document does not state.",
            "",
            "The section-span gold further down covers more documents but speaks to fewer "
            "values, and it cannot see an invented value on a field the document never "
            "states — so it is reported beside this number, never instead of it.",
        ]
        for note in rule_notes:
            lines += ["", f"One gold rule matters for reading the number: {note}"]
    else:
        lines += [span_headline(test), ""]
    lines += [
        "",
        "## The split",
        "",
        "| Split | Institutions | Documents | Role |",
        "|---|---:|---:|---|",
        f"| development | {len(split['development_schools'])} | {split['document_counts']['dev']} "
        "| every rule change was measured here |",
        f"| test | {len(split['test_schools'])} | {split['document_counts']['test']} "
        "| scored once, after the rules were frozen |",
        f"| removed entirely | {len(split['development_schools_removed_entirely'])} | - "
        "| the first rules were written while reading these, so they are neither |",
        "",
        "Development: " + ", ".join(f"`{school}`" for school in split["development_schools"]) + ".",
        "",
        "Test: " + ", ".join(f"`{school}`" for school in split["test_schools"]) + ".",
        "",
    ]
    if split.get("excluded_documents"):
        lines += ["Documents removed from the test split because they had already been looked "
                  "at:", ""]
        for item in split["excluded_documents"]:
            lines.append(f"- `{item['id'][:8]}` ({item['school']}): {item['reason']}")
        lines.append("")
    if split.get("test_gold_documents"):
        gold_split = split["test_gold_documents"]
        lines += [
            "### The value-level gold set",
            "",
            f"{gold_split['documents']} documents from "
            + ", ".join(f"`{school}`" for school in gold_split["schools"])
            + ", downloaded from each institution's published course listing before the "
            "extractor was ever run on them. Candidates were sampled evenly from those "
            "listings and kept by a readability gate only (a real text layer, 1,200-20,000 "
            "characters), never by how well the extractor later did.",
            "",
            "| Institution | Documents | Listing |",
            "|---|---:|---|",
        ]
        for school, entry in gold_split["by_school"].items():
            lines.append(f"| {school} | {entry['documents']} | "
                         f"{entry.get('listing', '')} |")
        lines.append("")
    lines += [
        "## Development split (used for tuning — not a result)",
        "",
        span_headline(dev),
        "",
        *field_table(dev),
        "",
        "### The coverage ledger (development split)",
        "",
        "The record now keeps every value it read and did not emit: a policy value",
        "overwritten by a later section, a nested sub-weight, a duplicate row, a row",
        "outside the subset that sums to 100, a title ranked below the chosen line, a",
        "code that looked like a room, a candidate below the threshold. Nothing is",
        "capped: a rejected candidate is listed however many there are, and a line that",
        "reads like a schedule or a grading row but produces no value is listed with",
        "the reason it produced none. That is why `meetings` is larger than the other",
        "fields: most of its losses are exam dates and office-hours lines, which are",
        "values of another kind rather than class times. This table counts the losses",
        "against the gold, so precision can be read next to what it cost. The test split",
        "was **not** scored again for this ledger: the one-touch rule matters more than",
        "a symmetrical table, and the test numbers above stand.",
        "",
        *_coverage_table(dev),
        "",
        "### Development sweep (the basis for the default threshold)",
        "",
        "The span gold judges a grading value by the weight it carries, not by the category "
        "it names, so this sweep is generous to grading and its bands mix fields with very "
        "different base rates. It is shown because the shipped default is decided here. "
        + _dev_sweep_sentence(dev)
        + "",
        "",
        *_sweep_table(dev),
        "",
        "## Test split, section-span gold (scored once)",
        "",
        span_headline(test),
        "",
        *field_table(test),
        "",
    ]
    lines += _gold_section(test)
    previous_path = Path("artifacts/field-extract/extract_eval_test.previous.json")
    previous = (json.loads(previous_path.read_text(encoding="utf-8"))
                if previous_path.exists() else None)
    if test_gold and previous and previous.get("field_gold"):
        old_gold = previous["field_gold"]["overall"]
        old_span = previous["overall"]
        new_span = test["overall"]
        old_bands = [band for band in previous["field_gold"]["reliability"] if band["values"]]
        new_bands = [band for band in test_gold["reliability"] if band["values"]]
        old_bound = (None if old_span["error_upper_bound_95"] is None
                     else 1 - old_span["error_upper_bound_95"])
        new_bound = (None if new_span["error_upper_bound_95"] is None
                     else 1 - new_span["error_upper_bound_95"])

        def _band_summary(bands):
            if not bands:
                return "no bands"
            return ", ".join(f"{band['low']:.2f}-{band['high']:.2f} "
                             f"{_fmt(band['precision'])}" for band in bands)
        lines += [
            "## What this pass changed, measured on the same two runs",
            "",
            f"- Value-level gold: **{old_gold['matched']}/{old_gold['judged']} = "
            f"{_fmt(old_gold['precision'])}** (bound {_fmt(old_gold['precision_lower_95'])}) "
            f"before, **{test_gold['overall']['matched']}/{test_gold['overall']['judged']} = "
            f"{_fmt(test_gold['overall']['precision'])}** "
            f"(bound {_fmt(test_gold['overall']['precision_lower_95'])}) now; "
            f"{old_gold['unjudged']} emitted values had no gold before, "
            f"{test_gold['overall']['unjudged']} now.",
            f"- Section-span gold: **{old_span['matched']}/{old_span['judged_values']} = "
            f"{_fmt(old_span['precision'])}** "
            f"(bound {_fmt(old_bound)}) "
            f"before, **{new_span['matched']}/{new_span['judged_values']} = "
            f"{_fmt(new_span['precision'])}** "
            f"(bound {_fmt(new_bound)}) "
            f"now. It fell, and the honest reason is coverage: the stricter rules emit "
            f"{old_span['judged_values'] + old_span['unjudged_values']} values on these "
            f"documents before and "
            f"{new_span['judged_values'] + new_span['unjudged_values']} now, so only "
            f"{new_span['judged_values']} values still have a span to be judged against. "
            f"A 4/7 sample cannot carry a precise estimate either way.",
            f"- Confidence bands (value-level gold): {_band_summary(old_bands)} before, "
            f"{_band_summary(new_bands)} now. The highest band no longer sits below the "
            f"lowest one, but the table is still not monotone; the reliability section "
            f"names the values that keep it from being monotone.",
            "",
        ]
    point = (test.get("field_gold") or {}).get("operating_point") or {}
    if test_gold:
        lines += [
            "## The claim, in the only form this sample supports",
            "",
            f"- **Target claim:** precision {point.get('target', TARGET_PRECISION):.2f} "
            f"certified at {CONFIDENCE:.0%} confidence — a false-positive rate the data can "
            f"rule out at 5% or better.",
            f"- **What is certified at the shipped default "
            f"(`min_confidence={final['min_confidence']}`):** precision no lower than "
            f"**{_fmt(test_gold['overall']['precision_lower_95'])}** on "
            f"{test_gold['overall']['judged']} judged values.",
            "- **What is not:** the target is not certified at *any* threshold in the table.",
            "",
        ]
        sweep_rows = test_gold.get("sweep") or []
        default_row = next((row for row in sweep_rows
                            if abs(row["threshold"] - final["min_confidence"]) < 1e-9), None)
        if default_row is not None:
            above = [row for row in sweep_rows if row["threshold"] > 0]
            better = [row for row in above if (row.get("precision_lower_95") or 0)
                      > (default_row.get("precision_lower_95") or 0) + 1e-12]
            if not above:
                verdict = "The table has no positive threshold to compare with the default."
            elif not better:
                verdict = ("Every threshold above zero measures a lower bound no better "
                           "than the default's, so no operating point can be set to make "
                           "the target true.")
            elif len(better) == len(above):
                verdict = ("Every threshold above zero measures a *better* lower bound than "
                           "the default, so the default is not the best operating point "
                           "this data can see.")
            else:
                best_above = max(above, key=lambda row: row.get("precision_lower_95") or 0)
                verdict = (f"{len(better)} of {len(above)} positive thresholds measure a "
                           f"better lower bound than the default; the strongest is "
                           f"`min_confidence={best_above['threshold']:.3f}` at "
                           f"{_fmt(best_above.get('precision'))} precision with bound "
                           f"{_fmt(best_above.get('precision_lower_95'))}, against the "
                           f"default's {_fmt(default_row.get('precision'))} and "
                           f"{_fmt(default_row.get('precision_lower_95'))}.")
            lines += [verdict, ""]
        lines += [
            "That is the honest answer to the mission's reliability ask: the tool must not be "
            "described as having an operating point that certifies 0.95 precision when this "
            "sample does not support it. Each value's percentage now comes from checks on "
            "its own evidence (an explicit label, the shape the field requires, the body "
            "being about the field) rather than from the section-type softmax, and the band "
            "table above is the measurement of whether that change made the percentage "
            "mean what it says.",
            "",
        ]
    # The value-level gold is the stratum with enough judged values to discuss a
    # 0.98 claim; the section-span overall (a handful of values) is not.
    test_overall = (test.get("field_gold") or {}).get("overall") \
        or {"judged": test.get("overall", {}).get("judged_values"),
            "matched": test.get("overall", {}).get("matched"),
            "precision_lower_95": 1 - (test.get("overall", {}).get("error_upper_bound_95") or 1.0)}
    dev_overall = dev.get("overall", {})
    test_n = test_overall.get("judged") or 0
    test_matched = test_overall.get("matched") or 0
    dev_n = dev_overall.get("judged_values") or 0
    dev_matched = dev_overall.get("matched") or 0
    test_lower = test_overall.get("precision_lower_95") or 0.0
    dev_lower = 1 - (dev_overall.get("error_upper_bound_95") or 1.0)
    perfect_lower = 1 - error_upper_bound(0, test_n) if test_n else 0.0
    # Judged values needed for a 0.98 one-sided lower bound with no misses.
    needed_for_98 = math.ceil(math.log(0.05) / math.log(0.98)) if test_n else 0
    if failures:
        lines += [
            "## Known failures found while reading the test documents",
            "",
            "Reading a test document can reveal a bug. Fixing it there would make the test "
            "set a development set, so these were recorded and fixed only where a development "
            "analogue existed; the test documents were not re-read, and the split's recorded "
            "run stands.",
            "",
            "Each row carries the state it reached: `fixed` with the development evidence "
            "that the cause is gone (and, where stated, the caveat that the original document "
            "was not re-read), or `open` with why the dev set could not settle it.",
            "",
            "| Document | Field | What happened | State in this pass |",
            "|---|---|---|---|",
        ]
        for item in failures:
            status = item.get("status", "open")
            detail = item.get("dev_check", "")
            cell = f"**{status}**" + (f" — {detail}" if detail else "")
            lines.append(f"| `{item['document']}` | `{item['field']}` | {item['note']} | {cell} |")
        lines.append("")
    lines += [
        "## What the split does and does not prove",
        "",
        "It proves the *rules* were not tuned on the test institutions. It does not make the "
        "encoder unseen: the shipped classifier checkpoint was trained on rows from the "
        "corpus schools, so a result on those schools still leans on a model that has read "
        "them. The two institutions added for the value-level gold are not in that training "
        "set, so their numbers are the only ones in this report where neither the rules nor "
        "the encoder had seen the documents.",
        "",
        "The test split was extracted once, in the pass that scored it, after every rule "
        "change up to that point had been measured on development; later passes fixed "
        "causes only on development and did not re-read or re-score it. The recorded "
        "artifact was then regenerated once for a reporting-only change: the section-span "
        "table now reports its own stratum, because it had been adding the value-level "
        "documents' rows into the same columns. No extraction rule, threshold or gold entry "
        "changed between those two runs; the headline and every value are identical. "
        "`extract_eval_test.previous.json` holds the previous pass's scored run (the source "
        "of the \"before\" numbers above) and `extract_eval_test.run1.json` the run before "
        "that.",
        "",
        f"**What this split can and cannot certify.** The value-level test gold judges "
        f"{test_n} values. That is enough to see a cause disappear and to put a lower bound "
        f"on the recorded run ({_fmt(test_lower)} at {test_matched}/{test_n}); it is not "
        f"enough to certify 0.98. Even a perfect {test_n}/{test_n} would certify only a "
        f"{_fmt(perfect_lower)} lower bound, and a 0.98 bound at zero misses needs at least "
        f"{needed_for_98} judged values. The development split ({dev_n} judged, "
        f"{dev_matched} correct, bound {_fmt(dev_lower)}) can show that a rule fixed a cause "
        f"or cost nothing; it cannot certify 0.98 either. No rule in this report was chosen "
        f"by a headline number, and none should be: a 0.98 claim needs a larger independent "
        f"gold set, not a tighter threshold.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_section_engine(checkpoint: str | Path, device: str):
    """Load whichever section classifier the checkpoint directory holds.

    The extraction pipeline only ever calls ``engine.classify(text, top_k=...)``, so
    both families plug in: an encoder checkpoint (``base_model`` in its config) and
    the Jev choice model (which has no ``base_model``). Without this the evaluator
    refused a Jev checkpoint with ``KeyError: 'base_model'`` and the better section
    classifier could never be measured end-to-end.
    """
    config_path = Path(checkpoint) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    # Only a Jev checkpoint writes a config.json with its option menu; encoder
    # checkpoints keep everything in checkpoint.pt.
    if "descriptions" in config or "config" in config:
        from .jev_model import JevEngine

        return JevEngine(checkpoint, device=device)
    from .field_model import FieldEngine

    return FieldEngine(checkpoint, device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score extraction against gold on held-out institutions")
    parser.add_argument("--ingested",
                        default="data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl")
    parser.add_argument("--corpus", default="data/processed/field/corpus.jsonl")
    parser.add_argument("--fold-plan", default="data/processed/field/fold_plan.json")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--context-checkpoint", default="artifacts/field-classifier-context")
    parser.add_argument("--exclude-school", action="append", default=None,
                        help="School used while developing the rules; repeatable")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Operating threshold to report at; defaults to the extractor's "
                             "own default (see pace.field_extract.DEFAULT_MIN_CONFIDENCE)")
    parser.add_argument("--hand-gold", default="data/processed/field/hand_gold.json",
                        help="Hand-read meetings/credits for documents the span gold misses")
    parser.add_argument("--field-gold", default="data/processed/field/gold_test_fields.json",
                        help="Value-level gold: every named field, read field by field")
    parser.add_argument("--extra-documents", default="data/raw/field-eval/syllabi.jsonl",
                        help="Documents fetched for the value-level gold set")
    parser.add_argument("--known-failures", default="data/processed/field/known_failures.json",
                        help="Bugs seen while reading test documents, recorded not fixed")
    parser.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: stop after N docs")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="dev",
                        help="Institutions used to develop the rules (dev), the untouched "
                             "set scored once at the end (test), or every held-out school")
    parser.add_argument("--write-split", action="store_true",
                        help="Write the dev/test split artifact and exit")
    parser.add_argument("--protocol-report", action="store_true",
                        help="Combine the recorded dev and test runs into EXTRACTION_EVAL.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args(argv)

    if args.write_split:
        write_split(args.ingested, args.fold_plan,
                    Path("data/processed/field/extract_split.json"),
                    extra_documents=args.extra_documents,
                    field_gold=args.field_gold)
        return

    if args.protocol_report:
        split_path = Path("data/processed/field/extract_split.json")
        if not split_path.exists():
            split_path.parent.mkdir(parents=True, exist_ok=True)
            write_split(args.ingested, args.fold_plan, split_path,
                        extra_documents=args.extra_documents, field_gold=args.field_gold)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        dev = json.loads(Path("artifacts/field-extract/extract_eval_dev.json")
                         .read_text(encoding="utf-8"))
        test = json.loads(Path("artifacts/field-extract/extract_eval_test.json")
                          .read_text(encoding="utf-8"))
        target = Path(args.markdown or "artifacts/field-extract/EXTRACTION_EVAL.md")
        write_protocol_report(split, dev, test, target, known_failures=args.known_failures)
        print(f"wrote {target}")
        return

    import torch

    from .field_extract import DEFAULT_MIN_CONFIDENCE

    operating = DEFAULT_MIN_CONFIDENCE if args.min_confidence is None else args.min_confidence
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = _load_section_engine(args.checkpoint, device)
    context = (_load_section_engine(args.context_checkpoint, device)
               if args.context_checkpoint and Path(args.context_checkpoint).exists() else None)

    plan = json.loads(Path(args.fold_plan).read_text(encoding="utf-8"))
    holdout_schools = {school for fold in plan for school in fold}
    excluded = set(args.exclude_school or DEFAULT_DEVELOPMENT_SCHOOLS)

    gold_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(args.corpus).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["group_id"].startswith("syllabus-doc:"):
                gold_by_doc[row["group_id"].split(":", 1)[1]].append(row)

    field_gold = load_gold(args.field_gold) if Path(args.field_gold).exists() else {}

    documents: list[dict[str, Any]] = []
    with Path(args.ingested).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("school") not in holdout_schools or record["school"] in excluded:
                continue
            if record["id"] in TEST_EXCLUDED_DOCUMENTS:
                continue
            part = split_of(record.get("school"))
            if part is None:
                continue
            if args.split != "both" and part != args.split:
                continue
            documents.append(record)
    if Path(args.extra_documents).exists() and args.split in ("test", "both"):
        with Path(args.extra_documents).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["id"] in field_gold:
                    documents.append(record)
    if not documents:
        raise SystemExit(f"no documents in the '{args.split}' split")
    documents.sort(key=lambda record: (str(record.get("school")), record["id"]))
    if args.limit:
        documents = documents[: args.limit]

    per_field: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"emitted": 0, "judged": 0, "matched": 0, "gold_rows": 0, "gold_covered": 0})
    # The coverage side: what was emitted against what the record read and did
    # not emit, so precision cannot be read without its denominator.
    emitted_by_field: dict[str, int] = defaultdict(int)
    suppressed_by_field: dict[str, int] = defaultdict(int)
    suppressed_reasons: Counter[str] = Counter()
    suppressed_reasons_by_field: dict[str, Counter[str]] = defaultdict(Counter)
    # The section-span table is its own stratum. Mixing the value-level documents
    # into it would make the rows sum to far more than the headline judges.
    span_per_field: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"emitted": 0, "judged": 0, "matched": 0, "gold_rows": 0, "gold_covered": 0})
    per_doc: list[dict[str, Any]] = []
    all_judged: list[dict[str, Any]] = []
    gold_values: list[dict[str, Any]] = []
    gold_documents: list[str] = []
    gold_recall_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    records: dict[str, dict[str, Any]] = {}
    titles_emitted = titles_grounded = 0

    for index, document in enumerate(documents, start=1):
        # Extraction always runs unfiltered: the threshold is applied afterwards so the
        # threshold table measures the same values the command would emit at each setting.
        record = extract_record(document["text"], engine, context, min_confidence=0.0)
        gold_rows = gold_by_doc.get(document["id"], [])
        records[document["id"]] = record
        title = (record.get("course") or {}).get("course_title")
        if title:
            titles_emitted += 1
            if title.get("grounded", True):
                titles_grounded += 1
        document_gold = field_gold.get(document["id"])
        if document_gold is not None:
            values = emitted_values(record)
            judged = score_document(values, document_gold)
            gold_documents.append(document["id"])
            for value in judged:
                gold_values.append({**value, "document": document["id"][:12],
                                    "school": document["school"]})
            for field, (hit, total) in gold_recall(document_gold, values).items():
                gold_recall_totals[field][0] += hit
                gold_recall_totals[field][1] += total
            judged = [{**value, "gold_source": "field"} for value in judged]
        else:
            judged = [{**value, "gold_source": "span"}
                      for value in evaluate_document(record, gold_rows)]
        kept = [value for value in judged if value["confidence"] >= operating]
        if document_gold is None:
            # Section-span gold covers one stratum; the value-level gold is reported
            # in its own table so the two are never summed into one precision.
            all_judged.extend(kept)
        if document_gold is None:
            # The ledger owns what a loss is; the evaluation only counts it.
            for field, reasons in summarize_losses(suppressed_values(record)).items():
                for reason, count in reasons.items():
                    suppressed_by_field[field] += count
                    suppressed_reasons[reason] += count
                    suppressed_reasons_by_field[field][reason] += count
        for value in kept:
            if document_gold is None:
                emitted_by_field[value["field"]] += 1
            stats = per_field[value["field"]]
            stats["emitted"] += 1
            if value["judged"]:
                stats["judged"] += 1
                stats["matched"] += int(value["matched"])
            if document_gold is None:
                span_stats = span_per_field[value["field"]]
                span_stats["emitted"] += 1
                if value["judged"]:
                    span_stats["judged"] += 1
                    span_stats["matched"] += int(value["matched"])
        for field, (hit, total) in recall_by_field(gold_rows, kept, record).items():
            per_field[field]["gold_rows"] += total
            per_field[field]["gold_covered"] += hit
            if document_gold is None:
                span_per_field[field]["gold_rows"] += total
                span_per_field[field]["gold_covered"] += hit
        per_doc.append({
            "id": document["id"][:12],
            "school": document["school"],
            "title": document.get("title"),
            "gold_rows": len(gold_rows),
            "field_gold": document_gold is not None,
            "emitted": len(kept),
            "matched": sum(1 for value in kept if value["matched"]),
        })
        print(json.dumps({"doc": index, "of": len(documents), "school": document["school"],
                          "emitted": len(kept)}, ensure_ascii=False), flush=True)

    def _table(source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        built = {}
        for field, stats in sorted(source.items()):
            precision = (stats["matched"] / stats["judged"]) if stats["judged"] else None
            recall = (stats["gold_covered"] / stats["gold_rows"]) if stats["gold_rows"] else None
            built[field] = {**stats, "precision": precision, "recall": recall}
        return built

    table = _table(span_per_field)
    coverage = {
        field: {
            "emitted": emitted_by_field.get(field, 0),
            "suppressed": suppressed_by_field.get(field, 0),
            "gold_items": table.get(field, {}).get("gold_rows", 0),
            "found": table.get(field, {}).get("gold_covered", 0),
            "recall": table.get(field, {}).get("recall"),
        }
        for field in sorted(set(emitted_by_field) | set(suppressed_by_field) | set(table))
    }

    judged_values = [value for value in all_judged if value["judged"]]
    bins = [(0.0, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]
    confidence_check = []
    for low, high in bins:
        in_bin = [value for value in judged_values if low <= value["confidence"] < high]
        correct = sum(1 for value in in_bin if value["matched"])
        confidence_check.append({"low": low, "high": min(high, 1.0), "values": len(in_bin),
                                 "correct": correct,
                                 "precision": correct / len(in_bin) if in_bin else None})
    matched = sum(1 for value in judged_values if value["matched"])
    overall = {
        "judged_values": len(judged_values),
        "matched": matched,
        "precision": matched / len(judged_values) if judged_values else None,
        "error_upper_bound_95": error_upper_bound(len(judged_values) - matched,
                                                  len(judged_values)),
        "unjudged_values": len(all_judged) - len(judged_values),
        "documents": len(documents),
        "schools": sorted({document["school"] for document in documents}),
        "excluded_schools": sorted(excluded),
        "min_confidence": operating,
    }
    hand_checked = None
    if args.hand_gold and Path(args.hand_gold).exists():
        hand_checked = score_hand_gold(
            records, json.loads(Path(args.hand_gold).read_text(encoding="utf-8")))
    grounding = {"titles_emitted": titles_emitted, "titles_grounded": titles_grounded,
                 "rate": titles_grounded / titles_emitted if titles_emitted else 0.0}
    overall["split"] = args.split
    judged_all = [value for value in all_judged if value["judged"]]
    result = {"overall": overall, "per_field": table, "confidence_check": confidence_check,
              "coverage": coverage,
              "coverage_reasons": dict(suppressed_reasons.most_common(12)),
              # Every reason, not the top few: the per-field counts have to add
              # up to that field's suppressed total or the ledger cannot be audited.
              "coverage_reasons_by_field": {
                  field: dict(counts.most_common())
                  for field, counts in sorted(suppressed_reasons_by_field.items())
              },
              "grounding": grounding, "hand_checked": hand_checked, "per_document": per_doc,
              "values": all_judged,
              "gold_sources": {"field": len(gold_values), "span": len(judged_all)},
              "reliability": gold_reliability(judged_all),
              "sweep": gold_sweep(judged_all)}
    result["operating_point"] = operating_point(result["sweep"], args.target_precision)
    if gold_values:
        kept_gold = [value for value in gold_values if value["confidence"] >= operating]
        judged_gold = [value for value in kept_gold if value["judged"]]
        correct = sum(1 for value in judged_gold if value["matched"])
        bound = error_upper_bound(len(judged_gold) - correct, len(judged_gold))
        precision = correct / len(judged_gold) if judged_gold else None
        sweep_rows = gold_sweep(gold_values)
        result["field_gold"] = {
            "documents": len(gold_documents),
            "schools": sorted({value["school"] for value in gold_values}),
            "values": gold_values,
            "overall": {
                "judged": len(judged_gold),
                "matched": correct,
                "unjudged": len(kept_gold) - len(judged_gold),
                "precision": precision,
                "precision_lower_95": None if bound is None else 1 - bound,
                "confidence": CONFIDENCE,
                "min_confidence": operating,
            },
            "per_field": gold_by_field(kept_gold),
            "recall": {field: list(hit) for field, hit in sorted(gold_recall_totals.items())},
            "reliability": gold_reliability(kept_gold),
            "sweep": sweep_rows,
            "operating_point": operating_point(sweep_rows, args.target_precision),
            "floor_operating_point": operating_point(sweep_rows, FLOOR_PRECISION),
        }
    known_path = Path(args.known_failures)
    if known_path.exists():
        result["known_failures"] = json.loads(known_path.read_text(encoding="utf-8"))

    output = Path(args.output or f"artifacts/field-extract/extract_eval_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(documents)} documents, {len(overall['schools'])} institutions")
    print(f"{'field':<22}{'emitted':>8}{'judged':>8}{'matched':>8}{'precision':>11}{'recall':>9}")
    for field, stats in table.items():
        print(f"{field:<22}{stats['emitted']:>8}{stats['judged']:>8}{stats['matched']:>8}"
              f"{_fmt(stats['precision']):>11}{_fmt(stats['recall']):>9}")
    bound = overall["error_upper_bound_95"]
    print(f"\nspan gold: precision {_fmt(overall['precision'])} on {overall['judged_values']} "
          f"judged values; 95% lower bound {_fmt(None if bound is None else 1 - bound)}; "
          f"{overall['unjudged_values']} unjudged")
    if "field_gold" in result:
        block = result["field_gold"]
        print(f"\nvalue-level gold: {block['overall']['matched']}/{block['overall']['judged']} "
              f"correct = {_fmt(block['overall']['precision'])} "
              f"(95% lower bound {_fmt(block['overall']['precision_lower_95'])}) on "
              f"{block['documents']} documents from {len(block['schools'])} institutions; "
              f"{block['overall']['unjudged']} values unjudged")
        print(f"{'threshold':>10}{'kept':>8}{'coverage':>10}{'precision':>11}{'lower95':>9}")
        for row in block["sweep"]:
            print(f"{row['threshold']:>10.3f}{row['accepted']:>8}{row['coverage']:>10.2f}"
                  f"{_fmt(row['precision']):>11}{_fmt(row['precision_lower_95']):>9}")
        point = block["operating_point"]
        if point["threshold"] is not None:
            print(f"\ncertified target {point['target']:.2f} at min_confidence="
                  f"{point['threshold']:.3f} (coverage {point['coverage']:.2f})")
        else:
            best = point.get("best") or {}
            print(f"\nno threshold certifies {point['target']:.2f}; best bound "
                  f"{_fmt(best.get('precision_lower_95'))} at "
                  f"min_confidence={best.get('threshold')}")
    if hand_checked:
        meetings, credits = hand_checked["meetings"], hand_checked["credits"]
        if meetings["gold_slots"]:
            print(f"\nhand-checked meetings {meetings['exact']}/{meetings['gold_slots']} exact; "
                  f"credits {credits['correct']}/{credits['gold_documents']}")
    if args.markdown:
        markdown = Path(args.markdown)
        write_markdown(result, markdown, split=args.split)
        print(f"wrote {markdown}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
