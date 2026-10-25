"""Score the extractor on documents from institutions in neither split.

The two numbers this project already publishes cannot answer "does the accuracy
hold on documents the work has never seen":

* the development split was used to choose every rule, so its precision is a
  training score;
* the test split has 66 judged values, which cannot certify 0.98 even if every
  one of them is correct.

This module adds a third measurement on documents from the two corpus schools
that belong to neither split (UC Berkeley, Georgia Tech). They are not a clean
never-seen sample — they were read while the very first rules were written, so
the number is an upper bound — but they have not been read in this pass and no
rule was changed for them. The rubric was written first
([`docs/heldout-rubric.md`](docs/heldout-rubric.md)), the labels were frozen with
a recorded SHA-256 before the extractor was run, and the score is reported as
measured.

The artifact is meant to be checkable without re-running the model. Every emitted
value appears once in ``values`` with its verdict (``matched``/``wrong``/
``unjudged``), the gold entry it matched or the entries it was compared against,
the rule that decided it, and its evidence. The headline counts are folds of that
list, not a separate calculation. A ``--limit`` run is marked ``truncated`` in
the artifact and refuses to overwrite the published output.

Run it with::

    python -m pace.field_heldout_eval --output artifacts/field-extract/heldout_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .field_calibration import error_upper_bound
from .field_extract_eval import emitted_values
from .field_gold import (ABSENT_RULE, CONFIDENCE, FIELDS, MATCH_RULES, UNJUDGED_RULE,
                         load_gold, match_entry, recall as gold_recall)
from .field_ledger import summarize as summarize_losses, suppressed_values

SELECTION = Path("data/processed/field/heldout_selection.json")
GOLD = Path("data/processed/field/gold_heldout_fields.json")
INGESTED = Path("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl")
RUBRIC = "docs/heldout-rubric.md"
DEFAULT_OUTPUT = Path("artifacts/field-extract/heldout_eval.json")

# The claim this set exists to test. A clean sample of N judged values certifies
# a precision lower bound of (1 - CONFIDENCE) ** (1 / N).
TARGET = 0.98


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selection(path: Path = SELECTION) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_freeze(selection: dict[str, Any], gold_path: Path = GOLD,
                 allow_changed_gold: bool = False) -> str:
    """Refuse to score a gold file that is not the one that was frozen."""
    digest = sha256(gold_path)
    frozen = selection.get("gold_sha256")
    if not frozen and not allow_changed_gold:
        raise SystemExit(
            f"{gold_path} has never been frozen: record its SHA-256 in "
            f"{SELECTION} before scoring")
    if frozen and digest != frozen and not allow_changed_gold:
        raise SystemExit(
            f"{gold_path} changed since it was frozen (frozen {frozen[:12]}..., "
            f"now {digest[:12]}...); scoring a re-labelled set after seeing the "
            f"output would defeat the point. Pass --allow-changed-gold only while "
            f"annotating, never for a reported run")
    return digest


def load_documents(selection: dict[str, Any],
                   ingested: Path = INGESTED) -> list[dict[str, Any]]:
    """The frozen documents, in selection order, straight from the corpus."""
    wanted = [entry["id"] for entry in selection["documents"]]
    found: dict[str, dict[str, Any]] = {}
    with Path(ingested).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["id"] in wanted:
                found[record["id"]] = record
    missing = [doc_id for doc_id in wanted if doc_id not in found]
    if missing:
        raise SystemExit(f"{len(missing)} selected documents are not in {ingested}: "
                         f"{[doc_id[:12] for doc_id in missing]}")
    return [found[doc_id] for doc_id in wanted]


def perfect_bound(judged: int, confidence: float = CONFIDENCE) -> float | None:
    """Precision lower bound at zero errors for a sample of this size.

    Derived from ``confidence``, the same level ``precision_lower_95`` uses, so
    the two cannot disagree about what interval they describe.
    """
    if judged <= 0:
        return None
    return (1 - confidence) ** (1 / judged)


def needed_for(target: float = TARGET, confidence: float = CONFIDENCE) -> int:
    """Smallest zero-error sample whose precision bound at ``confidence`` reaches ``target``."""
    return max(1, math.ceil(math.log(1 - confidence) / math.log(target)))


def run_protocol(selection: dict[str, Any], limit: int | None = None,
                 note: str | None = None) -> dict[str, Any]:
    """Provenance for the run, including whether it is a truncated debug run."""
    total = len(selection["documents"])
    scored = min(limit, total) if limit is not None else total
    protocol = {
        "rubric": RUBRIC,
        "rubric_sha256": sha256(Path(RUBRIC)) if Path(RUBRIC).exists() else None,
        "selection": str(SELECTION),
        "selection_rule": selection.get("selection_rule"),
        "caveat": selection.get("caveat"),
        "gold": str(GOLD),
        "gold_sha256": selection.get("gold_sha256"),
        "documents": scored,
        "documents_selected": total,
        "limit": limit,
        "truncated": scored < total,
        "rule_changes": "none: every rule, threshold and cap is the code as it stands",
    }
    if note:
        protocol["note"] = note
    return protocol


def _row(entry: dict[str, Any], value: dict[str, Any],
         document: dict[str, Any]) -> dict[str, Any]:
    """One emitted value and the full account of its verdict."""
    field = value["field"]
    entries = document.get(field) or []
    row: dict[str, Any] = {
        "document": entry["id"][:12],
        "school": entry["school"],
        "field": field,
        "text": value.get("text"),
        "confidence": value.get("confidence"),
        "evidence": value.get("evidence"),
        "gold_entry": None,
        "compared_against": None,
        "why_unjudged": None,
    }
    for key in ("days", "start_time", "end_time", "room", "weight"):
        if key in value:
            row[key] = value.get(key)
    if field in set(document.get("absent") or []):
        row.update(decision="wrong", rule=ABSENT_RULE, compared_against=[])
        return row
    if not entries:
        row.update(decision="unjudged", rule=UNJUDGED_RULE,
                   why_unjudged="the document neither states this field nor marks it absent")
        return row
    matched = match_entry(field, entries, value)
    row.update(decision="matched" if matched else "wrong", rule=MATCH_RULES.get(field),
               gold_entry=matched, compared_against=None if matched else entries)
    return row


def score(records: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]],
          selection: dict[str, Any], limit: int | None = None,
          note: str | None = None) -> dict[str, Any]:
    """Per-value verdicts, their folds, and the loss ledger on the held-out set."""
    scored = selection["documents"][:limit] if limit is not None else selection["documents"]
    rows: list[dict[str, Any]] = []
    recall: dict[str, list[int]] = {}
    suppressed: dict[str, dict[str, int]] = {}
    suppressed_total = 0
    uncovered: list[dict[str, Any]] = []
    per_document: list[dict[str, Any]] = []

    for entry in scored:
        document = gold[entry["id"]]
        record = records[entry["id"]]
        mine = emitted_values(record)
        document_rows = [_row(entry, value, document) for value in mine]
        rows.extend(document_rows)
        for field, (hit, total) in gold_recall(document, mine).items():
            recall.setdefault(field, [0, 0])[0] += hit
            recall[field][1] += total
            if hit == 0:
                uncovered.append({"document": entry["id"][:12], "field": field,
                                  "expected": document.get(field)})
        for field, reasons in summarize_losses(suppressed_values(record)).items():
            bucket = suppressed.setdefault(field, {})
            for reason, count in reasons.items():
                bucket[reason] = bucket.get(reason, 0) + count
                suppressed_total += count
        per_document.append({
            "id": entry["id"][:12],
            "school": entry["school"],
            "title": entry.get("title"),
            "emitted": len(document_rows),
            "judged": sum(1 for row in document_rows if row["decision"] != "unjudged"),
            "matched": sum(1 for row in document_rows if row["decision"] == "matched"),
            "suppressed": len(suppressed_values(record)),
        })

    judged = [row for row in rows if row["decision"] != "unjudged"]
    matched = [row for row in rows if row["decision"] == "matched"]
    wrong = [row for row in rows if row["decision"] == "wrong"]
    bound = error_upper_bound(len(wrong), len(judged), CONFIDENCE)

    per_field: dict[str, dict[str, Any]] = {}
    for field in FIELDS:
        mine = [row for row in rows if row["field"] == field]
        judged_here = [row for row in mine if row["decision"] != "unjudged"]
        correct = sum(1 for row in judged_here if row["decision"] == "matched")
        hit, total = recall.get(field, [0, 0])
        per_field[field] = {
            "emitted": len(mine),
            "judged": len(judged_here),
            "matched": correct,
            "errors": len(judged_here) - correct,
            "precision": correct / len(judged_here) if judged_here else None,
            "gold_items": total,
            "gold_covered": hit,
            "recall": hit / total if total else None,
            "suppressed": suppressed.get(field, {}),
        }
    for field, bucket in suppressed.items():
        per_field.setdefault(field, {"emitted": 0, "judged": 0, "matched": 0, "errors": 0,
                                     "precision": None, "gold_items": 0,
                                     "gold_covered": 0, "recall": None})["suppressed"] = bucket

    confidences = sorted(row["confidence"] or 0.0 for row in judged)
    return {
        "protocol": run_protocol(selection, limit, note),
        "overall": {
            "documents": len(scored),
            "emitted": len(rows),
            "judged": len(judged),
            "matched": len(matched),
            "errors": len(wrong),
            "precision": len(matched) / len(judged) if judged else None,
            "precision_lower_95": None if bound is None else 1 - bound,
            "recall": {
                "items": sum(total for _, total in recall.values()),
                "covered": sum(hit for hit, _ in recall.values()),
            },
        },
        "certification": {
            "confidence": CONFIDENCE,
            "target": TARGET,
            "judged": len(judged),
            "perfect_lower_95": perfect_bound(len(judged)),
            "needed_for_target": needed_for(),
            "certifies_target": bool(judged and perfect_bound(len(judged)) >= TARGET),
        },
        "by_field": per_field,
        "losses": {
            "emitted": len(rows),
            "suppressed": suppressed_total,
            "suppressed_by_field": {field: sum(bucket.values())
                                    for field, bucket in sorted(suppressed.items())},
        },
        "values": rows,
        "errors": [{key: row.get(key) for key in
                    ("document", "school", "field", "text", "confidence", "evidence", "days",
                     "start_time", "end_time", "room", "weight")}
                   for row in wrong],
        "uncovered": uncovered,
        "documents": per_document,
        "confidence": {"min": confidences[0] if confidences else None,
                       "median": confidences[len(confidences) // 2] if confidences else None},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Score field extraction on the held-out (never-tuned) documents")
    parser.add_argument("--selection", default=str(SELECTION))
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--ingested", default=str(INGESTED))
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--context-checkpoint", default="artifacts/field-classifier-context")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-changed-gold", action="store_true",
                        help="Annotating only: score a gold file after its hash changed")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug only: first N documents; the artifact is marked truncated")
    parser.add_argument("--note", default=None,
                        help="Free-text provenance, recorded in protocol.note (e.g. that this "
                             "is a second touch after a fix informed by the first)")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="Replace an existing published artifact; the one-touch number is "
                             "evidence and is frozen by default")
    args = parser.parse_args(argv)

    selection = load_selection(Path(args.selection))
    check_freeze(selection, Path(args.gold), allow_changed_gold=args.allow_changed_gold)
    gold = load_gold(Path(args.gold))
    documents = load_documents(selection, Path(args.ingested))
    target = Path(args.output)
    truncated = args.limit is not None and args.limit < len(documents)
    if truncated:
        if target.resolve() == DEFAULT_OUTPUT.resolve():
            raise SystemExit(
                f"--limit writes a truncated run; it cannot go to the published output "
                f"({DEFAULT_OUTPUT}). Pass --output to a scratch path")
        print(f"TRUNCATED RUN: {args.limit} of {len(documents)} documents; the artifact "
              f"is marked truncated and is not the held-out result", flush=True)
        documents = documents[: args.limit]
    if target.resolve() == DEFAULT_OUTPUT.resolve() and target.exists() \
            and not args.allow_overwrite:
        raise SystemExit(
            f"{DEFAULT_OUTPUT} is the published held-out artifact; it is frozen, so this run "
            f"will not replace it. Pass --output for a new artifact (or --allow-overwrite).")

    import torch

    from .field_extract import extract_record
    from .field_model import FieldEngine

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = FieldEngine(args.checkpoint, device=device)
    context = FieldEngine(args.context_checkpoint, device=device) \
        if args.context_checkpoint and Path(args.context_checkpoint).exists() else None

    records: dict[str, dict[str, Any]] = {}
    for index, document in enumerate(documents, start=1):
        records[document["id"]] = extract_record(document["text"], engine, context,
                                                 min_confidence=0.0)
        print(json.dumps({"doc": index, "of": len(documents),
                          "school": document["school"]}, ensure_ascii=False), flush=True)

    result = score(records, gold, selection, limit=args.limit, note=args.note)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overall = result["overall"]
    print(f"judged {overall['judged']}, correct {overall['matched']}, "
          f"precision {overall['precision']:.3f}, "
          f"lower bound {overall['precision_lower_95']:.3f} -> {target}")


if __name__ == "__main__":
    main()
