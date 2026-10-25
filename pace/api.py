"""The application-facing reader: one text in, one versioned record out.

Everything below this module is a research surface -- three checkpoints, two
tokenizers, a multitask head, rule engines and ledgers. An app should not have
to know any of that. :func:`understand_academic_document` is the single entry
point: it takes text, runs the reader that actually works on real documents,
and returns one dictionary whose shape is fixed by :data:`SCHEMA_VERSION`.

What the record does *not* claim:

* ``document_type`` is a gate, not just a decision. It comes from a four-class
  linear model (syllabus / course_catalog / schedule / other) with 0.9967 test
  accuracy on 302 rows and zero false-academic answers on 239 real negatives.
  When it says ``other`` the reader now *stops*: no course, no grading, no
  meetings, no entities, and every value it would have returned is listed under
  ``suppressed`` with the reason ``document_type_not_academic``. Transcript,
  gradebook, report card and graduation requirements are **not** classes: real
  documents of those kinds are handled deterministically or not at all. A real
  transcript currently lands in ``course_catalog`` at 0.94, so treat the type as
  a hint for routing *within* the academic flow, not as truth.
* Confidence is measured, not decorative. In-distribution the section reader is
  calibrated (ECE 0.0010). On documents from a university it never saw it is
  much better than it used to be but still not calibrated (ECE 0.1325, accuracy
  0.8390), so a band describes how much to trust a *preselection*, not whether a
  value is true. Values that come from a rule carry ``confidence_source: "rule"``
  and no band, because a rule's number is not a probability.
* Identity is deterministic. Every object carries an id derived from the
  document and where the value came from, so a UI can attach accept/reject/edit
  state without matching on strings, and an import can be replayed without
  duplicating rows. See :mod:`pace.ir_schema`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import READER, doc_type_dir
from .field_corpus import clean_html
from .field_extract import extract_record, type_sections
from .ir_schema import SCHEMA_VERSION, document_id, stable_id, validate

#: The frozen reader. Changing the checkpoint is an edit to :mod:`pace.config`,
#: never to this module or to an application.
DEFAULT_CHECKPOINT = str(READER["section_checkpoint"])
DEFAULT_CONTEXT_CHECKPOINT = str(READER["context_checkpoint"])

#: (floor, band). Only model probabilities are banded, never rule confidences.
BANDS: tuple[tuple[float, str], ...] = tuple(READER["bands"])  # type: ignore[assignment]

#: Measured, not chosen. Sections: 7,571 held-out rows and 447 rows from five
#: universities the model never trained on. Rule values: the one-touch extraction
#: gold set recorded in ``artifacts/field-extract/EXTRACTION_EVAL.md``.
CALIBRATION: dict[str, Any] = {
    "sections_in_distribution_ece": 0.0016,
    "sections_unseen_university_ece": 0.1665,
    "sections_unseen_university_accuracy": 0.7830,
    "rule_values_development_precision": 0.917,
    "rule_values_holdout_precision": 0.955,
    "holdout_documents": 12,
    "alt_reader": {
        "checkpoint": "artifacts/jev-field-v4",
        "sections_in_distribution_ece": 0.0010,
        "sections_unseen_documents_ece": 0.1325,
        "sections_unseen_documents_accuracy": 0.8390,
        "value_level_gold_precision": 0.939,
        "value_level_gold_lower_95": 0.867,
        "why_not_default": "picked a grading line as course_title on a sample syllabus",
    },
    "bands": {"high": ">=0.97", "medium": "0.85-0.97", "low": "<0.85"},
    "bands_apply_to": "section labels only (model probabilities)",
    "value_confidences": ("set by the rule that read the value (0.60-0.97), not a "
                          "probability; their measured precision is the rule number above"),
    "note": ("A high band is a preselect-and-confirm signal, not an assertion. "
             "On unseen universities 0.98+ model confidence measured 0.828 precision. "
             "Every value this module returns requires a human confirmation."),
}

#: Which entity kind each course field becomes, and whether it is a course object.
_COURSE_KINDS = {
    "course_title": "course",
    "course_code": "course_code",
    "instructor": "instructor",
    "credits": "credit",
}

_READERS: dict[tuple[str, str, str], "AcademicReader"] = {}


def confidence_band(probability: float) -> str:
    """The band for a model probability, from the measured sweep."""
    for floor, name in BANDS:
        if probability >= floor:
            return name
    return "low"


class AcademicReader:
    """Loads one field checkpoint and reads documents with it."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 context_checkpoint: str | Path | None = DEFAULT_CONTEXT_CHECKPOINT,
                 device: str | None = None):
        from .cli import classify_engine

        self.checkpoint = str(checkpoint)
        self.engine = classify_engine(self.checkpoint, device=device)
        self.context = None
        self.context_checkpoint = str(context_checkpoint) if context_checkpoint else None
        if context_checkpoint and Path(context_checkpoint).exists():
            self.context = classify_engine(context_checkpoint, device=device)
        self.family = ("jev-choice" if hasattr(self.engine, "option_ids")
                       else "encoder-classifier")
        self.parameters = sum(parameter.numel() for parameter in self.engine.model.parameters())
        self.max_tokens = int(getattr(self.engine, "max_state", 0) or 256)
        # The document-type gate is optional: without its artifact the reader still
        # reads fields, it just cannot say what kind of page it is looking at.
        self.doc_type = None
        try:
            from .doc_type import DocumentTypeClassifier

            self.doc_type = DocumentTypeClassifier.load(doc_type_dir())
        except (FileNotFoundError, ImportError):
            self.doc_type = None

    # -- one document ------------------------------------------------------ #

    def read(self, text: str, min_confidence: float = 0.0) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        cleaned, _ = clean_html(text)
        doc_id = document_id(cleaned)
        document_type = self.doc_type.classify(cleaned) if self.doc_type is not None else None
        blocked = document_type is not None and not document_type["academic"]

        sections = type_sections(cleaned, self.engine, self.context)
        record = extract_record(cleaned, self.engine, self.context,
                                min_confidence=min_confidence)

        suppressed: list[dict[str, Any]] = list(record["suppressed"])
        if blocked:
            suppressed.extend(_gate_suppressed(record))
            record = {"course": {}, "grading": [], "meetings": [], "policies": {},
                      "grading_total_percent": None, "grading_total_consistent": None}

        typed = [{
            "id": stable_id("section", doc_id, index, section["start"], section["label"]),
            "heading": section["text"],
            "label": section["label"],
            "task": getattr(self.engine, "label_tasks", {}).get(section["label"], "other"),
            "confidence": section["confidence"],
            "band": confidence_band(section["confidence"]),
            "confidence_source": "model",
            "start": section["start"],
            "end": section["end"],
            "source_text": cleaned[section["start"]:section["end"]],
        } for index, section in enumerate(sections)] if not blocked else []

        entries = _evidence_entries(record, cleaned, doc_id)
        evidence = [entry["item"] for entry in entries]
        entities, relations = _entities_and_relations(entries, doc_id)
        warnings = _warnings(record, typed, cleaned, evidence)
        if blocked:
            warnings = [f"this does not look like an academic document "
                        f"({document_type['label']} {document_type['confidence']:.2f}); "
                        f"extraction was suppressed and no academic fields are returned"]

        return {
            "schema_version": SCHEMA_VERSION,
            "schema": {
                "version": SCHEMA_VERSION,
                "migrated_from": None,
                "id_policy": "sha256(document_id, kind, origin); stable across reads",
            },
            "document": {
                "id": doc_id,
                "characters": len(text),
                "cleaned_characters": len(cleaned),
                "truncated_for_evidence": len(cleaned) != len(text),
                "evidence_coordinates": "cleaned_text",
                "sections_detected": len(typed),
                "sections_suppressed": len(sections) if blocked else 0,
                "entities_suppressed": (0 if not blocked else
                                        len(record["course"]) + len(record["grading"])
                                        + len(record["meetings"]) + len(record["policies"])),
                "gate": "blocked" if blocked else "open",
                "document_type": document_type,
                "document_type_unavailable_because": (
                    None if document_type is not None else
                    f"{doc_type_dir()}/doc_type.joblib is not present"),
            },
            "course": {key: value["value"] for key, value in record["course"].items()},
            "grading": record["grading"],
            "grading_total_percent": record.get("grading_total_percent"),
            "grading_total_consistent": record.get("grading_total_consistent"),
            "meetings": record["meetings"],
            "policies": record["policies"],
            "sections": typed,
            "entities": entities,
            "relations": relations,
            "evidence": evidence,
            "suppressed": suppressed,
            "confidence": dict(CALIBRATION),
            "model": {
                "engine_version": READER["engine_version"],
                "checkpoint": self.checkpoint,
                "context_checkpoint": (self.context_checkpoint
                                       if self.context is not None else None),
                "family": self.family,
                "parameters": self.parameters,
                "max_state_tokens": self.max_tokens,
                "temperature": getattr(self.engine, "temperature", None),
                "document_type_checkpoint": (str(doc_type_dir())
                                             if self.doc_type is not None else None),
            },
            "warnings": warnings,
            "requires_confirmation": True,
            "commit_policy": "proposals_only_user_confirmation_required",
        }

    def read_path(self, path: str | Path) -> dict[str, Any]:
        """Read a text file. PDFs and images must go through ``pace.ingest`` first."""
        return self.read(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Evidence, identity, relations
# --------------------------------------------------------------------------- #

def _span(entry: dict[str, Any], cleaned: str) -> tuple[int | None, int | None, bool]:
    evidence = entry.get("evidence") or {}
    start, end = evidence.get("start"), evidence.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None, None, False
    if not (0 <= start < end <= len(cleaned)):
        return start, end, False
    written = evidence.get("text")
    if isinstance(written, str) and written.strip():
        return start, end, written.strip() in cleaned[start:end]
    return start, end, True


def _item(field: str, entry: dict[str, Any], cleaned: str,
          value: Any = None, source: str = "rule") -> dict[str, Any]:
    """One emitted value with its exact source span.

    A value read by a rule carries the rule's own number, not a probability, so
    it gets no band. Only section labels were scored as probabilities.
    """
    start, end, grounded = _span(entry, cleaned)
    confidence = float(entry.get("confidence", 0.0) or 0.0)
    return {
        "field": field,
        "value": value if value is not None else entry.get("value"),
        "confidence": confidence,
        "confidence_source": source,
        "band": confidence_band(confidence) if source == "model" else None,
        "source_text": (entry.get("evidence") or {}).get("text"),
        "start": start,
        "end": end,
        "grounded_in_source": grounded,
        "requires_confirmation": True,
    }


def _evidence_entries(record: dict[str, Any], cleaned: str,
                      doc_id: str) -> list[dict[str, Any]]:
    """Every emitted value, paired with the record object it came from.

    The pairing is what lets an entity point at its own evidence: the public
    ``evidence`` list and the ``entities`` list are built from the same walk, so
    an id can never refer to a row the other list does not have.
    """
    entries: list[dict[str, Any]] = []

    def add(field: str, entry: dict[str, Any], value: Any = None) -> None:
        item = _item(field, entry, cleaned, value=value)
        item = {"id": stable_id("evidence", doc_id, field, item.get("start"),
                                item.get("end"), item.get("value")), **item}
        entries.append({"field": field, "entry": entry, "item": item})

    for field, entry in record["course"].items():
        add(field, entry)
    for entry in record["grading"]:
        add("grading", entry, f"{entry.get('category')} {entry.get('weight')}%")
    for entry in record["meetings"]:
        label = " ".join(part for part in (entry.get("days") or []) if part)
        clock = f"{entry.get('start_time') or '?'}-{entry.get('end_time') or '?'}"
        add("meetings", entry, f"{label} {clock}".strip())
    for field, entry in record["policies"].items():
        add(field, entry)
    return entries


def _entities_and_relations(entries: list[dict[str, Any]],
                            doc_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Entities and the relations that are *structural*, not predicted.

    Three relation types are emitted, and only these, because the record itself
    states them: a grading row is a category with a weight (``WEIGHT_OF``), a
    course and its instructor line are one assertion (``TAUGHT_BY``), a meeting
    has a clock (``MEETS_ON``). Model-predicted course relations
    (``PREREQUISITE_OF`` and friends) live in ``pace.real_relations`` and are
    deliberately *not* mixed into this list -- a UI must be able to tell a
    parsed fact from a proposal.
    """
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    by_field: dict[str, dict[str, Any]] = {}
    for entry in entries:
        field, item = entry["field"], entry["item"]
        if field == "grading":
            kind = "grade_category"
        elif field == "meetings":
            kind = "meeting"
        elif field in _COURSE_KINDS:
            kind = _COURSE_KINDS[field]
        else:
            kind = "policy"
        entity = {
            "id": stable_id("entity", doc_id, kind, field,
                            item.get("start"), item.get("value")),
            "kind": kind,
            "field": field,
            "value": item.get("value"),
            "confidence": item.get("confidence"),
            "confidence_source": item.get("confidence_source"),
            "band": item.get("band"),
            "evidence_id": item["id"],
            "requires_confirmation": True,
        }
        if field == "grading":
            entity["weight"] = entry["entry"].get("weight")
        entities.append(entity)
        by_field.setdefault(field, entity)

        if field == "grading":
            relations.append({
                "id": stable_id("relation", doc_id, "WEIGHT_OF", entity["id"]),
                "type": "WEIGHT_OF",
                "from": entity["id"],
                "to": f"{entity['weight']}%",
                "weight": entity["weight"],
                "evidence_id": item["id"],
                "predicted": False,
                "requires_confirmation": True,
            })
        elif field == "meetings":
            parts = [day for day in (entry["entry"].get("days") or []) if day]
            clock = (f"{entry['entry'].get('start_time')}-{entry['entry'].get('end_time')}")
            relations.append({
                "id": stable_id("relation", doc_id, "MEETS_ON", entity["id"]),
                "type": "MEETS_ON",
                "from": entity["id"],
                "to": " ".join(parts + [clock]).strip(),
                "evidence_id": item["id"],
                "predicted": False,
                "requires_confirmation": True,
            })

    course = by_field.get("course_title") or by_field.get("course_code")
    instructor = by_field.get("instructor")
    if course is not None and instructor is not None:
        relations.append({
            "id": stable_id("relation", doc_id, "TAUGHT_BY", course["id"], instructor["id"]),
            "type": "TAUGHT_BY",
            "from": course["id"],
            "to": instructor["id"],
            "evidence_id": instructor["evidence_id"],
            "predicted": False,
            "requires_confirmation": True,
        })
    return entities, relations


def _gate_suppressed(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Every value the extractor produced, kept as an audit trail of the gate."""
    losses: list[dict[str, Any]] = []
    for field, entry in record["course"].items():
        losses.append({"field": field, "value": entry.get("value"),
                       "reason": "document_type_not_academic",
                       "evidence": (entry.get("evidence") or {}).get("text", "")[:120]})
    for item in record["grading"]:
        losses.append({"field": "grading", "value": f"{item['category']} {item['weight']}%",
                       "reason": "document_type_not_academic",
                       "evidence": (item.get("evidence") or {}).get("text", "")[:120]})
    for item in record["meetings"]:
        losses.append({"field": "meetings", "value": item.get("start_time"),
                       "reason": "document_type_not_academic",
                       "evidence": (item.get("evidence") or {}).get("text", "")[:120]})
    for field, entry in record["policies"].items():
        losses.append({"field": field, "value": entry.get("value"),
                       "reason": "document_type_not_academic",
                       "evidence": (entry.get("evidence") or {}).get("text", "")[:120]})
    return losses


def _warnings(record: dict[str, Any], sections: list[dict[str, Any]],
              cleaned: str, evidence: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not sections:
        warnings.append("no section heading was recognised; fields were read from whole lines")
    if record.get("grading") and not record.get("grading_total_consistent", True):
        warnings.append(
            f"grading weights sum to {record.get('grading_total_percent')}, not 100: "
            "the document may print nested sub-weights or the reader may have missed a row")
    ungrounded = [item["field"] for item in evidence
                  if item["start"] is not None and not item["grounded_in_source"]]
    if ungrounded:
        warnings.append(f"{len(ungrounded)} value(s) do not match their own source span: "
                        f"{sorted(set(ungrounded))}")
    if not record["course"] and not record["grading"] and not record["meetings"]:
        warnings.append("nothing was recognised as this document's course, grading or meetings")
    if len(cleaned) > 60_000:
        warnings.append("very long document: sections are read up to the detector's limit, "
                        "nothing beyond it is silently summarised")
    return warnings


# --------------------------------------------------------------------------- #
# Module-level convenience, with one reader per checkpoint
# --------------------------------------------------------------------------- #

def reader(checkpoint: str | Path = DEFAULT_CHECKPOINT,
           context_checkpoint: str | Path | None = DEFAULT_CONTEXT_CHECKPOINT,
           device: str | None = None) -> AcademicReader:
    key = (str(checkpoint), str(context_checkpoint), str(device))
    if key not in _READERS:
        _READERS[key] = AcademicReader(checkpoint, context_checkpoint, device=device)
    return _READERS[key]


def understand_academic_document(text: str, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                                 context_checkpoint: str | Path | None = DEFAULT_CONTEXT_CHECKPOINT,
                                 device: str | None = None,
                                 min_confidence: float = 0.0) -> dict[str, Any]:
    """Read one document. The only call an application needs to make."""
    return reader(checkpoint, context_checkpoint, device=device).read(
        text, min_confidence=min_confidence)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Read an academic document into the app record")
    parser.add_argument("path", help="UTF-8 text file, or - for stdin")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--context-checkpoint", default=DEFAULT_CONTEXT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--validate", action="store_true",
                        help="exit non-zero if the record breaks its own schema")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    import sys

    text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    result = understand_academic_document(text, checkpoint=args.checkpoint,
                                          context_checkpoint=args.context_checkpoint,
                                          device=args.device,
                                          min_confidence=args.min_confidence)
    if args.validate:
        problems = validate(result)
        if problems:
            raise SystemExit("IR validation failed: " + "; ".join(problems))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
