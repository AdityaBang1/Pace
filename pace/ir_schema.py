"""Identity, versioning and validation for the record an application consumes.

The UI needs to attach state to things -- this row was accepted, that value was
edited, this course is the one the schedule already mentions, this import can be
undone. None of that can be keyed on a string, because strings repeat (a
syllabus can print two "Homework" rows) and because they change (the user fixes
a typo in the title). So every object in the record gets an id that is derived
from where it came from, not from when it was read.

Deterministic ids have one property worth stating out loud: reading the same
document twice, on two machines, produces the same ids. That is what makes the
Python reference and the mobile engine comparable, and what lets an import be
replayed without duplicating rows.

Ids are *not* meant to survive the user editing the source document. A changed
document is a new document, with a new ``document.id`` and new children.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

#: Bumped when the shape of the record changes in a way a consumer can notice.
#: 0.3 adds ``document.id``, ``entities``, ``relations``, per-object ids, the
#: document-type gate fields and the ``schema`` block.
SCHEMA_VERSION = "pace.academic-ir/0.3"

#: Versions this module can still read. ``migrate`` upgrades them in memory.
SUPPORTED_VERSIONS = ("pace.academic-ir/0.2", SCHEMA_VERSION)

MIGRATIONS = {"pace.academic-ir/0.2": "pace.academic-ir/0.3"}

_PREFIXES = {
    "document": "doc",
    "section": "sec",
    "entity": "ent",
    "evidence": "ev",
    "grading": "grd",
    "meeting": "mtg",
    "policy": "pol",
    "relation": "rel",
    "import": "imp",
}


def digest(*parts: Any, length: int = 12) -> str:
    """A short, stable hash of the parts that identify an object."""
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def document_id(text: str) -> str:
    """The id of the *source text*, not of the read."""
    return "doc_" + digest(text, length=16)


def stable_id(kind: str, document: str, *parts: Any) -> str:
    """The id of one object inside one document.

    ``kind`` selects the prefix; the document id and the identifying parts make
    the hash. Two identical rows in the same document are told apart by the
    caller passing their offsets.
    """
    if kind not in _PREFIXES:
        raise KeyError(f"unknown id kind {kind!r}")
    return f"{_PREFIXES[kind]}_{digest(document, kind, *parts)}"


def migrate(record: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a 0.2 record to the current shape, in place, and return it.

    A 0.2 record has no ids and no entities. The ids it would have had are
    recoverable from its own evidence offsets, so the upgrade is deterministic
    rather than random.
    """
    version = record.get("schema_version")
    if version == SCHEMA_VERSION:
        return record
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported Academic IR version {version!r}")

    text = _source_text(record)
    doc_id = record.setdefault("document", {}).setdefault("id", document_id(text))
    record["document"].setdefault("gate", "open")
    record["schema_version"] = SCHEMA_VERSION

    evidence = record.setdefault("evidence", [])
    for item in evidence:
        if "id" not in item:
            item["id"] = stable_id("evidence", doc_id, item.get("field"),
                                   item.get("start"), item.get("end"),
                                   item.get("value"))
    for index, section in enumerate(record.get("sections") or []):
        section.setdefault("id", stable_id("section", doc_id, index,
                                           section.get("start"), section.get("label")))
    for index, item in enumerate(record.get("grading") or []):
        item.setdefault("id", stable_id("grading", doc_id, index,
                                        item.get("category"), item.get("weight")))
    for index, item in enumerate(record.get("meetings") or []):
        item.setdefault("id", stable_id("meeting", doc_id, index,
                                        item.get("days"), item.get("start_time")))
    record.setdefault("entities", [])
    record.setdefault("relations", [])
    return record


def _source_text(record: dict[str, Any]) -> str:
    """The best stand-in for the source when only a record is at hand.

    A 0.2 record kept its evidence text but not the document, so the document id
    is derived from the evidence in order. It is stable for that record even
    though it will not equal the id the reader would compute from the text.
    """
    parts = [record.get("schema_version") or ""]
    for item in record.get("evidence") or []:
        parts.append(str(item.get("source_text") or ""))
        parts.append(str(item.get("start")))
    return "\n".join(parts)


def validate(record: dict[str, Any]) -> list[str]:
    """Every promise the shape makes, checked. Returns problems, not a bool.

    Kept dependency-free so both the Python tests and the export script can call
    it, and so the mobile port has an unambiguous reference for the rules.
    """
    problems: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version is {record.get('schema_version')!r}")
    for key in ("document", "course", "grading", "meetings", "policies", "sections",
                "entities", "relations", "evidence", "confidence", "model",
                "warnings"):
        if key not in record:
            problems.append(f"missing key {key}")
    document = record.get("document") or {}
    if not document.get("id"):
        problems.append("document.id is missing")
    if document.get("gate") not in {"open", "blocked"}:
        problems.append("document.gate is not open/blocked")
    if record.get("requires_confirmation") is not True:
        problems.append("requires_confirmation is not true")

    seen: set[str] = set()
    for collection in ("sections", "entities", "relations", "evidence"):
        for item in record.get(collection) or []:
            identifier = item.get("id")
            if not identifier:
                problems.append(f"{collection} item without id")
                continue
            if identifier in seen:
                problems.append(f"duplicate id {identifier}")
            seen.add(identifier)

    evidence_ids = {item.get("id") for item in record.get("evidence") or []}
    for entity in record.get("entities") or []:
        if entity.get("evidence_id") and entity["evidence_id"] not in evidence_ids:
            problems.append(f"entity {entity.get('id')} points at missing evidence")
    for relation in record.get("relations") or []:
        if relation.get("evidence_id") and relation["evidence_id"] not in evidence_ids:
            problems.append(f"relation {relation.get('id')} points at missing evidence")
        for end in ("from", "to"):
            target = relation.get(end)
            if target and str(target).startswith(("ent_", "grd_", "mtg_", "pol_")) \
                    and target not in seen:
                problems.append(f"relation {relation.get('id')} has dangling {end}")

    for item in record.get("evidence") or []:
        if item.get("requires_confirmation") is not True:
            problems.append(f"evidence {item.get('id')} is not confirmation-gated")
        start, end = item.get("start"), item.get("end")
        if isinstance(start, int) and isinstance(end, int) and not (0 <= start < end):
            problems.append(f"evidence {item.get('id')} has a bad span")
    return problems


def entity_types(record: dict[str, Any]) -> Iterable[str]:
    """The entity kinds present, for a UI that wants to group them."""
    return sorted({str(entity.get("kind")) for entity in record.get("entities") or []})
