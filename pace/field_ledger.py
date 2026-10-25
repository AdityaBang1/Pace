"""What the extractor read and did not emit, and how those losses are counted.

Extraction decides whether a value is good enough for the record; this module
decides what happens to the ones that are not. Keeping the two apart means a
rule can log a loss without knowing the record's JSON shape, and the
evaluation can count losses without re-deriving a reason from rule text.

One shape, one place:

- :class:`Ledger` is what the extractor writes to. ``note`` fixes the JSON
  shape (``field``, ``value``, ``reason``, ``evidence``) and the evidence
  truncation, so a new rule cannot invent a fifth key or forget the reason.
- :func:`suppressed_values` is the read side the record and its callers use.
- :func:`summarize` folds per-item reasons (``duplicate of 'Homework'``,
  ``component of 'Exams'``) to their rule and counts them per field, so the
  counts add up to the field's loss total instead of hiding in a long tail of
  one-off labels.
"""
from __future__ import annotations

from typing import Any, Iterable

# Long enough to identify the line a value came from, short enough that a
# suppressed list over a whole document stays readable.
EVIDENCE_CHARS = 120


class Ledger:
    """Every value read and not emitted, in the order it was dropped."""

    def __init__(self) -> None:
        self._losses: list[dict[str, Any]] = []

    def note(self, field: str, value: Any, reason: str, evidence: Any = "") -> None:
        self._losses.append({
            "field": field,
            "value": value,
            "reason": reason,
            "evidence": str(evidence or "")[:EVIDENCE_CHARS],
        })

    def extend(self, field: str, items: Iterable[dict[str, Any]]) -> None:
        """Record candidates a rule collected and then discarded as a batch."""
        for item in items:
            self.note(field, item.get("value"), str(item.get("reason")),
                      item.get("evidence"))

    def losses(self) -> list[dict[str, Any]]:
        return list(self._losses)

    def __len__(self) -> int:
        return len(self._losses)


def suppressed_values(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The record's losses, without the evidence offsets.

    This is the public contract: a reader can count what the extractor threw
    away instead of taking the emitted precision on trust.
    """
    return [{"field": item.get("field"), "value": item.get("value"),
             "reason": item.get("reason")}
            for item in record.get("suppressed") or []]


def reason_group(reason: str) -> str:
    """Fold a per-item reason to the rule that produced it.

    ``duplicate of 'font-size'`` and ``duplicate of 'Homework'`` are one rule;
    left unfolded they become a tail of single-count keys and a table of
    "top reasons" silently covers fewer losses than it claims.
    """
    if reason.startswith("duplicate of '"):
        return "duplicate of an earlier row"
    if reason.startswith("component of '"):
        return "a component of the parent weight"
    return reason


def summarize(losses: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Losses per field per folded reason, most frequent reason first."""
    counts: dict[str, dict[str, int]] = {}
    for item in losses:
        field = str(item.get("field") or "unknown")
        reason = reason_group(str(item.get("reason")))
        bucket = counts.setdefault(field, {})
        bucket[reason] = bucket.get(reason, 0) + 1
    return {field: dict(sorted(bucket.items(), key=lambda pair: (-pair[1], pair[0])))
            for field, bucket in sorted(counts.items())}
