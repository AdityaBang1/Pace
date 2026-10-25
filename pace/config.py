"""The one place that names the reader an application ships with.

Before this module the checkpoint path was written in three places (the API's
default, the CLI, the docs) and a fourth time in whatever script happened to be
open. Freezing the reader means one dictionary, imported by the API, the CLI and
the mobile asset exporter, so "which model is the app using" has a single
answer.

The choice is evidence-based and recorded, not arbitrary::

    field-classifier-minilm   22.7 M params, encoder, sections 0.9982 / 0.9711,
                              extraction gold 62/67 = 0.925
    jev-field-v4             15.3 M params, choice model, sections 0.9983 / 0.9734,
                              extraction gold 62/66 = 0.939, but picked
                              "Problem Sets 30%" as the course title of a sample
                              syllabus on 2026-10-24

So the encoder ships until title selection in the choice model is fixed. Swapping
is one edit here plus re-running ``scripts/export_mobile_assets.py``; no UI, no
database and no schema changes.
"""
from __future__ import annotations

from pathlib import Path

from .ir_schema import SCHEMA_VERSION

READER: dict[str, object] = {
    "engine_version": "0.3.0",
    "schema_version": SCHEMA_VERSION,
    #: The shipped section reader.
    "section_checkpoint": "artifacts/field-classifier-minilm",
    "context_checkpoint": "artifacts/field-classifier-context",
    #: The document-type gate. Its label ``other`` blocks extraction.
    "doc_type_dir": "artifacts/doc-type",
    #: Pairwise course matching (used for cross-document proposals).
    "matcher_dir": "artifacts/entity-resolution",
    #: Section-label bands, from the measured sweep.
    "bands": ((0.97, "high"), (0.85, "medium"), (0.0, "low")),
    "frozen_on": "2026-10-24",
    "why": ("higher extraction gold than the choice model on the same 12 documents, "
            "and no live course-title regression"),
}

#: Labels that mean "this is not an academic document". A gate hit suppresses
#: every extracted value instead of returning them with a warning.
NON_ACADEMIC = "other"


def section_checkpoint() -> Path:
    return Path(str(READER["section_checkpoint"]))


def context_checkpoint() -> Path:
    return Path(str(READER["context_checkpoint"]))


def doc_type_dir() -> Path:
    return Path(str(READER["doc_type_dir"]))


def matcher_dir() -> Path:
    return Path(str(READER["matcher_dir"]))
