import json
from pathlib import Path

from pace.field_extract import extract_record
from pace.field_extract_eval import (DEFAULT_DEVELOPMENT_SCHOOLS, TEST_EXCLUDED_DOCUMENTS,
                                     split_of)
from pace.field_ledger import suppressed_values
from pace.field_model import FieldEngine

plan = json.loads(Path("data/processed/field/fold_plan.json").read_text(encoding="utf-8"))
holdout = {school for fold in plan for school in fold}
engine = FieldEngine("artifacts/field-classifier-minilm", device="cpu")
context = FieldEngine("artifacts/field-classifier-context", device="cpu")
for line in Path("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl").open(
        encoding="utf-8"):
    record = json.loads(line)
    if record.get("school") not in holdout or record["school"] in DEFAULT_DEVELOPMENT_SCHOOLS:
        continue
    if record["id"] in TEST_EXCLUDED_DOCUMENTS or split_of(record.get("school")) != "dev":
        continue
    result = extract_record(record["text"], engine, context, min_confidence=0.0)
    title = (result.get("course") or {}).get("course_title")
    losses = [value for value in suppressed_values(result) if value["field"] == "course_title"]
    abstained = any(value["value"] is None for value in losses)
    if title or abstained:
        print(json.dumps({"id": record["id"][:12],
                          "emitted": None if not title else title["value"],
                          "rule": None if not title else title.get("rule"),
                          "confidence": None if not title else title["confidence"],
                          "abstained": abstained}, ensure_ascii=False))
