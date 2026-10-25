import json
import sys
from pathlib import Path

from pace.field_corpus import clean_html
from pace.field_extract import extract_course_name, type_sections
from pace.field_ledger import Ledger
from pace.field_model import FieldEngine

targets = set(sys.argv[1:])
engine = FieldEngine("artifacts/field-classifier-minilm", device="cpu")
context = (FieldEngine("artifacts/field-classifier-context", device="cpu")
           if Path("artifacts/field-classifier-context").exists() else None)
for path in ("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl",
             "data/raw/field-eval/syllabi.jsonl"):
    if not Path(path).exists():
        continue
    for line in Path(path).open(encoding="utf-8"):
        record = json.loads(line)
        if record["id"][:12] not in targets:
            continue
        text, _ = clean_html(record["text"])
        sections = type_sections(text, engine, context, 0.8)
        ledger = Ledger()
        value = extract_course_name(text, engine, sections, ledger=ledger)
        print("=" * 60, record["id"][:12], "->", None if value is None else value["value"])
        for item in ledger.losses():
            if item["field"] == "course_title":
                print("  ", repr(item["value"]), "|", item["reason"], "|", item["evidence"][:70])
