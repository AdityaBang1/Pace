import json
import sys
from pathlib import Path

from pace.field_corpus import clean_html
from pace.field_extract import _clean_candidate, _lines

targets = set(sys.argv[1:])
for path in ("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl",
             "data/raw/field-eval/syllabi.jsonl"):
    if not Path(path).exists():
        continue
    for line in Path(path).open(encoding="utf-8"):
        record = json.loads(line)
        if record["id"][:12] not in targets:
            continue
        text, _ = clean_html(record["text"])
        print("=" * 60, record["id"][:12])
        for index, (raw, start, _) in enumerate(_lines(text[:800])):
            cleaned = _clean_candidate(raw)
            if cleaned:
                print(f"{index:>3} {start:>4} {cleaned!r}")
