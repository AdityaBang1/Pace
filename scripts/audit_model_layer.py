"""What the model layer actually does today.

Every number this writes is measured on this checkout, not quoted from a
document. The probe is deliberately read-only: it loads the shipped
checkpoints, runs real text through the same entry points the CLI uses, and
writes one JSON file per question under ``artifacts/model-audit``.

    python scripts/audit_model_layer.py --out artifacts/model-audit
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Corpus helpers
# --------------------------------------------------------------------------- #

REAL_SYLLABI = Path("data/raw/field-eval/syllabi.jsonl")
FIELD_TEST = Path("data/processed/field_mix_ext/test.jsonl")
INGESTED = Path("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl")

NON_ACADEMIC = {
    "receipt": """CORNER MARKET
123 Elm Street
Date: 10/14/2026   Time: 14:32
2 x Coffee ........ 3.50
1 x Sandwich ...... 7.25
Subtotal          10.75
Tax                0.86
TOTAL             11.61
VISA **** 4412
Thank you for your purchase""",
    "menu": """THE BLUE PLATE
Starters
Soup of the day 6
Garlic bread 5
Mains
Roast chicken 18
Mushroom risotto 16
Sides
Fries 4
Salad 5
Open Tue-Sun 5pm-10pm
Kitchen closes 9:30pm""",
    "article": """Why cities are planting more trees

Urban heat is not evenly distributed. A study of 300 neighbourhoods found that
blocks with mature canopy were up to four degrees cooler on summer afternoons
than bare blocks a few streets away. The authors argue that planting budgets
should follow surface temperature rather than population alone, and that
maintenance matters more than the number of saplings.""",
    "bank_statement": """STATEMENT PERIOD 01/09/2026 - 30/09/2026
Opening balance        1,204.55
09/03  Direct deposit   2,100.00
09/05  Rent            -1,450.00
09/11  Groceries         -212.40
09/18  Transfer out      -300.00
Closing balance        1,342.15
Interest earned            0.00""",
}

#: Substitutions a scanner actually makes: shape-identical glyphs, not random noise.
OCR_SWAPS = [("0", "O"), ("1", "l"), ("5", "S"), ("8", "B"), ("m", "rn"),
             ("2", "Z"), ("6", "G"), ("9", "g")]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Read by newline only: real captured text contains form feeds and Unicode
    # line separators, which str.splitlines() would treat as record breaks.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corrupt(text: str, rate: float, rng: random.Random) -> str:
    """Replace a fraction of tokens' glyphs the way a bad scan would."""
    out: list[str] = []
    for token in text.split(" "):
        if token and rng.random() < rate:
            source, target = rng.choice(OCR_SWAPS)
            out.append(token.replace(source, target, 1) if source in token else token)
        else:
            out.append(token)
    return " ".join(out)


def pick_documents(count: int = 3) -> list[dict[str, Any]]:
    """Real syllabi the extractor's gold set never touched."""
    rows = load_jsonl(REAL_SYLLABI)
    preferred = [row for row in rows if row["school"] == "Yale University"]
    return (preferred or rows)[:count]


# --------------------------------------------------------------------------- #
# 1. Inventory
# --------------------------------------------------------------------------- #

def inventory() -> dict[str, Any]:
    import torch

    from pace.field_model import FieldEngine
    from pace.jev_model import JevEngine

    report: dict[str, Any] = {"torch": torch.__version__,
                             "cuda_available": torch.cuda.is_available()}
    for name, path in (("jev-field-v2", "artifacts/jev-field-v2"),
                       ("jev-field", "artifacts/jev-field"),
                       ("field-classifier-minilm", "artifacts/field-classifier-minilm"),
                       ("field-classifier-context", "artifacts/field-classifier-context")):
        directory = Path(path)
        if not directory.exists():
            report[name] = {"missing": str(directory)}
            continue
        started = time.perf_counter()
        engine = (JevEngine(directory) if (directory / "config.json").exists()
                  and "descriptions" in json.loads((directory / "config.json").read_text("utf-8"))
                  else FieldEngine(directory))
        load_seconds = time.perf_counter() - started
        model = getattr(engine, "model", None)
        parameters = sum(p.numel() for p in model.parameters()) if model else None
        size = (directory / "model.pt").stat().st_size if (directory / "model.pt").exists() else None
        tokenizer = getattr(engine, "tokenizer", None)
        report[name] = {
            "load_seconds": round(load_seconds, 2),
            "parameters": parameters,
            "checkpoint_bytes": size,
            "checkpoint_mb": round(size / 1e6, 1) if size else None,
            "tokenizer": type(tokenizer).__name__,
            "max_tokens": (int(getattr(engine, "max_state", 0) or 0)
                           if name.startswith("jev") else 256),
            "labels": len(getattr(engine, "labels", [])),
            "temperature": getattr(engine, "temperature", None),
            "output_keys": sorted(engine.classify("Example text").keys()),
        }
    return report


# --------------------------------------------------------------------------- #
# 2. End to end on a real syllabus
# --------------------------------------------------------------------------- #

def end_to_end(documents: list[dict[str, Any]]) -> dict[str, Any]:
    from pace.field_corpus import detect_document_sections
    from pace.field_extract import extract_record, type_sections
    from pace.field_model import FieldEngine
    from pace.jev_model import JevEngine

    minilm = FieldEngine("artifacts/field-classifier-minilm")
    context = FieldEngine("artifacts/field-classifier-context")
    jev = JevEngine("artifacts/jev-field-v2")

    results = []
    for document in documents:
        text = document["text"]
        sections = detect_document_sections(text)
        timed = time.perf_counter()
        typed = type_sections(text, minilm, context, threshold=0.8)
        section_seconds = time.perf_counter() - timed
        record = extract_record(text, minilm, context, threshold=0.8)
        jev_headings = jev.classify_many([section["text"] for section in sections],
                                         top_k=2) if sections else []
        results.append({
            "document": {"school": document["school"], "id": document["id"][:12],
                         "characters": len(text)},
            "sections_detected": len(sections),
            "sections_typed": [
                {"heading": item["text"][:70], "label": item["label"],
                 "confidence": item["confidence"], "source": item["source"],
                 "jev_label": jev_headings[index]["label"] if index < len(jev_headings) else None,
                 "jev_percent": jev_headings[index]["percent"] if index < len(jev_headings) else None}
                for index, item in enumerate(typed)
            ],
            "record": {
                "course": {key: value["value"] for key, value in record["course"].items()},
                "grading": [{"category": item["category"], "weight": item["weight"],
                             "confidence": item["confidence"]}
                            for item in record["grading"]],
                "grading_total": record.get("grading_total_percent"),
                "meetings": [{"days": item["days"], "start": item["start_time"],
                              "end": item["end_time"], "room": item["room"]}
                             for item in record["meetings"]],
                "policies": {key: value["value"] for key, value in record["policies"].items()},
                "suppressed": len(record["suppressed"]),
            },
            "section_seconds": round(section_seconds, 2),
        })
    return {"documents": results}


# --------------------------------------------------------------------------- #
# 3. Non-academic input and OCR tolerance
# --------------------------------------------------------------------------- #

def off_domain(jev) -> dict[str, Any]:
    from pace.synthetic_doc_model import SyntheticDocumentEngine

    out: dict[str, Any] = {"field_menu": {}, "document_type": {}}
    for name, text in NON_ACADEMIC.items():
        result = jev.classify(text, top_k=3)
        out["field_menu"][name] = {"label": result["label"], "percent": result["percent"],
                                   "top": [(item["label"], item["percent"])
                                           for item in result["top"]]}
    # One line at a time, which is how the app will actually feed text in.
    lines = {
        "receipt_total": "TOTAL 11.61",
        "receipt_card": "VISA **** 4412",
        "menu_item": "Roast chicken 18",
        "menu_hours": "Open Tue-Sun 5pm-10pm",
        "article_sentence": "Urban heat is not evenly distributed.",
        "bank_balance": "Closing balance 1,342.15",
        "late_work": "Late work loses 10% per day.",
        "syllabus_weight": "Tests 40%",
    }
    out["lines"] = {name: {key: value for key, value in jev.classify(text, top_k=3).items()
                           if key in ("label", "percent", "top")}
                    for name, text in lines.items()}

    # Two document-type artifacts ship; record which one actually loads.
    loadable = None
    for candidate in ("artifacts/synthetic-doc-type", "artifacts/synthetic-multitask"):
        try:
            loadable = SyntheticDocumentEngine(candidate, device="cpu")
            out["document_type_checkpoint"] = candidate
            break
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            out.setdefault("document_type_load_failures", {})[candidate] = \
                f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    if loadable is None:
        return out
    engine = loadable
    for name, text in {**NON_ACADEMIC,
                       "syllabus": pick_documents(1)[0]["text"]}.items():
        result = engine.analyze_document(text)
        document_type = result["document_type"]
        out["document_type"][name] = {
            "label": document_type["label"],
            "confidence": round(document_type["confidence"], 4),
            "entities": len(result["entities"]),
            "entity_types": dict(Counter(entity["type"] for entity in result["entities"])),
        }
    return out


def ocr_tolerance(jev, limit: int = 400, seed: int = 20261024) -> dict[str, Any]:
    from pace.field_model import FieldEngine

    minilm = FieldEngine("artifacts/field-classifier-minilm")
    rows = load_jsonl(FIELD_TEST)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(limit, len(rows)))
    out: dict[str, Any] = {}
    for model, engine in (("jev-field-v2", jev), ("field-classifier-minilm", minilm)):
        scores = {}
        for rate in (0.0, 0.05, 0.15, 0.30):
            texts = [row["text"] if rate == 0 else corrupt(row["text"], rate, rng)
                     for row in sample]
            correct = 0
            for row, text in zip(sample, texts):
                if engine.classify(text, top_k=1)["label"] == row["label"]:
                    correct += 1
            scores[f"{rate:.2f}"] = round(correct / len(sample), 4)
        out[model] = scores
    out["rows"] = len(sample)
    out["note"] = "glyph substitutions of 0/O, 1/l, 5/S, 8/B, m/rn, 2/Z, 6/G, 9/g"
    return out


# --------------------------------------------------------------------------- #
# 4. Truncation, long input and latency
# --------------------------------------------------------------------------- #

def truncation(jev) -> dict[str, Any]:
    """What the state tower does when the text is longer than its window."""
    filler = "The course meets on Monday and Wednesday in the science building. "
    out = {}
    for words in (20, 60, 127, 200, 400):
        text = (filler * 20)[: words * 6]
        probs = jev.probabilities([text])[0]
        ranking = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        out[str(words)] = {"label": ranking[0][0], "percent": round(ranking[0][1] * 100, 2),
                           "tokens_sent": min(words, jev.max_state)}
    # A document-level ask through the field menu, which is what a naive caller
    # would do: the whole syllabus as one "state".
    document = pick_documents(1)[0]["text"]
    result = jev.classify(document, top_k=3)
    out["whole_syllabus"] = {"characters": len(document), "label": result["label"],
                             "percent": result["percent"],
                             "top": [(item["label"], item["percent"])
                                     for item in result["top"]]}
    return out


def latency(jev) -> dict[str, Any]:
    import torch

    from pace.field_model import FieldEngine
    from pace.jev_model import JevEngine

    texts = [row["text"] for row in load_jsonl(FIELD_TEST)[:512]]
    out: dict[str, Any] = {}
    for device in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        for name, factory in (("jev-field-v2", lambda d=device: JevEngine("artifacts/jev-field-v2", device=d)),
                              ("field-classifier-minilm",
                               lambda d=device: FieldEngine("artifacts/field-classifier-minilm", device=d))):
            engine = factory()
            engine.classify_many(texts[:16], top_k=1)  # warm up
            started = time.perf_counter()
            engine.classify_many(texts, top_k=1)
            elapsed = time.perf_counter() - started
            out[f"{name}@{device}"] = {
                "rows": len(texts),
                "total_seconds": round(elapsed, 2),
                "ms_per_row": round(elapsed / len(texts) * 1000, 2),
                "rows_per_second": round(len(texts) / elapsed, 1),
            }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Audit the current model layer")
    parser.add_argument("--out", default="artifacts/model-audit")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args(argv)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    from pace.jev_model import JevEngine

    jev = JevEngine("artifacts/jev-field-v2")
    documents = pick_documents(3)

    steps = {
        "00_inventory": inventory,
        "01_end_to_end": lambda: end_to_end(documents),
        "02_off_domain": lambda: off_domain(jev),
        "03_ocr": lambda: ocr_tolerance(jev),
        "04_truncation": lambda: truncation(jev),
        "05_latency": lambda: latency(jev),
    }
    for name, step in steps.items():
        if name in args.skip:
            continue
        started = time.perf_counter()
        result = step()
        path = output / f"{name}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"{name}: {round(time.perf_counter() - started, 1)}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
