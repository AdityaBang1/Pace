"""Export everything the mobile engine needs, and prove it matches Python.

The app must not carry a second, quietly different implementation of the
reader. This script is the bridge: it takes the frozen checkpoint from
:mod:`pace.config`, writes mobile-ready artifacts into ``assets/models``, runs
each artifact back through ONNX Runtime (or its portable weights) on the same
fixtures the app will use, and writes the expected Academic IR next to them.

Two artifacts are graphs and one is weights:

* ``section-model.onnx`` / ``section-model-int8.onnx`` -- the section reader.
  Both are exported and measured; the manifest ships the one that keeps parity.
* ``doc-type.json`` + ``doc-type.bin`` -- the document gate. scikit-learn's
  char_wb TF-IDF has no ONNX converter, so the fitted pipeline is flattened to
  its vocabulary, IDFs, per-fold logistic weights and calibration curves. The
  TypeScript port reads exactly those numbers.
* ``matcher.json`` -- the pairwise course matcher's 14 coefficients.

Outputs::

    assets/models/…            the artifacts the app loads
    assets/models/manifest.json  ids, sizes, hashes, labels, chosen files
    fixtures/*.txt             the documents both engines read
    fixtures/expected/*.json   the Academic IR Python produces
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from pace.api import understand_academic_document
from pace.config import READER

FIXTURES = Path("fixtures")
ASSETS = Path("assets/models")

FIXTURE_TEXT: dict[str, str] = {
    "syllabus_math.txt": """MATH 1A: Calculus I
Fall 2026

Instructor: Dr. Lena Ortiz
Office Hours: Mondays 3-5pm in Room 214

Meeting Times
TU TH 11:00-12:15 in Evans 10

Grading
Problem Sets 30%
Midterm Exam 30%
Final Exam 40%

Attendance Policy
Students may miss two lectures without penalty. Late work loses 10% per day.
""",
    "syllabus_chem.txt": """AP Chemistry
Course Syllabus 2026-2027

Teacher: Dr. Allison Johnson
Email: ajohnson@northside.edu

Class meets Monday, Wednesday, Friday 8:00-8:50am in Room 118.

Grading Policy
Tests 40%
Labs 25%
Homework 20%
Final Exam 15%

Academic Integrity
All lab reports must be your own work.
""",
    "schedule.txt": """Student Schedule - Fall 2026

Period 1  AP Chemistry        Dr. Johnson     Room 118   M W F 8:00-8:50
Period 2  AP Calculus BC      Ms. Chen        Room 204   M T W TH F 9:05-9:55
Period 3  US History          Mr. Williams    Room 310   M W F 10:10-11:00
Period 4  Spanish III         Sra. Ruiz       Room 122   M T TH F 11:15-12:05
""",
    "catalog.txt": """Course Catalog 2026-2027
Mathematics

MATH 1A. Calculus I. 4 units.
Limits, derivatives, and integrals of functions of one variable.
Prerequisite: MATH 0B or placement examination.

MATH 1B. Calculus II. 4 units.
Techniques of integration, sequences and series, and applications.
Prerequisite: MATH 1A.

CHEM 1A. General Chemistry. 4 units.
Stoichiometry, atomic structure, and chemical bonding.
Corequisite: CHEM 1L.
""",
    "receipt.txt": """BLUE BOTTLE COFFEE
Order #4471        Table 12
2 x Cortado            9.00
1 x Almond Croissant   4.25
Subtotal               13.25
Tax                     1.16
Total                  14.41
Card **** 8821   Tip 20%
Thank you
""",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_fixtures() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "expected").mkdir(parents=True, exist_ok=True)
    for name, text in FIXTURE_TEXT.items():
        (FIXTURES / name).write_text(text, encoding="utf-8")


def _fixture_heads() -> list[str]:
    return [line for name in ("syllabus_math.txt", "syllabus_chem.txt", "catalog.txt")
            for line in FIXTURE_TEXT[name].splitlines() if line.strip()][:24]


def export_encoder(checkpoint_dir: str, output_name: str,
                   probe_texts: list[str]) -> dict[str, Any]:
    """One MiniLM checkpoint -> ONNX fp32 (+ int8), with parity on probe texts."""
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    from pace.field_model import FIELD_TASKS
    from pace.model import PaceClassifier

    checkpoint = Path(str(checkpoint_dir)) / "model.pt"
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels: list[str] = list(meta["labels"])
    max_length = int(meta.get("max_length", 256))
    config_dir = str(checkpoint.parent / meta.get("encoder_config", "encoder_config"))
    model = PaceClassifier(meta["base_model"], config_dir=config_dir,
                           num_classes=len(labels))
    model.load_state_dict(meta["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint.parent / meta.get("tokenizer_dir", "tokenizer")),
        local_files_only=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids, attention_mask)

    ASSETS.mkdir(parents=True, exist_ok=True)
    fp32 = ASSETS / output_name
    int8 = ASSETS / output_name.replace(".onnx", "-int8.onnx")
    if not fp32.exists():
        torch.onnx.export(
            Wrapper(model).eval(),
            (torch.ones((1, max_length), dtype=torch.int64),
             torch.ones((1, max_length), dtype=torch.int64)),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                          "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    if not int8.exists():
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    texts = probe_texts
    encoded = [tokenizer(text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt") for text in texts]
    torch_logits = []
    for row in encoded:
        with torch.inference_mode():
            torch_logits.append(model(row["input_ids"], row["attention_mask"]).numpy())
    reference = np.concatenate(torch_logits)

    def parity(path: Path) -> tuple[float, float, float]:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = []
        started = time.perf_counter()
        for row in encoded:
            out.append(session.run(["logits"], {
                "input_ids": row["input_ids"].numpy().astype(np.int64),
                "attention_mask": row["attention_mask"].numpy().astype(np.int64)})[0])
        elapsed = (time.perf_counter() - started) * 1000 / len(encoded)
        logits = np.concatenate(out)
        agreement = float(np.mean(reference.argmax(-1) == logits.argmax(-1)))
        return agreement, float(np.max(np.abs(reference - logits))), elapsed

    fp32_agreement, fp32_diff, fp32_ms = parity(fp32)
    int8_agreement, int8_diff, int8_ms = parity(int8)
    # int8 has to keep the label on enough rows to be worth its risk. On this
    # corpus it does not: the section graph held 0.875 on 24 headings and the
    # context graph 0.758 on 400 rows, so fp32 ships and the size is a known cost.
    measured_enough = len(texts) >= 50
    shipped = int8 if (int8_agreement >= 0.99 and measured_enough) else fp32
    return {
        "file": shipped.name,
        "bytes": shipped.stat().st_size,
        "sha256": sha256(shipped),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "labels": labels,
        "tasks": {label: FIELD_TASKS.get(label, "other") for label in labels},
        "max_length": max_length,
        "temperature": float(meta.get("temperature", 1.0) or 1.0),
        "candidates": {
            fp32.name: {"bytes": fp32.stat().st_size, "sha256": sha256(fp32),
                        "label_agreement": fp32_agreement,
                        "max_abs_logit_diff": fp32_diff,
                        "cpu_single_row_ms": round(fp32_ms, 2)},
            int8.name: {"bytes": int8.stat().st_size, "sha256": sha256(int8),
                        "label_agreement": int8_agreement,
                        "max_abs_logit_diff": int8_diff,
                        "cpu_single_row_ms": round(int8_ms, 2)},
        },
        "parity": {"rows": len(texts), "label_agreement": int8_agreement
                   if shipped is int8 else fp32_agreement},
        "shipped_because": ("int8 kept label agreement on >=50 rows and is a quarter of the size"
                            if shipped is int8 else
                            "int8 did not keep label agreement on enough rows"),
    }


def export_section_model() -> dict[str, Any]:
    return export_encoder(str(READER["section_checkpoint"]), "section-model.onnx",
                          _fixture_heads())


def export_context_model() -> dict[str, Any]:
    """The heading+value fallback model.

    The app shipped without it first and two of the five fixture sections came out
    wrong ("Meeting Times" as office hours at 0.68), because the fallback is what
    the Python reader uses when a heading is unfamiliar. Shipping it costs ~90 MB
    and buys exact section and policy parity, so it is shipped.
    """
    probes = ["Grading: Homework 30% Midterm Exam 30% Final Exam 40%",
              "Meeting Times: TU TH 11:00-12:15 in Evans 10",
              "Attendance Policy: Students may miss two lectures without penalty.",
              "Academic Integrity: All lab reports must be your own work.",
              "Office Hours: Mondays 3-5pm in Room 214",
              "Prerequisite: MATH 1A."]
    return export_encoder(str(READER["context_checkpoint"]), "context-model.onnx", probes)


def export_doc_type() -> dict[str, Any]:
    """Flatten the fitted gate to vocabulary, IDFs, weights and calibrators."""
    import joblib

    from pace.doc_type import DocumentTypeClassifier

    classifier = DocumentTypeClassifier.load("artifacts/doc-type")
    pipeline = classifier.pipeline
    features = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]
    blocks = []
    matrices = []
    offset = 0
    for name, vectorizer in features.transformer_list:
        vocabulary = vectorizer.vocabulary_
        idf = vectorizer.idf_.astype(np.float32)
        width = len(vocabulary)
        blocks.append({
            "name": name,
            "analyzer": vectorizer.analyzer,
            "ngram_min": int(vectorizer.ngram_range[0]),
            "ngram_max": int(vectorizer.ngram_range[1]),
            "lowercase": bool(vectorizer.lowercase),
            "sublinear_tf": bool(vectorizer.sublinear_tf),
            "token_pattern": (vectorizer.token_pattern
                              if vectorizer.analyzer == "word" else None),
            "terms": [term for term, _ in sorted(vocabulary.items(), key=lambda kv: kv[1])],
            "idf": [round(float(value), 6) for value in idf],
            "weight_offset": offset,
            "weight_width": width,
        })
        offset += width

    folds = []
    for fold in model.calibrated_classifiers_:
        coefficients = fold.estimator.coef_.astype(np.float32)
        matrices.append(coefficients)
        folds.append({
            "intercept": [float(value) for value in fold.estimator.intercept_],
            "calibration": [[float(c.a_), float(c.b_)] for c in fold.calibrators],
            "class_order": [list(map(str, model.classes_)).index(str(value))
                            for value in fold.classes],
        })
    weights = np.concatenate(matrices, axis=0).astype("<f4")
    (ASSETS / "doc-type.bin").write_bytes(weights.tobytes())

    payload = {
        "classes": [str(name) for name in model.classes_],
        "threshold": classifier.threshold,
        "blocks": blocks,
        "folds": folds,
        "weights": {"file": "doc-type.bin", "dtype": "float32-le",
                    "shape": list(weights.shape),
                    "fold_rows": [matrix.shape[0] for matrix in matrices]},
    }
    target = ASSETS / "doc-type.model"
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    # Agreement with sklearn itself, on the fixtures and on one negative each.
    agreements, diffs = [], []
    for text in list(FIXTURE_TEXT.values()) + ["Late work loses 10% per day."]:
        reference = classifier.probabilities([text])[0]
        ours = _doc_type_from_payload(payload, weights, text)
        order = payload["classes"]
        agreements.append(int(np.argmax([ours[name] for name in order])) ==
                          int(np.argmax([reference.get(name, 0.0) for name in order])))
        diffs.append(max(abs(ours[name] - float(reference.get(name, 0.0))) for name in order))
    return {
        "file": target.name,
        "weights_file": "doc-type.bin",
        "bytes": target.stat().st_size + (ASSETS / "doc-type.bin").stat().st_size,
        "sha256": sha256(target),
        "classes": payload["classes"],
        "threshold": payload["threshold"],
        "features": int(offset),
        "parity": {"rows": len(agreements),
                   "label_agreement": float(np.mean(agreements)),
                   "max_abs_probability_diff": float(max(diffs))},
    }


def _doc_type_from_payload(payload: dict[str, Any], weights: np.ndarray,
                           text: str) -> dict[str, float]:
    """A reference implementation of the flattened gate, used for verification."""
    prepared = " ".join((text or "").split())
    vector = np.zeros(sum(block["weight_width"] for block in payload["blocks"]))
    for block in payload["blocks"]:
        counts: dict[str, int] = {}
        terms = block["terms"]
        index = {term: position for position, term in enumerate(terms)}
        for gram in _ngrams(prepared, block):
            position = index.get(gram)
            if position is not None:
                counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            value = 1.0 + np.log(count) if block["sublinear_tf"] else float(count)
            vector[block["weight_offset"] + index[gram]] = value * block["idf"][index[gram]]
        start = block["weight_offset"]
        stop = start + block["weight_width"]
        norm = np.linalg.norm(vector[start:stop])
        if norm:
            vector[start:stop] /= norm
    probabilities = np.zeros(len(payload["classes"]))
    row = 0
    for fold in payload["folds"]:
        rows = len(fold["class_order"])
        decision = weights[row:row + rows] @ vector + np.asarray(fold["intercept"])
        row += rows
        calibrated = []
        for position in range(rows):
            a, b = fold["calibration"][position]
            # sklearn's Platt scaling is expit(-(a*x + b)), not the other sign.
            calibrated.append(1.0 / (1.0 + np.exp(a * decision[position] + b)))
        calibrated = np.asarray(calibrated)
        calibrated /= calibrated.sum()
        probabilities += calibrated
    probabilities /= len(payload["folds"])
    return dict(zip(payload["classes"], probabilities))


def _ngrams(text: str, block: dict[str, Any]):
    low = text.lower() if block["lowercase"] else text
    if block["analyzer"] == "word":
        import re

        tokens = re.findall(block["token_pattern"], low, flags=re.UNICODE)
        for size in range(block["ngram_min"], block["ngram_max"] + 1):
            for start in range(len(tokens) - size + 1):
                yield " ".join(tokens[start:start + size])
    else:
        for word in low.split():
            padded = f" {word} "
            for size in range(block["ngram_min"], block["ngram_max"] + 1):
                for start in range(len(padded) - size + 1):
                    yield padded[start:start + size]


def export_matcher() -> dict[str, Any]:
    from pace.entity_resolution import FEATURE_NAMES, CourseMatcher

    matcher = CourseMatcher.load("artifacts/entity-resolution")
    target = ASSETS / "matcher.model"
    payload = {
        "features": FEATURE_NAMES,
        "coefficients": [float(value) for value in matcher.model.coef_[0]],
        "intercept": float(matcher.model.intercept_[0]),
        "threshold": matcher.threshold,
        "decision_policy": ("veto on level/name-number conflict; identical names with "
                            "different catalogue numbers are 'uncertain'"),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    probes = [("AP Chem", "AP Chemistry"), ("AP Calculus AB", "AP Calculus BC"),
              ("Biology", "AP Biology"), ("CHEM 1A", "CHEM 1L"),
              ("MATH 1A: Calculus", "MATH 1A Calculus I")]
    return {
        "file": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "threshold": matcher.threshold,
        "features": FEATURE_NAMES,
        "probes": [{"left": left, "right": right,
                    "probability": matcher.decide(left, right).probability,
                    "decision": matcher.decide(left, right).decision}
                   for left, right in probes],
    }


def copy_tokenizer() -> dict[str, Any]:
    """Copy the WordPiece vocabulary as a text asset the app can read.

    The app does not need the HF *tokenizer.json*: it implements the standard
    BERT basic tokenizer and WordPiece, and the vocabulary is the only data that
    has to travel. Everything else (lowercasing, accent stripping, max chars) is
    read from the config and recorded here.
    """
    source = Path(str(READER["section_checkpoint"])) / "tokenizer"
    target = ASSETS / "tokenizer"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "vocab.txt", target / "vocab.model")
    config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    return {
        "dir": "tokenizer",
        "files": {"vocab.model": sha256(target / "vocab.model")},
        "config": {
            "do_lower_case": bool(config.get("do_lower_case", True)),
            "strip_accents": config.get("strip_accents"),
            "tokenize_chinese_chars": bool(config.get("tokenize_chinese_chars", True)),
            "model_max_length": config.get("model_max_length"),
        },
    }


def export_rules() -> dict[str, Any]:
    """The heading vocabulary and policy cues, so the port reads the same words."""
    from pace.field_corpus import SYLLABUS_SECTION_MAP, _BOUNDARY_ONLY_HEADINGS
    from pace.field_extract import _POLICY_CUES, _SECTION_VALUES

    payload = {
        "section_map": SYLLABUS_SECTION_MAP,
        "boundary_only": sorted(_BOUNDARY_ONLY_HEADINGS),
        "heading_max_chars": 70,
        "section_values": _SECTION_VALUES,
        "policy_cues": {name: pattern.pattern for name, pattern in _POLICY_CUES.items()},
    }
    target = ASSETS / "rules.model"
    target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"file": target.name, "bytes": target.stat().st_size,
            "sha256": sha256(target), "sections": len(SYLLABUS_SECTION_MAP)}


def write_expected() -> list[dict[str, Any]]:
    written = []
    for name in FIXTURE_TEXT:
        text = (FIXTURES / name).read_text(encoding="utf-8")
        record = understand_academic_document(text)
        target = FIXTURES / "expected" / (Path(name).stem + ".json")
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        written.append({"fixture": name, "gate": record["document"]["gate"],
                        "document_type": (record["document"]["document_type"] or {}).get("label"),
                        "entities": len(record["entities"]),
                        "relations": len(record["relations"]),
                        "evidence": len(record["evidence"])})
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-section", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    write_fixtures()
    if args.force:
        for stale in ("section-model.onnx", "section-model-int8.onnx"):
            (ASSETS / stale).unlink(missing_ok=True)

    section = None
    context = None
    if args.skip_section:
        previous = ASSETS / "manifest.model"
        if previous.exists():
            section = json.loads(previous.read_text(encoding="utf-8"))["models"]["section"]
            section.pop("id", None)
            context = json.loads(previous.read_text(encoding="utf-8"))["models"].get("context")
            if context:
                context.pop("id", None)
    else:
        section = export_section_model()
        context = export_context_model()
    doc_type = export_doc_type()
    matcher = export_matcher()
    rules = export_rules()
    tokenizer = copy_tokenizer()
    manifest = {
        "engineVersion": str(READER["engine_version"]),
        "schemaVersion": str(READER["schema_version"]),
        "frozenOn": str(READER["frozen_on"]),
        "why": str(READER["why"]),
        "models": {
            "section": {"id": "field-classifier-minilm", **(section or {})},
            "context": {"id": "field-classifier-context", **(context or {})},
            "documentType": {"id": "doc-type", **doc_type},
            "entityResolution": {"id": "course-matcher", **matcher},
            "rules": {"id": "section-headings-and-policy-cues", **rules},
        },
        "tokenizer": tokenizer,
    }
    (ASSETS / "manifest.model").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = write_expected()
    print(json.dumps({"manifest": str(ASSETS / "manifest.model"), "fixtures": summary},
                     indent=2))


if __name__ == "__main__":
    main()
