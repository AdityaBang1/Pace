"""Compare encoder depths of the synthetic multitask model on accuracy, size and speed.

The three variants are trained on the same corpus, the same seed and the same
school/template hold-out, so the only difference is how many transformer layers
the shared encoder keeps. The comparison answers a size/accuracy question with
measurements instead of adjectives: parameters, checkpoint bytes, held-out span
F1, and the wall-clock cost of reading one document.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_VARIANTS = {
    "6 layers": "artifacts/synthetic-multitask-v8",
    "4 layers": "artifacts/synthetic-multitask-l4-v8",
    "2 layers": "artifacts/synthetic-multitask-l2-v8",
}


def _load_metrics(directory: str) -> dict[str, Any]:
    return json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))


def _window_latency(checkpoint_dir: str, runs: int = 50) -> dict[str, Any]:
    """Batch-1 forward time for one window, on CUDA when it is available."""
    from .synthetic_doc_model import PaceMultiTask
    from transformers import AutoTokenizer

    path = Path(checkpoint_dir)
    meta = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(path / meta["tokenizer_dir"], local_files_only=True)
    model = PaceMultiTask(meta["base_model"], str(path / meta["encoder_config"]))
    model.load_state_dict(meta["state_dict"])
    encoded = tokenizer("Course syllabus BIO 210 Cell Biology. Instructor: Dr. Avery Morgan.",
                        truncation=True, max_length=int(meta["max_length"]), return_tensors="pt")
    results: dict[str, Any] = {}
    for device_name in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        device = torch.device(device_name)
        model.to(device).eval()
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            for _ in range(5):
                model(ids, mask)
            samples = []
            for _ in range(runs):
                if device_name == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                model(ids, mask)
                if device_name == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
        results[device_name] = {
            "latency_ms_p50": statistics.median(samples),
            "latency_ms_p95": sorted(samples)[int(0.95 * (len(samples) - 1))],
        }
    model.to("cpu")
    return results


def _document_latency(checkpoint_dir: str, text: str, repeats: int = 3) -> dict[str, Any]:
    """End-to-end time to read one whole document through the sliding window."""
    from .synthetic_doc_model import SyntheticDocumentEngine

    engine = SyntheticDocumentEngine(checkpoint_dir, device=None)
    engine.analyze_document(text)
    samples = []
    windows = 0
    for _ in range(repeats):
        started = time.perf_counter()
        result = engine.analyze_document(text)
        samples.append((time.perf_counter() - started) * 1000)
        windows = result["window_count"]
    return {"device": str(engine.device), "windows": windows,
            "latency_ms_p50": statistics.median(samples), "characters": len(text)}


def compare(variants: dict[str, str], document_path: str | None = None) -> dict[str, Any]:
    text = ""
    if document_path:
        text = Path(document_path).read_text(encoding="utf-8")
    rows = []
    for label, directory in variants.items():
        path = Path(directory)
        if not (path / "metrics.json").exists():
            continue
        metrics = _load_metrics(directory)
        clean = metrics["entity_span_metrics_clean"]
        corrupted = metrics["entity_span_metrics_corrupted"]
        int8_path = path / "model_int8.pt"
        int8_report = {}
        if (path / "int8_report.json").exists():
            int8_report = json.loads((path / "int8_report.json").read_text(encoding="utf-8"))
        row = {
            "variant": label,
            "checkpoint_dir": directory,
            "encoder_layers": metrics.get("encoder_layers"),
            "parameters": metrics["parameter_count"],
            "checkpoint_bytes": (path / "model.pt").stat().st_size,
            "int8_checkpoint_bytes": int8_path.stat().st_size if int8_path.exists() else None,
            "int8_entity_span_f1": int8_report.get("entity_span_f1"),
            "document_type_accuracy_clean": metrics["test_clean_text"]["accuracy"],
            "document_type_accuracy_corrupted": metrics["test_ocr_text_corrupted"]["pytorch_multitask"]["accuracy"],
            "entity_span_f1_clean": clean["f1"],
            "entity_span_precision_clean": clean["precision"],
            "entity_span_recall_clean": clean["recall"],
            "entity_span_f1_corrupted": corrupted["f1"],
            "entity_span_f1_corrupted_context_only": metrics["entity_span_metrics_corrupted_context_only"]["f1"],
            "transcript_entity_span_f1": metrics["transcript_metrics"]["entity_span_f1"],
            "gold_spans": clean["true_spans"],
            "false_positive_spans": clean["false_positives"],
            "false_negative_spans": clean["false_negatives"],
        }
        row["window_latency"] = _window_latency(directory)
        if text:
            row["document_latency"] = _document_latency(directory, text)
        rows.append(row)
    payload = {
        "task": "synthetic academic document type + entity extraction",
        "variants": rows,
        "comparison_basis": (
            "Same seed, same generated corpus, same school/template hold-out for every variant. "
            "The only difference is the number of transformer layers the shared encoder keeps."
        ),
        "scope_limit": (
            "Synthetic-only measurements on procedurally generated text. They are not real-document, "
            "real OCR, institution-transfer or production evidence."
        ),
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare encoder depths of the synthetic multitask model.")
    parser.add_argument("--output", default="artifacts/synthetic-multitask/depth_comparison.json")
    parser.add_argument("--document", default="data/samples/mock_transcript_lepid.txt")
    args = parser.parse_args(argv)
    payload = compare(DEFAULT_VARIANTS, args.document)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in payload["variants"]:
        latency = row["window_latency"].get("cuda") or row["window_latency"]["cpu"]
        print(json.dumps({
            "variant": row["variant"],
            "encoder_layers": row["encoder_layers"],
            "parameters": row["parameters"],
            "checkpoint_bytes": row["checkpoint_bytes"],
            "int8_checkpoint_bytes": row["int8_checkpoint_bytes"],
            "entity_span_f1_clean": row["entity_span_f1_clean"],
            "entity_span_f1_corrupted": row["entity_span_f1_corrupted"],
            "document_type_accuracy": row["document_type_accuracy_clean"],
            "window_latency_ms_p50": latency["latency_ms_p50"],
            "document_latency_ms_p50": (row.get("document_latency") or {}).get("latency_ms_p50"),
        }))


if __name__ == "__main__":
    main()
