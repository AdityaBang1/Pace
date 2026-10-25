"""Export a field classifier to ONNX (fp32 + dynamic int8) and measure it on CPU.

The model is meant to run on a phone, so the questions that matter are not
"how many parameters" but "how big is the file, does int8 keep the labels, and
how long does one section take on a CPU". This script answers those three on a
real held-out file and writes the numbers next to the model.

    python scripts/export_field_onnx.py \
        --checkpoint artifacts/field-classifier-mix \
        --test data/processed/field_mix/test.jsonl \
        --output artifacts/field-classifier-mix/onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.field_model import load_field_checkpoint  # noqa: E402
from pace.field_train import load_jsonl  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)


def _sessions(fp32_path: Path, int8_path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return (
        ort.InferenceSession(str(fp32_path), sess_options=options,
                             providers=["CPUExecutionProvider"]),
        ort.InferenceSession(str(int8_path), sess_options=options,
                             providers=["CPUExecutionProvider"]),
    )


def _bench(runner, tokenizer, texts: list[str], max_length: int) -> dict[str, float]:
    times = []
    for text in texts:
        start = time.perf_counter()
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="np")
        runner(encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64))
        times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    return {
        "latency_ms_p50": float(ordered[len(ordered) // 2]),
        "latency_ms_p95": float(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))]),
        "mean_ms": float(np.mean(ordered)),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="ONNX/INT8 export for the field classifier")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-mix")
    parser.add_argument("--test", default="data/processed/field_mix/test.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=400, help="Rows used for parity and timing")
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint_dir / "onnx"
    output.mkdir(parents=True, exist_ok=True)

    model, checkpoint, tokenizer = load_field_checkpoint(checkpoint_dir, device="cpu")
    labels = list(checkpoint["labels"])
    index = {label: i for i, label in enumerate(labels)}
    max_length = int(checkpoint.get("max_length", 160))
    temperature = float(checkpoint.get("temperature", 1.0))

    fp32_path = output / "model_fp32.onnx"
    int8_path = output / "model_int8.onnx"
    if not fp32_path.exists():
        dummy_ids = torch.ones((1, max_length), dtype=torch.int64)
        dummy_mask = torch.ones((1, max_length), dtype=torch.int64)
        torch.onnx.export(
            Wrapper(model).eval(), (dummy_ids, dummy_mask), str(fp32_path),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        exported = onnx.load(str(fp32_path), load_external_data=True)
        onnx.checker.check_model(exported)
        onnx.save_model(exported, str(fp32_path), save_as_external_data=False)
        sidecar = Path(f"{fp32_path}.data")
        if sidecar.exists():
            sidecar.unlink()
    if not int8_path.exists():
        quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path),
                         weight_type=QuantType.QInt8)

    rows = [row for row in load_jsonl(args.test) if row["label"] in index][: args.limit]
    texts = [row["text"] for row in rows]
    gold = np.array([index[row["label"]] for row in rows])
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length,
                        return_tensors="np")
    ids, mask = encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64)

    with torch.inference_mode():
        reference = model(torch.from_numpy(ids), torch.from_numpy(mask)).float().numpy()
    fp32_sess, int8_sess = _sessions(fp32_path, int8_path)
    fp32 = fp32_sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
    int8 = int8_sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]

    def top1(logits):
        return (logits / temperature).argmax(axis=-1)

    report = {
        "checkpoint": str(checkpoint_dir),
        "base_model": checkpoint["base_model"],
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "labels": len(labels),
        "max_length": max_length,
        "rows": len(rows),
        "sizes_bytes": {
            "pytorch_checkpoint": (checkpoint_dir / "model.pt").stat().st_size,
            "onnx_fp32": fp32_path.stat().st_size,
            "onnx_int8": int8_path.stat().st_size,
        },
        "parity": {
            "fp32_max_abs_logit_diff": float(np.max(np.abs(reference - fp32))),
            "fp32_label_agreement": float(np.mean(top1(reference) == top1(fp32))),
            "int8_label_agreement": float(np.mean(top1(reference) == top1(int8))),
        },
        "accuracy_on_file": {
            "pytorch": float(np.mean(top1(reference) == gold)),
            "onnx_fp32": float(np.mean(top1(fp32) == gold)),
            "onnx_int8": float(np.mean(top1(int8) == gold)),
        },
        "cpu_latency": {
            "threads": 4,
            "rows": len(texts),
            "pytorch": _bench(lambda i, m: model(torch.from_numpy(i),
                                                 torch.from_numpy(m)), tokenizer, texts, max_length),
            "onnx_fp32": _bench(lambda i, m: fp32_sess.run(["logits"],
                                                           {"input_ids": i, "attention_mask": m}),
                                tokenizer, texts, max_length),
            "onnx_int8": _bench(lambda i, m: int8_sess.run(["logits"],
                                                           {"input_ids": i, "attention_mask": m}),
                                tokenizer, texts, max_length),
        },
    }
    (output / "onnx_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("parameter_count", "sizes_bytes", "parity", "cpu_latency")}, indent=2))


if __name__ == "__main__":
    main()
