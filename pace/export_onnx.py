"""Export fine-tuned MiniLM checkpoints to ONNX (FP32 + dynamic INT8) and verify parity, accuracy, and latency."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from .model import MAX_LENGTH, PaceClassifier


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: PaceClassifier):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)


def export_and_benchmark_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    test_jsonl_path: str | Path | None = None,
    latency_runs: int = 200,
) -> dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels: list[str] = list(meta["labels"])
    max_length: int = int(meta.get("max_length", MAX_LENGTH))
    temperature: float = float(meta.get("temperature", 1.0))

    model = PaceClassifier(
        meta["base_model"],
        config_dir=str(ckpt_path.parent / meta["encoder_config"]),
        num_classes=len(labels),
    )
    model.load_state_dict(meta["state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt_path.parent / meta["tokenizer_dir"]),
        local_files_only=True,
    )

    wrapper = OnnxExportWrapper(model).eval()
    dummy_ids = torch.ones((1, max_length), dtype=torch.int64)
    dummy_mask = torch.ones((1, max_length), dtype=torch.int64)

    fp32_path = out_dir / "model_fp32.onnx"
    int8_path = out_dir / "model_int8.onnx"

    if not fp32_path.exists():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        # Consolidate any external data into single self-contained ONNX file
        onnx_model = onnx.load(str(fp32_path), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)
        data_sidecar = Path(f"{fp32_path}.data")
        if data_sidecar.exists():
            data_sidecar.unlink()

    if not int8_path.exists():
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 8
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    test_file = Path(test_jsonl_path) if test_jsonl_path else (ckpt_path.parent / "test.jsonl")
    test_rows = [json.loads(line) for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Evaluate held-out test set: PyTorch on GPU (if available, else CPU), ONNX FP32/INT8 on CPU
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_eval = model.to(eval_device)
    batch_size = 256
    pt_logits_all = []
    fp32_logits_all = []
    int8_logits_all = []
    y_true = [labels.index(r["label"]) for r in test_rows]

    with torch.inference_mode():
        for start in range(0, len(test_rows), batch_size):
            batch_texts = [r["text"] for r in test_rows[start : start + batch_size]]
            enc = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            ids_np = enc["input_ids"].astype(np.int64)
            mask_np = enc["attention_mask"].astype(np.int64)

            pt_out = (
                model_eval(
                    torch.from_numpy(ids_np).to(eval_device),
                    torch.from_numpy(mask_np).to(eval_device),
                )
                .cpu()
                .numpy()
            )
            fp32_out = fp32_sess.run(["logits"], {"input_ids": ids_np, "attention_mask": mask_np})[0]
            int8_out = int8_sess.run(["logits"], {"input_ids": ids_np, "attention_mask": mask_np})[0]

            pt_logits_all.append(pt_out)
            fp32_logits_all.append(fp32_out)
            int8_logits_all.append(int8_out)

    model.to("cpu")

    pt_logits = np.concatenate(pt_logits_all, axis=0)
    fp32_logits = np.concatenate(fp32_logits_all, axis=0)
    int8_logits = np.concatenate(int8_logits_all, axis=0)

    pt_preds = pt_logits.argmax(axis=-1)
    fp32_preds = fp32_logits.argmax(axis=-1)
    int8_preds = int8_logits.argmax(axis=-1)

    max_abs_diff_fp32 = float(np.max(np.abs(pt_logits - fp32_logits)))
    mean_abs_diff_fp32 = float(np.mean(np.abs(pt_logits - fp32_logits)))
    fp32_label_agreement = float(np.mean(pt_preds == fp32_preds))
    int8_label_agreement = float(np.mean(pt_preds == int8_preds))

    # Single-example CPU latency benchmark (including tokenization) on natural test rows
    sample_texts = [r["text"] for r in test_rows[: min(latency_runs, len(test_rows))]]
    while len(sample_texts) < latency_runs:
        sample_texts.extend(sample_texts[: latency_runs - len(sample_texts)])

    for warmup_text in sample_texts[:10]:
        enc = tokenizer(warmup_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        ids_np = enc["input_ids"].astype(np.int64)
        mask_np = enc["attention_mask"].astype(np.int64)
        with torch.inference_mode():
            model(torch.from_numpy(ids_np), torch.from_numpy(mask_np))
        fp32_sess.run(["logits"], {"input_ids": ids_np, "attention_mask": mask_np})
        int8_sess.run(["logits"], {"input_ids": ids_np, "attention_mask": mask_np})

    def _bench_cpu(runner) -> dict[str, float]:
        times_ms = []
        for txt in sample_texts:
            t0 = time.perf_counter()
            enc = tokenizer(txt, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
            ids_np = enc["input_ids"].astype(np.int64)
            mask_np = enc["attention_mask"].astype(np.int64)
            runner(ids_np, mask_np)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        ordered = sorted(times_ms)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))]
        return {
            "latency_ms_p50": float(p50),
            "latency_ms_p95": float(p95),
            "latency_ms_mean": float(np.mean(ordered)),
            "throughput_rows_per_sec": float(1000.0 / np.mean(ordered)),
        }

    with torch.inference_mode():
        pt_cpu_bench = _bench_cpu(lambda i, m: model(torch.from_numpy(i), torch.from_numpy(m)))
    fp32_cpu_bench = _bench_cpu(lambda i, m: fp32_sess.run(["logits"], {"input_ids": i, "attention_mask": m}))
    int8_cpu_bench = _bench_cpu(lambda i, m: int8_sess.run(["logits"], {"input_ids": i, "attention_mask": m}))

    metadata = {
        "source_checkpoint": str(ckpt_path),
        "base_model": meta["base_model"],
        "labels": labels,
        "max_length": max_length,
        "temperature": temperature,
        "ambiguous_values": meta.get("ambiguous_values", {}),
        "valid_subjects": meta.get("valid_subjects", []),
        "artifacts": {
            "pytorch_checkpoint_bytes": ckpt_path.stat().st_size,
            "onnx_fp32_path": str(fp32_path),
            "onnx_fp32_bytes": fp32_path.stat().st_size,
            "onnx_int8_path": str(int8_path),
            "onnx_int8_bytes": int8_path.stat().st_size,
            "int8_size_reduction_ratio": round(ckpt_path.stat().st_size / int8_path.stat().st_size, 2),
        },
        "numerical_parity_on_test": {
            "test_rows": len(test_rows),
            "max_abs_logit_diff_pt_vs_onnx_fp32": max_abs_diff_fp32,
            "mean_abs_logit_diff_pt_vs_onnx_fp32": mean_abs_diff_fp32,
            "pt_vs_onnx_fp32_label_agreement": fp32_label_agreement,
            "pt_vs_onnx_int8_label_agreement": int8_label_agreement,
        },
        "held_out_test_metrics": {
            "pytorch_fp32": {
                "accuracy": float(accuracy_score(y_true, pt_preds)),
                "macro_f1": float(f1_score(y_true, pt_preds, average="macro", zero_division=0)),
                "errors": int(np.sum(pt_preds != np.array(y_true))),
            },
            "onnx_fp32": {
                "accuracy": float(accuracy_score(y_true, fp32_preds)),
                "macro_f1": float(f1_score(y_true, fp32_preds, average="macro", zero_division=0)),
                "errors": int(np.sum(fp32_preds != np.array(y_true))),
            },
            "onnx_int8_dynamic": {
                "accuracy": float(accuracy_score(y_true, int8_preds)),
                "macro_f1": float(f1_score(y_true, int8_preds, average="macro", zero_division=0)),
                "errors": int(np.sum(int8_preds != np.array(y_true))),
            },
        },
        "cpu_single_row_benchmark": {
            "runs": latency_runs,
            "intra_op_threads": 8,
            "pytorch_cpu_fp32": pt_cpu_bench,
            "onnxruntime_cpu_fp32": fp32_cpu_bench,
            "onnxruntime_cpu_int8": int8_cpu_bench,
            "int8_speedup_over_pytorch_cpu_p50": round(
                pt_cpu_bench["latency_ms_p50"] / max(int8_cpu_bench["latency_ms_p50"], 1e-6), 2
            ),
        },
    }
    (out_dir / "onnx_report.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export MiniLM checkpoints to ONNX FP32 and INT8 and benchmark.")
    p.add_argument("--checkpoint", default="artifacts/usc-miniLM/model.pt")
    p.add_argument("--output", default="artifacts/usc-miniLM/onnx")
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args(argv)
    report = export_and_benchmark_checkpoint(args.checkpoint, args.output, latency_runs=args.runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
