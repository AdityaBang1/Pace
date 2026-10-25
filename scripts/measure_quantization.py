"""fp32 vs dynamic-int8 for the selected Jev field model.

Sizes, latency per row, argmax agreement and probability drift, measured on the
same real rows the checkpoint was scored on. This is the phone-deployment
question: does the model still make the same decisions when its Linear layers
are stored as 8-bit integers.

    python scripts/measure_quantization.py --checkpoint artifacts/jev-field-v2 \
        --rows data/processed/field_mix/test.jsonl --output artifacts/jev-field-v2/quantization.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pace.field_train import load_jsonl
from pace.jev_model import JevEngine

# torch's sparsity fast-path probe reads ``layer.linear1.weight`` -- which is a
# function, not a tensor, once the Linear is a dynamic quantized module -- so the
# encoder raises AttributeError under quantization unless it is turned off.
torch.backends.mha.set_fastpath_enabled(False)


def _sizes(model: torch.nn.Module, tmp: Path) -> dict[str, float]:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), tmp)
    return {"state_dict_mb": round(tmp.stat().st_size / 1e6, 3)}


def _timed(engine: JevEngine, texts: list[str], repeats: int = 3) -> dict[str, float]:
    engine.probabilities(texts[:8])  # warm caches
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        engine.probabilities(texts)
        best = min(best, time.perf_counter() - start)
    return {"seconds": round(best, 4), "ms_per_row": round(best * 1000 / len(texts), 3)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Quantization parity, Jev field model")
    parser.add_argument("--checkpoint", default="artifacts/jev-field-v2")
    parser.add_argument("--rows", default="data/processed/field_mix/test.jsonl")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.rows)[: args.limit]
    texts = [row["text"] for row in rows]
    targets = [row.get("label") for row in rows]

    fp32 = JevEngine(args.checkpoint, device="cpu")
    fp32_latency = _timed(fp32, texts, repeats=3)
    fp32_probs = fp32.probabilities(texts)
    labels = fp32.labels
    fp32_argmax = [max(row, key=row.get) for row in fp32_probs]
    fp32_matrix = np.asarray([[row[label] for label in labels] for row in fp32_probs])

    workdir = Path("artifacts/quantization")
    fp32_info = _sizes(fp32.model, workdir / "jev-fp32.pt")

    import copy

    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(fp32.model).cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)

    class _QuantizedEngine(JevEngine):
        """A loaded engine with its weights swapped for the int8 module."""

        def __init__(self, source: JevEngine, model: torch.nn.Module):
            self.model = model
            self.checkpoint = source.checkpoint
            self.tokenizer = source.tokenizer
            self.device = torch.device("cpu")
            self.labels = list(source.labels)
            self.label_tasks = dict(source.label_tasks)
            self.question = source.question
            self.temperature = source.temperature
            self.max_state = source.max_state
            self.option_ids = source.option_ids
            self.option_mask = source.option_mask

    shim = _QuantizedEngine(fp32, quantized)

    int8_probs = shim.probabilities(texts)
    int8_latency = _timed(shim, texts, repeats=3)
    int8_argmax = [max(row, key=row.get) for row in int8_probs]
    int8_matrix = np.asarray([[row[label] for label in labels] for row in int8_probs])
    int8_info = _sizes(quantized, workdir / "jev-int8-dynamic.pt")

    agreement = float(np.mean([a == b for a, b in zip(fp32_argmax, int8_argmax)]))
    scored = [i for i, target in enumerate(targets) if target in labels]
    accuracy = {
        "fp32": float(np.mean([fp32_argmax[i] == targets[i] for i in scored])),
        "int8": float(np.mean([int8_argmax[i] == targets[i] for i in scored])),
        "scored_rows": len(scored),
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "rows": len(texts),
        "labels": len(labels),
        "fp32": {**fp32_info, **fp32_latency},
        "int8_dynamic": {**int8_info, **int8_latency},
        "size_ratio": round(fp32_info["state_dict_mb"] / int8_info["state_dict_mb"], 3),
        "speedup": round(fp32_latency["ms_per_row"] / int8_latency["ms_per_row"], 3),
        "argmax_agreement": agreement,
        "probability_mae": float(np.abs(fp32_matrix - int8_matrix).mean()),
        "probability_max_abs_delta": float(np.abs(fp32_matrix - int8_matrix).max()),
        "accuracy_fp32": accuracy["fp32"],
        "accuracy_int8": accuracy["int8"],
        "rows_that_changed": [[texts[i][:120], fp32_argmax[i], int8_argmax[i]]
                              for i in range(len(texts)) if fp32_argmax[i] != int8_argmax[i]][:20],
        "notes": "eager torch dynamic quantization; MHA fastpath disabled (torch bug with quantized Linear)",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
