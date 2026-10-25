"""Export the Jev choice model to ONNX, quantize it, and measure what changed.

The plan asks for a deployable artifact and for quantization to be measured rather
than assumed. This does both: it writes FP32 ONNX, applies ONNX Runtime's dynamic
INT8 quantization, then reports file size, CPU latency and — the part that matters
— how often the quantized graph still agrees with PyTorch on real rows, and what
the metric delta is on the same test file.

The option menu is baked into the exported graph as constants, because a caller
does not change it at inference: the descriptions are the model's interface.

    python scripts/export_jev_onnx.py --checkpoint artifacts/jev-field-v2 \\
        --rows data/processed/transfer/unseen-syllabus/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


class _ExportModel(torch.nn.Module):
    """The same maths, without nn.TransformerEncoder's nested-tensor path.

    Tracing the stock encoder baked the example's batch and sequence length into
    its attention reshapes, and scripting it failed on ``prim::is_nested``. This
    re-implements that module's pre-LN layer body over the very same submodules —
    ``self_attn``, ``linear1/2``, ``norm1/2``, ``dropout1/2`` — so nothing is
    re-trained and the outputs are identical, but every reshape is symbolic.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, state_mask, option_ids, option_mask):
        model = self.model
        encoder = model.encoder
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = encoder.project(encoder.embed_norm(
            encoder.embed(input_ids) + encoder.position(positions)))
        padding = attention_mask == 0
        for layer in encoder.blocks.layers:
            normed = layer.norm1(hidden)
            attended, _ = layer.self_attn(normed, normed, normed,
                                          key_padding_mask=padding, need_weights=False)
            hidden = hidden + layer.dropout1(attended)
            normed = layer.norm2(hidden)
            feed = layer.linear2(layer.dropout(torch.nn.functional.gelu(layer.linear1(normed))))
            hidden = hidden + layer.dropout2(feed)
        hidden = encoder.norm(hidden)
        weights = state_mask.unsqueeze(-1).to(hidden.dtype)
        state = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        state = torch.nn.functional.normalize(model.state_head(state), dim=-1)
        option_weights = option_mask.unsqueeze(-1).to(torch.float32)
        embedded = encoder.embed(option_ids)
        pooled = (embedded * option_weights).sum(dim=1) / option_weights.sum(dim=1).clamp_min(1.0)
        options = torch.nn.functional.normalize(model.option_head(pooled), dim=-1)
        scale = model.log_scale.exp().clamp(1.0, 50.0)
        return scale * (state @ options.transpose(0, 1)) + model.option_bias


def _export(engine, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.onnx"
    example_ids, example_attention, example_state = engine._encode(["Room 303"])
    wrapper = _ExportModel(engine.model).eval()
    with torch.no_grad():
        reference = engine.model(example_ids, example_attention, example_state,
                                 engine.option_ids, engine.option_mask)
        produced = wrapper(example_ids, example_attention, example_state,
                           engine.option_ids, engine.option_mask)
    if float((reference - produced).abs().max()) > 1e-4:
        raise SystemExit("the export wrapper does not reproduce the model's logits")
    # Tracing alone freezes the example's sequence length inside the attention
    # reshapes, so the dynamic dimensions have to be declared symbolically and the
    # dynamo exporter used; dynamic_axes is ignored by that path.
    from torch.export import Dim

    batch, sequence = Dim("batch", min=1, max=512), Dim("sequence", min=1, max=512)
    shapes = {"input_ids": {0: batch, 1: sequence},
              "attention_mask": {0: batch, 1: sequence},
              "state_mask": {0: batch, 1: sequence},
              # The option menu is a constant of the checkpoint, not an axis.
              "option_ids": {}, "option_mask": {}}
    torch.onnx.export(
        wrapper,
        (example_ids, example_attention, example_state, engine.option_ids, engine.option_mask),
        str(path),
        input_names=["input_ids", "attention_mask", "state_mask", "option_ids", "option_mask"],
        output_names=["logits"],
        dynamic_shapes=shapes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=True,
    )
    return path


def _quantize(source: Path, target: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
    return target


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def _latency(session, engine, texts: list[str], runs: int = 20) -> dict:
    encoded = [(engine._encode([text])) for text in texts]
    # Warm up so the first allocation is not counted.
    for ids, attention, state in encoded[:3]:
        session.run(None, {"input_ids": ids.numpy(), "attention_mask": attention.numpy(),
                           "state_mask": state.numpy(),
                           "option_ids": engine.option_ids.cpu().numpy(),
                           "option_mask": engine.option_mask.cpu().numpy()})
    durations = []
    for index in range(runs):
        ids, attention, state = encoded[index % len(encoded)]
        start = time.perf_counter()
        session.run(None, {"input_ids": ids.numpy(), "attention_mask": attention.numpy(),
                           "state_mask": state.numpy(),
                           "option_ids": engine.option_ids.cpu().numpy(),
                           "option_mask": engine.option_mask.cpu().numpy()})
        durations.append((time.perf_counter() - start) * 1000)
    array = np.asarray(durations)
    return {"runs": runs, "median_ms": float(np.median(array)),
            "p90_ms": float(np.percentile(array, 90))}


def _predict(session, engine, texts: list[str], batch_size: int = 64) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        ids, attention, state = engine._encode(batch)
        logits = session.run(None, {"input_ids": ids.numpy(),
                                    "attention_mask": attention.numpy(),
                                    "state_mask": state.numpy(),
                                    "option_ids": engine.option_ids.cpu().numpy(),
                                    "option_mask": engine.option_mask.cpu().numpy()})[0]
        probs = torch.softmax(torch.from_numpy(logits) / max(engine.temperature, 1e-6), dim=-1)
        rows.append(probs.numpy())
    return np.concatenate(rows, axis=0)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="ONNX + INT8 export for the field reader")
    parser.add_argument("--checkpoint", default="artifacts/jev-field-v2")
    parser.add_argument("--rows", default="data/processed/transfer/unseen-syllabus/test.jsonl")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    from sklearn.metrics import accuracy_score, f1_score

    from pace.field_train import load_jsonl
    from pace.jev_model import JevEngine

    checkpoint = Path(args.checkpoint)
    output = Path(args.output or checkpoint / "onnx")
    engine = JevEngine(checkpoint, device="cpu")

    fp32_path = _export(engine, output)
    int8_path = _quantize(fp32_path, output / "model.int8.onnx")

    rows = load_jsonl(args.rows)
    index = {label: position for position, label in enumerate(engine.labels)}
    kept = [row for row in rows if row["label"] in index]
    texts = [row["text"] for row in kept]
    truth = np.asarray([index[row["label"]] for row in kept])

    torch_probs = np.asarray([[item[label] for label in engine.labels]
                              for item in engine.probabilities(texts)])
    report = {"checkpoint": str(checkpoint), "rows": len(kept),
              "files": {}, "latency": {}, "quality": {}}
    for name, session_path in (("fp32", fp32_path), ("int8", int8_path)):
        session = _session(session_path)
        probs = _predict(session, engine, texts)
        prediction = probs.argmax(axis=1)
        torch_prediction = torch_probs.argmax(axis=1)
        report["files"][name] = {"path": str(session_path),
                                 "bytes": session_path.stat().st_size}
        report["latency"][name] = _latency(session, engine, texts, runs=args.runs)
        report["quality"][name] = {
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "agreement_with_torch": float((prediction == torch_prediction).mean()),
            "mean_absolute_probability_shift": float(
                np.abs(probs - torch_probs).mean()),
        }

    report["torch"] = {
        "accuracy": float(accuracy_score(truth, torch_probs.argmax(axis=1))),
        "macro_f1": float(f1_score(truth, torch_probs.argmax(axis=1), average="macro",
                                   zero_division=0)),
        "bytes": (checkpoint / "model.pt").stat().st_size,
    }
    (output / "onnx_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
