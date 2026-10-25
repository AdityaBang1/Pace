"""Refit a checkpoint's temperature on a validation split.

Every checkpoint carries one scalar that divides its logits before softmax, so
its percentages mean something. A weighted average of two fine-tunes is not one
of the two -- its confidence is systematically too sharp or too flat -- so the
scalar has to be fitted again after averaging. Fitting is done on validation
rows only; the test sets are scored afterwards, once.

    python scripts/fit_temperature.py --checkpoint artifacts/jv6-soup73 \
        --rows data/processed/field_mix_ext/validation.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pace.field_train import load_jsonl
from pace.jev_model import load_jev
from pace.jev_train import fit_temperature


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Refit a checkpoint's temperature")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rows", nargs="+",
                        default=["data/processed/field_mix_ext/validation.jsonl"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint, _ = load_jev(args.checkpoint, device)
    from pace.jev_model import JevEngine

    engine = JevEngine(args.checkpoint, device=device)
    index = {label: position for position, label in enumerate(engine.labels)}
    texts, targets = [], []
    for path in args.rows:
        for row in load_jsonl(path):
            if row.get("label") in index:
                texts.append(row["text"])
                targets.append(index[row["label"]])

    logits = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            input_ids, attention, state_mask = engine._encode(batch)
            logits.append(model(input_ids, attention, state_mask,
                                engine.option_ids, engine.option_mask).float().cpu())
    matrix = torch.cat(logits)
    temperature = fit_temperature(matrix, targets)
    checkpoint["temperature"] = float(temperature)
    torch.save(checkpoint, Path(args.checkpoint) / "model.pt")
    config_path = Path(args.checkpoint) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["temperature"] = float(temperature)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": args.checkpoint, "rows": len(targets),
                      "temperature": float(temperature),
                      "previous": float(engine.temperature)}, indent=2))


if __name__ == "__main__":
    main()
