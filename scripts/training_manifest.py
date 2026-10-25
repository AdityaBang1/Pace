"""One manifest that says exactly what each checkpoint is and what trained it.

The training logs record epochs and losses but not the identity of the data or the
bytes of the weights, which is the part that has to be reproducible. This walks
the checkpoint directories and writes a single JSON manifest: architecture,
parameter count, label count, weight hash, the training corpus files with row
counts and hashes, the recorded hyper-parameters, and the temperature.

    python scripts/training_manifest.py --output artifacts/MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            if remaining is not None:
                chunk = chunk[: remaining]
                remaining -= len(chunk)
            digest.update(chunk)
            if remaining is not None and remaining <= 0:
                break
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def describe(directory: Path) -> dict[str, Any] | None:
    model = directory / "model.pt"
    config = directory / "config.json"
    if not model.exists() or not config.exists():
        return None
    data = json.loads(config.read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    entry: dict[str, Any] = {
        "checkpoint": str(directory),
        "config": {key: value for key, value in data.items()
                   if key not in {"labels", "descriptions"}},
        "labels": len(data.get("labels") or []),
        "weights": {"path": str(model), "bytes": model.stat().st_size,
                    "sha256_head": _sha256(model, 1 << 20)},
    }
    tokenizer = directory / "tokenizer.json"
    if tokenizer.exists():
        entry["tokenizer"] = {"path": str(tokenizer), "sha256": _sha256(tokenizer)}
    for key in ("parameter_count", "train_rows", "validation_rows", "test_rows",
                "best_epoch", "epochs_run", "temperature", "train_seconds", "device",
                "option_dropout", "class_weight_power", "sampler_power", "arch"):
        if key in metrics:
            entry[key] = metrics[key]
    history = metrics.get("history") or []
    if history:
        entry["epochs"] = [{key: item.get(key) for key in
                            ("epoch", "loss", "validation_accuracy", "validation_macro_f1", "seconds")}
                           for item in history]
    entry["metrics_file"] = str(metrics_path) if metrics_path.exists() else None
    return entry


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write the checkpoint/training manifest")
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--data", nargs="+", default=[
        "data/processed/field_mix/train.jsonl",
        "data/processed/field_mix/validation.jsonl",
        "data/processed/field_mix/test.jsonl",
        "data/processed/field_mix_dropout/train.jsonl",
    ])
    parser.add_argument("--output", default="artifacts/MANIFEST.json")
    args = parser.parse_args(argv)

    root = Path(args.root)
    checkpoints = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        described = describe(directory)
        if described:
            checkpoints.append(described)

    corpora = []
    for entry in args.data:
        path = Path(entry)
        if path.exists():
            corpora.append({"path": str(path), "rows": _row_count(path),
                            "sha256": _sha256(path)})

    manifest = {
        "checkpoints": checkpoints,
        "training_corpora": corpora,
        "notes": ("Weights are identified by a hash of the first MiB, which is enough to "
                  "tell two fine-tunes apart without storing the full digest for a 61 MB "
                  "file per checkpoint."),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoints": len(checkpoints),
                      "corpora": len(corpora),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
