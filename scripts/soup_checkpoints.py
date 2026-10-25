"""Average the weights of checkpoints that share an initialisation.

Fine-tuning the same encoder on the same corpus twice lands in two nearby basins;
averaging them ("model soup") usually keeps the in-distribution score of the best
member and recovers some of the generalisation either one loses alone. It costs
no training and no inference time, which is why it is worth measuring before
accepting a single fine-tune as final.

Only checkpoints with identical architecture and label menu can be averaged, so
the script refuses anything else rather than silently dropping tensors.

    python scripts/soup_checkpoints.py --inputs artifacts/jev-field-v2 artifacts/jv3-c2-domw \
        --output artifacts/jv4-soup
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(directory: Path) -> dict:
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    return checkpoint


def soup(directories: list[Path], weights: list[float] | None = None) -> dict:
    checkpoints = [_load(directory) for directory in directories]
    reference = checkpoints[0]
    labels = reference["labels"]
    descriptions = reference["descriptions"]
    for directory, checkpoint in zip(directories[1:], checkpoints[1:]):
        if checkpoint["labels"] != labels:
            raise SystemExit(f"{directory} has a different label menu; cannot average")
        if checkpoint["config"] != reference["config"]:
            raise SystemExit(f"{directory} has a different architecture; cannot average")
    share = weights or [1.0 / len(checkpoints)] * len(checkpoints)
    if len(share) != len(checkpoints) or abs(sum(share) - 1.0) > 1e-6:
        raise SystemExit("weights must match the inputs and sum to 1")

    averaged = {}
    for key, value in reference["state_dict"].items():
        if not torch.is_floating_point(value):
            averaged[key] = value.clone()
            continue
        stacked = torch.zeros_like(value, dtype=torch.float32)
        for checkpoint, weight in zip(checkpoints, share):
            stacked += checkpoint["state_dict"][key].float() * weight
        averaged[key] = stacked.to(value.dtype)
    merged = dict(reference)
    merged["state_dict"] = averaged
    merged["soup_of"] = [str(directory) for directory in directories]
    merged["soup_weights"] = share
    return merged


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Average identical-architecture checkpoints")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    args = parser.parse_args(argv)

    directories = [Path(item) for item in args.inputs]
    merged = soup(directories, args.weights)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output / "model.pt")
    source = directories[0]
    for name in ("tokenizer.json", "config.json"):
        if (source / name).exists() and not (output / name).exists():
            (output / name).write_bytes((source / name).read_bytes())
    print(json.dumps({"output": str(output), "soup_of": merged["soup_of"],
                      "weights": merged["soup_weights"],
                      "parameter_count": merged.get("parameter_count")}, indent=2))


if __name__ == "__main__":
    main()
