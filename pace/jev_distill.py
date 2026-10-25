"""Distil the 15.3 M reader into a 10.3 M one, then check what it cost.

The shipped reader is small, but the smallest checkpoint in this project is 10.3 M
and it is 33% cheaper to run. Distillation asks whether the smaller model can keep
the larger one's decisions rather than re-learning them from hard labels: the
student is trained on the teacher's full 42-way distribution (KL, temperature 2)
plus the true label, so it inherits the teacher's margins between confusable
fields instead of only its final answer.

    python -m pace.jev_distill --teacher artifacts/jev-field-v4 \
        --output artifacts/jev-distill-10m --epochs 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from .field_train import load_jsonl
from .jev_model import (DEFAULT_CONFIG, DEFAULT_QUESTION, JevChoiceModel, JevConfig,
                        load_jev, save_jev)
from .jev_train import ChoiceDataset, _choice_collate, _label_counts, evaluate_choice


def _teacher_logits(teacher: JevChoiceModel, dataset: ChoiceDataset, device: torch.device,
                    batch_size: int = 192) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: _choice_collate(batch, dataset.pad))
    option_ids = dataset.option_ids.to(device)
    option_mask = dataset.option_mask.to(device)
    chunks: list[torch.Tensor] = []
    teacher.eval()
    with torch.inference_mode():
        for input_ids, attention, state_mask, _ in loader:
            logits = teacher(input_ids.to(device), attention.to(device),
                             state_mask.to(device), option_ids, option_mask)
            chunks.append(logits.float().cpu())
    return torch.cat(chunks)


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fit one temperature on held-out logits, as the other checkpoints do."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    labels = targets.long()

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp(0.05, 20.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def distill(teacher_dir: Path, output: Path, tokenizer: Tokenizer,
            train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict],
            epochs: int = 3, batch_size: int = 192, lr: float = 1e-3,
            temperature: float = 2.0, alpha: float = 0.5, patience: int = 2,
            arch: dict[str, Any] | None = None, device: str | None = None,
            encoder: Path | None = None) -> dict[str, Any]:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    teacher, checkpoint, _ = load_jev(teacher_dir, device=target)
    teacher.eval()
    labels = list(checkpoint["labels"])
    descriptions = list(checkpoint["descriptions"])

    config = JevConfig.from_dict({**DEFAULT_CONFIG, **(arch or {}),
                                  "vocab_size": tokenizer.get_vocab_size()})
    model = JevChoiceModel(config, labels, descriptions).to(target)
    if encoder is not None and (Path(encoder) / "model.pt").exists():
        # The language-only checkpoint has the same architecture as the student, so
        # distillation starts from learned text rather than noise.
        source = torch.load(Path(encoder) / "model.pt", map_location="cpu", weights_only=False)
        own = model.state_dict()
        compatible = {key: value for key, value in source["state_dict"].items()
                      if key in own and own[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        print(json.dumps({"student_initialised_from": str(encoder),
                          "transferred": len(compatible)}), flush=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    train_dataset = ChoiceDataset(train_rows, tokenizer, labels)
    validation_dataset = ChoiceDataset(validation_rows, tokenizer, labels)
    test_dataset = ChoiceDataset(test_rows, tokenizer, labels)

    print(json.dumps({"students_parameters": model.parameter_count(),
                      "teacher_parameters": teacher.parameter_count(),
                      "labels": len(labels), "rows": len(train_rows)}), flush=True)
    started = time.perf_counter()
    soft = _teacher_logits(teacher, train_dataset, target)
    print(json.dumps({"teacher_forward_seconds": round(time.perf_counter() - started, 1),
                      "soft_targets": list(soft.shape)}), flush=True)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda batch: _choice_collate(batch, train_dataset.pad))
    option_ids = train_dataset.option_ids.to(target)
    option_mask = train_dataset.option_mask.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    counts = _label_counts(train_dataset, len(labels))
    weights = torch.tensor((counts.mean() / np.maximum(counts, 1.0)) ** 0.5,
                           dtype=torch.float32, device=target) if counts.any() else None

    best: dict[str, Any] = {"accuracy": -1.0, "epoch": 0}
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        epoch_start = time.perf_counter()
        for step, (input_ids, attention, state_mask, targets) in enumerate(loader):
            index = step * batch_size
            batch_soft = soft[index:index + len(targets)].to(target)
            logits = model(input_ids.to(target), attention.to(target),
                           state_mask.to(target), option_ids, option_mask)
            hard = F.cross_entropy(logits.float(), targets.to(target),
                                   weight=weights, label_smoothing=0.05)
            soft_loss = F.kl_div(F.log_softmax(logits.float() / temperature, dim=-1),
                                 F.log_softmax(batch_soft / temperature, dim=-1),
                                 reduction="batchmean", log_target=True) * (temperature ** 2)
            loss = alpha * hard + (1.0 - alpha) * soft_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(targets)
            seen += len(targets)
        metrics = evaluate_choice(model, validation_dataset, target)
        record = {"epoch": epoch, "loss": total / max(seen, 1),
                  "validation_accuracy": metrics["accuracy"],
                  "validation_macro_f1": metrics["macro_f1"],
                  "seconds": round(time.perf_counter() - epoch_start, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["accuracy"] > best["accuracy"]:
            best = {"accuracy": metrics["accuracy"], "epoch": epoch,
                    "state": {key: value.detach().cpu()
                              for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best["state"])
    calibration = evaluate_choice(model, validation_dataset, target)
    fitted = _fit_temperature(calibration["logits"], torch.as_tensor(calibration["targets"]))
    test_metrics = evaluate_choice(model, test_dataset, target, temperature=fitted)
    test_metrics.pop("logits", None)
    test_metrics.pop("targets", None)
    result = {
        "teacher": str(teacher_dir),
        "parameter_count": model.parameter_count(),
        "teacher_parameter_count": teacher.parameter_count(),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "history": history,
        "distillation_temperature": temperature,
        "temperature": fitted,
        "alpha": alpha,
        "question": DEFAULT_QUESTION,
        "arch": arch or {},
        "validation": {key: value for key, value in history[-1].items()
                       if key.startswith("validation")},
        "test": test_metrics,
        "device": str(target),
    }
    save_jev(model, tokenizer, output, temperature=fitted,
             extra={"stage": "distilled", "question": DEFAULT_QUESTION,
                    "teacher": str(teacher_dir),
                    "metrics": {key: value for key, value in result.items()
                                if key not in {"history"}}})
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Distil the field reader into a smaller model")
    parser.add_argument("--teacher", default="artifacts/jev-field-v4")
    parser.add_argument("--output", default="artifacts/jev-distill-10m")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--train", default="data/processed/field_mix_dropout/train.jsonl")
    parser.add_argument("--validation", default="data/processed/field_mix/validation.jsonl")
    parser.add_argument("--test", default="data/processed/field_mix/test.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--encoder", default=None,
                        help="Language-only checkpoint to initialise the student from")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    teacher = Path(args.teacher)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else teacher / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    arch = {key: value for key, value in (("hidden", args.hidden), ("layers", args.layers),
                                          ("heads", args.heads), ("ffn", args.ffn))
            if value is not None}
    result = distill(teacher, output, tokenizer, load_jsonl(args.train),
                     load_jsonl(args.validation), load_jsonl(args.test),
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     temperature=args.temperature, alpha=args.alpha,
                     patience=args.patience, arch=arch, device=args.device,
                     encoder=Path(args.encoder) if args.encoder else None)
    print(json.dumps({key: result[key] for key in
                      ("parameter_count", "teacher_parameter_count", "best_epoch", "test")},
                     indent=2))


if __name__ == "__main__":
    main()
