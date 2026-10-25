"""Train the real-text academic field classifier.

The recipe is deliberately closer to a competitive fine-tune than to a quick
baseline, because the whole point of this model is to be trustworthy on real
documents:

* layer-wise learning-rate decay over the encoder (lower layers move less),
* linear warmup followed by cosine decay,
* mixed-precision bf16 on GPU,
* length-grouped dynamic-padding batches (fast without truncating real text),
* inverse-sqrt class weights so rare field types are not drowned out,
* early stopping on validation macro-F1 with best-state restore,
* a scalar temperature fitted on the validation split only, so the reported
  percentages are calibrated while the test split stays untouched,
* optional multi-seed runs so the reported numbers carry a variance estimate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

from .field_model import DEFAULT_BASE_MODEL, FIELD_LABELS, FieldClassifierModel, FIELD_TASKS

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class FieldRows(Dataset):
    """Tokenised rows kept unpadded so batches can pad to their own longest member."""

    def __init__(self, rows: Sequence[dict[str, Any]], labels: Sequence[str], tokenizer,
                 max_length: int):
        self.labels = list(labels)
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        encoded = tokenizer([r["text"] for r in rows], truncation=True,
                            max_length=max_length, add_special_tokens=True)
        self.input_ids = list(encoded["input_ids"])
        self.attention_mask = list(encoded["attention_mask"])
        self.y = torch.tensor([self.label_to_id[r["label"]] for r in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def lengths(self) -> list[int]:
        return [len(ids) for ids in self.input_ids]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[index], dtype=torch.long),
            "labels": self.y[index],
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch to its own longest sequence (never to the global maximum)."""
    width = max(int(item["input_ids"].size(0)) for item in batch)
    input_ids, attention = [], []
    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad = width - ids.size(0)
        if pad:
            ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad, dtype=mask.dtype)])
        input_ids.append(ids)
        attention.append(mask)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


class LengthGroupedBatches(torch.utils.data.Sampler):
    """Shuffle, then sort inside mega-batches by length so padding is cheap."""

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, mega: int = 50):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.mega = mega
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        chunk = self.batch_size * self.mega
        batches: list[list[int]] = []
        for start in range(0, len(indices), chunk):
            block = sorted(indices[start:start + chunk], key=lambda i: self.lengths[i])
            for offset in range(0, len(block), self.batch_size):
                batches.append(block[offset:offset + self.batch_size])
        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


_LAYER_INDEX = re.compile(r"(?:^|\.)layers?\.(\d+)\.")


def _layer_id(name: str, num_layers: int) -> int:
    """Index a parameter's transformer block, across encoder families.

    BERT/MiniLM/MPNet name blocks ``encoder.layer.N``, ModernBERT names them
    ``layers.N``; both must decay, or a deeper encoder silently receives one
    flat learning rate for every block.
    """
    if "embeddings" in name:
        return 0
    match = _LAYER_INDEX.search(name)
    if match:
        try:
            return min(num_layers, int(match.group(1))) + 1
        except ValueError:
            return num_layers + 1
    if name.startswith("classifier"):
        return num_layers + 2
    return num_layers + 1


def build_param_groups(model: FieldClassifierModel, encoder_lr: float, head_lr: float,
                       decay: float, weight_decay: float) -> list[dict[str, Any]]:
    """Layer-wise LR decay, with no weight decay on biases and normalisations."""
    num_layers = int(model.encoder.config.num_hidden_layers)
    max_id = num_layers + 2
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer = _layer_id(name, num_layers)
        no_decay = name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name
        key = (layer, no_decay)
        if key not in groups:
            base = head_lr if name.startswith("classifier") else encoder_lr
            groups[key] = {
                "params": [],
                "lr": base * (decay ** (max_id - layer)),
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"layer{layer}{'_nodecay' if no_decay else ''}",
            }
        groups[key]["params"].append(param)
    ordered = [groups[key] for key in sorted(groups)]
    print(json.dumps({"param_groups": [{"name": g["name"], "lr": g["lr"],
                                        "tensors": len(g["params"])} for g in ordered]}), flush=True)
    return ordered


def cosine_schedule(step: int, total: int, warmup: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def class_weights(labels: Sequence[str], counts: Counter, power: float = 0.5,
                  cap: float = 8.0) -> torch.Tensor:
    """Inverse-sqrt frequency weights, capped so one rare class cannot dominate."""
    total = sum(counts.get(label, 0) for label in labels)
    present = sum(1 for label in labels if counts.get(label, 0) > 0)
    weights = []
    for label in labels:
        count = counts.get(label, 0)
        if count == 0:
            weights.append(0.0)
            continue
        raw = (total / max(1.0, present * count)) ** power
        weights.append(min(cap, raw))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_logits(model: FieldClassifierModel, loader: DataLoader, device: torch.device,
                   amp_dtype: torch.dtype | None) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[int] = []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(ids, mask)
        else:
            out = model(ids, mask)
        logits.append(out.float().cpu())
        targets.extend(batch["labels"].tolist())
    return torch.cat(logits), np.array(targets)


def fit_temperature(logits: torch.Tensor, targets: np.ndarray) -> float:
    """Fit a single temperature on held-out data (LBFGS on log T)."""
    logits = logits.detach().clone().float()
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=60, line_search_fn="strong_wolfe")
    target = torch.as_tensor(targets, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp().clamp(0.05, 20.0), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                                bins: int = 10) -> float:
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence >= low) & ((confidence < high) if index < bins - 1 else confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return ece


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        mask = confidence >= threshold
        rows.append({
            "threshold": threshold,
            "coverage": float(mask.mean()),
            "precision": float(correct[mask].mean()) if mask.any() else None,
        })
    return rows


def evaluate_logits(logits: torch.Tensor, targets: np.ndarray, labels: Sequence[str],
                    temperature: float = 1.0) -> dict[str, Any]:
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    confidence, prediction = probs.max(dim=-1)
    confidence = confidence.numpy()
    prediction = prediction.numpy()
    correct = (prediction == targets)
    report = classification_report(targets, prediction, labels=list(range(len(labels))),
                                   target_names=list(labels), output_dict=True, zero_division=0)
    matrix = confusion_matrix(targets, prediction)
    pairs: Counter = Counter()
    for true_index, pred_index in zip(targets, prediction):
        if true_index != pred_index:
            pairs[f"{labels[int(true_index)]} -> {labels[int(pred_index)]}"] += 1
    return {
        "accuracy": float(accuracy_score(targets, prediction)),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, prediction, average="weighted", zero_division=0)),
        "nll": float(torch.nn.functional.cross_entropy(logits / max(temperature, 1e-6),
                                                       torch.as_tensor(targets, dtype=torch.long))),
        "ece_10bin": expected_calibration_error(confidence, correct),
        "high_confidence_threshold": 0.9,
        "high_confidence_coverage": float((confidence >= 0.9).mean()),
        "high_confidence_precision": float(correct[confidence >= 0.9].mean())
        if (confidence >= 0.9).any() else None,
        "risk_coverage": risk_coverage(confidence, correct),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": pairs.most_common(25),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels},
        "per_class_support": {label: int(report[label]["support"]) for label in labels},
        "_confidence": confidence,
        "_prediction": prediction,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def oversample_rare(rows: list[dict[str, Any]], minimum: int, seed: int) -> list[dict[str, Any]]:
    """Repeat rare field types so the head still sees them often enough to learn.

    Real field types are long-tailed: a handful of them (attendance policy,
    office hours, materials) appear in only a few real documents. Oversampling
    plus inverse-sqrt class weights keeps them from being silently dropped.
    """
    if minimum <= 0:
        return list(rows)
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    output: list[dict[str, Any]] = []
    for label, items in by_label.items():
        output.extend(items)
        if len(items) < minimum:
            repeats = minimum // max(1, len(items))
            output.extend(items * repeats)
            remainder = minimum - len(items) * repeats
            if remainder:
                output.extend(rng.sample(items, remainder))
    rng.shuffle(output)
    return output


def train_one(seed: int, train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
              test_rows: list[dict[str, Any]], output: Path, base_model: str = DEFAULT_BASE_MODEL,
              epochs: int = 6, batch_size: int = 64, encoder_lr: float = 3e-5, head_lr: float = 1e-3,
              layer_decay: float = 0.95, weight_decay: float = 0.01, warmup_ratio: float = 0.1,
              max_length: int = 192, patience: int = 2, class_weight_power: float = 0.5,
              oversample_min: int = 400, device: str | None = None,
              eval_extra: dict[str, list[dict[str, Any]]] | None = None,
              workers: int = 0, verbose: bool = True,
              save_each_epoch: bool = False) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if target_device.type == "cuda" else None

    counts = Counter(row["label"] for row in train_rows)
    seen = set(counts)
    for extra in (validation_rows, test_rows):
        seen.update(row["label"] for row in extra)
    labels = [label for label in FIELD_LABELS if label in seen]
    train_rows = oversample_rare(train_rows, oversample_min, seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    datasets = {
        "train": FieldRows(train_rows, labels, tokenizer, max_length),
        "validation": FieldRows(validation_rows, labels, tokenizer, max_length),
        "test": FieldRows(test_rows, labels, tokenizer, max_length),
    }
    train_lengths = datasets["train"].lengths()
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(datasets["train"], batch_sampler=LengthGroupedBatches(
            train_lengths, batch_size, seed), collate_fn=lambda b: collate(b, pad_id),
            num_workers=workers, pin_memory=target_device.type == "cuda",
            persistent_workers=workers > 0),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size * 2, shuffle=False,
                                 collate_fn=lambda b: collate(b, pad_id), num_workers=workers,
                                 persistent_workers=workers > 0),
        "test": DataLoader(datasets["test"], batch_size=batch_size * 2, shuffle=False,
                           collate_fn=lambda b: collate(b, pad_id), num_workers=workers,
                           persistent_workers=workers > 0),
    }

    model = FieldClassifierModel(base_model, num_classes=len(labels)).to(target_device)
    weights = class_weights(labels, counts, power=class_weight_power).to(target_device)
    groups = build_param_groups(model, encoder_lr, head_lr, layer_decay, weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=encoder_lr, weight_decay=weight_decay)
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, total_steps, warmup_steps))
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    history: list[dict[str, Any]] = []
    best_macro = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start = time.perf_counter()

    def checkpoint_payload(state: dict[str, torch.Tensor], temperature: float,
                           epoch: int, history_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "state_dict": state,
            "base_model": base_model,
            "num_classes": len(labels),
            "labels": labels,
            "encoder_config": "encoder_config",
            "tokenizer_dir": "tokenizer",
            "max_length": max_length,
            "temperature": temperature,
            "seed": seed,
            "selected_epoch": epoch,
            "class_weight_power": class_weight_power,
            "layer_decay": layer_decay,
            "encoder_lr": encoder_lr,
            "head_lr": head_lr,
            "batch_size": batch_size,
            "epochs_run": len(history_rows),
        }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses: list[float] = []
        for batch in loaders["train"]:
            ids = batch["input_ids"].to(target_device, non_blocking=True)
            mask = batch["attention_mask"].to(target_device, non_blocking=True)
            targets = batch["labels"].to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(ids, mask)
                    loss = criterion(logits.float(), targets)
            else:
                logits = model(ids, mask)
                loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        val_logits, val_targets = predict_logits(model, loaders["validation"], target_device, amp_dtype)
        val_metrics = evaluate_logits(val_logits, val_targets, labels)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        if verbose:
            print(json.dumps(record), flush=True)
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_each_epoch:
                # A multi-hour run must leave a usable model behind if it is
                # interrupted; the temperature is refitted at the end.
                output.mkdir(parents=True, exist_ok=True)
                AutoConfig.from_pretrained(base_model).save_pretrained(output / "encoder_config")
                tokenizer.save_pretrained(output / "tokenizer")
                torch.save(checkpoint_payload(best_state, 1.0, best_epoch, history),
                           output / "model.pt")
        elif epoch - best_epoch >= patience:
            if verbose:
                print(json.dumps({"early_stop_at_epoch": epoch, "best_epoch": best_epoch,
                                  "best_validation_macro_f1": best_macro}), flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(target_device)

    val_logits, val_targets = predict_logits(model, loaders["validation"], target_device, amp_dtype)
    temperature = fit_temperature(val_logits, val_targets)
    test_logits, test_targets = predict_logits(model, loaders["test"], target_device, amp_dtype)
    test_metrics = evaluate_logits(test_logits, test_targets, labels, temperature)
    test_metrics.pop("_confidence")
    test_metrics.pop("_prediction")
    val_metrics = evaluate_logits(val_logits, val_targets, labels, temperature)
    val_metrics.pop("_confidence")
    val_metrics.pop("_prediction")

    output.mkdir(parents=True, exist_ok=True)
    AutoConfig.from_pretrained(base_model).save_pretrained(output / "encoder_config")
    tokenizer.save_pretrained(output / "tokenizer")
    torch.save(checkpoint_payload(best_state, temperature, best_epoch, history), output / "model.pt")

    extra: dict[str, Any] = {}
    for name, rows in (eval_extra or {}).items():
        if not rows:
            continue
        dataset = FieldRows(rows, labels, tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id), num_workers=workers,
                            persistent_workers=workers > 0)
        logits, targets = predict_logits(model, loader, target_device, amp_dtype)
        metrics = evaluate_logits(logits, targets, labels, temperature)
        metrics.pop("_confidence")
        metrics.pop("_prediction")
        extra[name] = metrics

    result = {
        "seed": seed,
        "base_model": base_model,
        "labels": labels,
        "label_count": len(labels),
        "label_tasks": {label: FIELD_TASKS.get(label, "other") for label in labels},
        "train_rows": len(train_rows),
        "train_rows_before_oversampling": sum(counts.values()),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "train_label_counts": dict(counts),
        "labels_without_training_examples": [label for label in labels
                                              if counts.get(label, 0) == 0],
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "temperature": temperature,
        "validation": val_metrics,
        "test": test_metrics,
        "extra_evaluations": extra,
        "device": str(target_device),
        "gpu": torch.cuda.get_device_name(0) if target_device.type == "cuda" else None,
        "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9)
        if target_device.type == "cuda" else None,
        "train_seconds": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": (output / "model.pt").stat().st_size,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the real-text academic field classifier")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="artifacts/field-classifier")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--layer-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--oversample-min", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261019])
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-each-epoch", action="store_true",
                        help="Write model.pt whenever validation improves (long runs)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Tokenizer workers (needs fork start method; on Windows spawn "
                             "fails under `python -m`, so raise --batch-size instead)")
    args = parser.parse_args(argv)

    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    test_rows = load_jsonl(args.test)
    results = []
    for seed in args.seeds:
        output = Path(args.output) if len(args.seeds) == 1 else Path(args.output) / f"seed-{seed}"
        print(json.dumps({"seed": seed, "output": str(output)}), flush=True)
        results.append(train_one(
            seed, train_rows, validation_rows, test_rows, output,
            base_model=args.base_model, epochs=args.epochs, batch_size=args.batch_size,
            encoder_lr=args.encoder_lr, head_lr=args.head_lr, layer_decay=args.layer_decay,
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            max_length=args.max_length, patience=args.patience,
            class_weight_power=args.class_weight_power, oversample_min=args.oversample_min,
            device=args.device, workers=args.workers, save_each_epoch=args.save_each_epoch))
    summary = {
        "seeds": args.seeds,
        "test_macro_f1": [r["test"]["macro_f1"] for r in results],
        "test_accuracy": [r["test"]["accuracy"] for r in results],
        "mean_macro_f1": float(np.mean([r["test"]["macro_f1"] for r in results])),
        "std_macro_f1": float(np.std([r["test"]["macro_f1"] for r in results])),
        "mean_accuracy": float(np.mean([r["test"]["accuracy"] for r in results])),
        "std_accuracy": float(np.std([r["test"]["accuracy"] for r in results])),
        "per_run": results,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_run"}, indent=2))


if __name__ == "__main__":
    main()
