"""Train and compare bounded academic classification models."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModel

from .data import SEED, generate_examples, save_jsonl, split_by_group
from .labels import TASK_LABELS
from .model import BASE_MODEL, CLASS_NAMES, MAX_LENGTH, PaceClassifier


class EncodedRows(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.encoded = tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor([CLASS_NAMES.index(f"{r['task']}::{r['label']}") for r in rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"input_ids": self.encoded["input_ids"][i], "attention_mask": self.encoded["attention_mask"][i], "labels": self.labels[i]}


def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.inference_mode():
        for batch in loader:
            labels_all.extend(batch["labels"].tolist())
            logits_all.append(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float().cpu())
    logits = torch.cat(logits_all)
    y = np.asarray(labels_all)
    pred = logits.argmax(-1).numpy()
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "logits": logits, "labels": y}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default="artifacts")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed);    torch.manual_seed(args.seed)

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; base_model={args.base_model}; torch={torch.__version__}", flush=True)
    if device.type == "cuda": print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    rows = generate_examples(args.examples, args.seed)
    splits = split_by_group(rows, args.seed)
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items(): save_jsonl(data_dir / f"{name}.jsonl", part)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out / "tokenizer")
    AutoConfig.from_pretrained(args.base_model).save_pretrained(out / "encoder_config")
    (out / "labels.json").write_text(json.dumps({"classes": CLASS_NAMES, "tasks": TASK_LABELS,
        "max_length": MAX_LENGTH, "pooling": "attention_masked_mean"}, indent=2) + "\n", encoding="utf-8")
    datasets = {n: EncodedRows(v, tokenizer) for n, v in splits.items()}
    loaders = {n: DataLoader(d, batch_size=args.batch_size, shuffle=(n == "train"), num_workers=0)
               for n, d in datasets.items()}
    model = PaceClassifier(args.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_f1, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); start = time.perf_counter(); total_loss = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += loss.item()
        val = evaluate(model, loaders["validation"], device)
        record = {"epoch": epoch, "train_loss": total_loss / len(loaders["train"]),
                  "validation_accuracy": val["accuracy"], "validation_macro_f1": val["macro_f1"],
                  "epoch_seconds": time.perf_counter()-start}
        history.append(record); print(json.dumps(record), flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "base_model": args.base_model,
                        "encoder_config": "encoder_config", "tokenizer_dir": "tokenizer",
                        "class_names": CLASS_NAMES, "task_labels": TASK_LABELS,
                        "max_length": MAX_LENGTH, "seed": args.seed,
                        "epoch": epoch, "history": history}, out / "pace_classifier.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state); model.eval()
    # Export encoder + task head separately at FP32 for portable iOS conversion and inference.
    export_path = out / "model_state.pt"
    torch.save(best_state, export_path)
    test = evaluate(model, loaders["test"], device)
    # Scalar temperature optimized on validation logits; held-out test is not used to fit it.
    val = evaluate(model, loaders["validation"], device)
    temperature = torch.nn.Parameter(torch.ones(()))
    optimizer_t = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)
    val_y = torch.tensor(val["labels"], dtype=torch.long)
    def closure():
        optimizer_t.zero_grad()
        loss = torch.nn.functional.cross_entropy(val["logits"] / temperature.clamp(0.05, 20), val_y)
        loss.backward(); return loss
    optimizer_t.step(closure)
    temperature_value = float(temperature.detach().clamp(0.05, 20))
    test_probs = torch.softmax(test["logits"] / temperature_value, -1)
    test_y = torch.tensor(test["labels"])
    confidence, prediction = test_probs.max(-1)
    ece = 0.0
    for lo in torch.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - (prediction[mask] == test_y[mask]).float().mean()).abs())
    high = confidence >= 0.9
    metrics = {"test_accuracy": float((prediction == test_y).float().mean()),
               "test_macro_f1": float(f1_score(test["labels"], prediction.numpy(), average="macro")),
               "test_ece_10bin": ece, "temperature": temperature_value,
               "high_confidence_threshold": 0.9,
               "high_confidence_coverage": float(high.float().mean()),
               "high_confidence_precision": float((prediction[high] == test_y[high]).float().mean()) if high.any() else None,
               "examples": len(rows), "splits": {k: len(v) for k,v in splits.items()},
               "classes": len(CLASS_NAMES), "base_model": args.base_model, "epochs_trained": len(history),
               "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "seed": args.seed, "real_document_metrics": None,
               "checkpoint_bytes": (out / "pace_classifier.pt").stat().st_size,
               "state_dict_bytes": export_path.stat().st_size,
               "parameter_count": sum(p.numel() for p in model.parameters())}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    (out / "training_history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps({"temperature": temperature_value, "validation_only": True}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__": main()
