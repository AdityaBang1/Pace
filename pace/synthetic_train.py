"""Train compact PyTorch document-type classifier on template-held-out synthetic academic docs."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

from .model import BASE_MODEL, MAX_LENGTH, PaceClassifier
from .synthetic_academic import DOCUMENT_TYPES, generate_dataset, split_by_template


class DocumentRows(Dataset):
    def __init__(self, rows, tokenizer, labels, max_length):
        self.labels = labels
        self.y = torch.tensor([labels.index(row["document_type"]) for row in rows], dtype=torch.long)
        self.encoded = tokenizer(
            [row.get("ocr_text", row["text"]) for row in rows],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encoded["input_ids"][idx],
            "attention_mask": self.encoded["attention_mask"][idx],
            "labels": self.y[idx],
        }


def _predict(model, loader, device):
    model.eval()
    logits, truth = [], []
    with torch.inference_mode():
        for batch in loader:
            logits.append(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float().cpu())
            truth.extend(batch["labels"].tolist())
    return torch.cat(logits), np.asarray(truth)


def _group_split_metrics(rows, predictions):
    labels = [row["document_type"] for row in rows]
    return {
        "rows": len(rows),
        "template_families": sorted({row["template_family"] for row in rows}),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(labels, predictions, labels=list(DOCUMENT_TYPES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(DOCUMENT_TYPES)).tolist(),
        "errors": [
            {"true": row["document_type"], "predicted": pred, "template_family": row["template_family"], "text": row["text"]}
            for row, pred in zip(rows, predictions) if row["document_type"] != pred
        ],
    }


def train_and_evaluate(
    output: str | Path = "artifacts/synthetic-doc-type",
    count: int = 12000,
    seed: int = 20261019,
    epochs: int = 5,
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
    base_model: str = BASE_MODEL,
    corruption_rate: float = 0.02,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    docs = generate_dataset(count, seed, corruption_rate=corruption_rate)
    splits = split_by_template(docs, seed)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    labels = list(DOCUMENT_TYPES)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, strip_accents="unicode")
    sparse = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    train_x = vectorizer.fit_transform([row["text"] for row in splits["train"]])
    sparse.fit(train_x, [row["document_type"] for row in splits["train"]])
    sparse_preds = sparse.predict(vectorizer.transform([row["text"] for row in splits["test"]])).tolist()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    AutoConfig.from_pretrained(base_model).save_pretrained(output_dir / "encoder_config")
    datasets = {name: DocumentRows(rows, tokenizer, labels, max_length) for name, rows in splits.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"), num_workers=0)
        for name, dataset in datasets.items()
    }

    model = PaceClassifier(base_model, num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_macro_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits, val_y = _predict(model, loaders["validation"], device)
        val_preds = [labels[int(i)] for i in val_logits.argmax(-1).tolist()]
        truth = [labels[int(i)] for i in val_y]
        val_macro = float(f1_score(truth, val_preds, average="macro", zero_division=0))
        record = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "validation_accuracy": float(accuracy_score(truth, val_preds)),
            "validation_macro_f1": val_macro,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No model checkpoint selected")
    model.load_state_dict(best_state)
    test_logits, test_y = _predict(model, loaders["test"], device)
    neural_preds = [labels[int(i)] for i in test_logits.argmax(-1).tolist()]
    test_truth = [labels[int(i)] for i in test_y]

    ckpt = {
        "state_dict": best_state,
        "base_model": base_model,
        "num_classes": len(labels),
        "encoder_config": "encoder_config",
        "tokenizer_dir": "tokenizer",
        "labels": labels,
        "max_length": max_length,
        "seed": seed,
        "task": "synthetic academic document type classification",
        "template_split": {name: sorted({row["template_family"] for row in rows}) for name, rows in splits.items()},
    }
    torch.save(ckpt, output_dir / "model.pt")

    # Clean text-only test uses clean text generated from the same held-out templates.
    clean_test_texts = [row["text"] for row in splits["test"]]
    clean_enc = tokenizer(clean_test_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    clean_logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(splits["test"]), batch_size):
            ids = clean_enc["input_ids"][start:start + batch_size].to(device)
            mask = clean_enc["attention_mask"][start:start + batch_size].to(device)
            clean_logits.append(model(ids, mask).cpu())
    clean_preds = [labels[int(i)] for i in torch.cat(clean_logits).argmax(-1).tolist()]

    metrics = {
        "task": "synthetic academic document type classification",
        "data": "Procedurally generated synthetic academic worlds and text documents; mild character-level OCR-like corruption only, not rendered/image OCR.",
        "seed": seed,
        "generated_examples": count,
        "corruption_rate": corruption_rate,
        "split_protocol": "template-family-disjoint within each document class; synthetic worlds are generated independently per document",
        "template_families_by_split": {name: sorted({row["template_family"] for row in rows}) for name, rows in splits.items()},
        "rows_by_split": {name: len(rows) for name, rows in splits.items()},
        "labels": labels,
        "training_device": str(device),
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "checkpoint_bytes": (output_dir / "model.pt").stat().st_size,
        "history": history,
        "test_ocr_text_corrupted": _group_split_metrics(splits["test"], neural_preds),
        "test_clean_text": _group_split_metrics(splits["test"], clean_preds),
        "baseline_tfidf_logistic": _group_split_metrics(splits["test"], sparse_preds),
        "safety_note": "Synthetic-only results are not evidence of real-document performance, OCR accuracy, or deployment readiness.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a PyTorch document classifier on template-held-out synthetic data.")
    parser.add_argument("--output", default="artifacts/synthetic-doc-type")
    parser.add_argument("--count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--corruption-rate", type=float, default=0.02)
    args = parser.parse_args(argv)
    metrics = train_and_evaluate(args.output, args.count, args.seed, args.epochs, args.batch_size,
                                 args.max_length, args.base_model, args.corruption_rate)
    print(json.dumps({key: metrics[key] for key in (
        "generated_examples", "rows_by_split", "template_families_by_split", "parameter_count", "checkpoint_bytes",
        "test_ocr_text_corrupted", "test_clean_text", "baseline_tfidf_logistic", "safety_note"
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
