"""Train the Jev-shaped field decision model, backwards from our corpus.

Two stages, because that is what the shape allows:

``pretrain``
    Masked-language-model training on the project's own real text. A 10 M
    encoder cannot arrive with a large corpus's worth of language baked in the
    way MiniLM did, so it learns the vocabulary of academic documents first
    (headings, catalog columns, syllabus vocabulary) from 400k unlabelled rows.

``choice``
    The Jev objective: state + question + 42 described options, cross-entropy on
    the correct option, one temperature fitted on held-out rows. The option
    menu is made of descriptions, so a field can be re-described without
    retraining the text tower.

Both stages run on one GPU in minutes, and the resulting checkpoint is small
enough for a phone.

    python -m pace.jev_train --stage tokenizer --output artifacts/jev-field
    python -m pace.jev_train --stage pretrain --output artifacts/jev-field
    python -m pace.jev_train --stage choice --output artifacts/jev-field \
        --train data/processed/field_mix/train.jsonl \
        --validation data/processed/field_mix/validation.jsonl \
        --test data/processed/field_mix/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import BertPreTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

from .jev_model import (CLS, DEFAULT_CONFIG, DEFAULT_QUESTION, FIELD_DESCRIPTIONS, MASK,
                        PAD, SEP, UNK, JevChoiceModel, JevConfig, mean_pool, save_jev)
from .field_train import load_jsonl


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

def train_tokenizer(paths: list[str | Path], output: str | Path, vocab_size: int = 16000,
                    limit: int = 400_000) -> Tokenizer:
    """A 16 K WordPiece vocabulary over the project's own real text.

    Every character seen in the corpus is kept as a single-character token by
    the WordPiece trainer, so an unseen word still splits into known pieces
    instead of one [UNK] — the same property TinyDecide advertises.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordPiece(unk_token=UNK, continuing_subword_prefix="##"))
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = BertPreTokenizer()

    def iterator() -> Iterator[str]:
        seen = 0
        for path in paths:
            with Path(path).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not row.get("text"):
                        continue
                    yield row["text"].lower()
                    seen += 1
                    if seen >= limit:
                        return

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=[PAD, UNK, CLS, SEP, MASK],
        continuing_subword_prefix="##", min_frequency=3, show_progress=True,
    )
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.save(str(output / "tokenizer.json"))
    print(json.dumps({"tokenizer": str(output / "tokenizer.json"),
                      "vocab_size": tokenizer.get_vocab_size()}), flush=True)
    return tokenizer


def raw_texts(paths: list[str | Path], limit: int) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if len(texts) >= limit:
                    return texts
                if not line.strip():
                    continue
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if text:
                    texts.append(text)
    return texts


# --------------------------------------------------------------------------- #
# Masked-language-model pretraining
# --------------------------------------------------------------------------- #

class MaskedDataset(Dataset):
    """Whole-word masked LM rows, re-masked every epoch.

    TinyDecide masks single WordPiece pieces. Masking whole words instead forces
    the encoder to use syntax and surrounding words rather than finishing a
    fragment, and clearing the mask each epoch means the same passage is never
    seen twice with the same holes. Token ids are cached once as int16 so a
    million rows cost a few hundred MB instead of several GB.

    Word starts are recovered from the WordPiece ``##`` prefix, so a mask covers
    every piece of the word plus ~20% of the time the following word, which is
    the span-masking recipe ELECTRA-style pretraining uses.
    """

    def __init__(self, texts: list[str], tokenizer: Tokenizer, max_length: int = 128,
                 mask_prob: float = 0.15, span_prob: float = 0.2, seed: int = 20261019):
        cls = tokenizer.token_to_id(CLS)
        sep = tokenizer.token_to_id(SEP)
        self.mask_id = tokenizer.token_to_id(MASK)
        self.pad = tokenizer.token_to_id(PAD)
        self.vocab_size = tokenizer.get_vocab_size()
        self.mask_prob = mask_prob
        self.span_prob = span_prob
        self.seed = seed
        self.epoch = 0
        # One flat int16 buffer plus an offset per row: a million short passages
        # cost tens of MB and random access stays cache-friendly, where a list of
        # per-row arrays costs gigabytes and slows the whole loop down.
        encoded: list[list[int]] = []
        boundaries: list[list[bool]] = []
        for text in texts:
            encoding = tokenizer.encode(text)
            ids = encoding.ids[: max_length - 2]
            pieces = encoding.tokens[: max_length - 2]
            encoded.append([cls, *ids, sep])
            # A token begins a word when it is not a continuing subword piece.
            boundaries.append([True, *[not piece.startswith("##") for piece in pieces], True])
        lengths = np.asarray([len(ids) for ids in encoded], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self.tokens = np.asarray([token for ids in encoded for token in ids], dtype=np.int16)
        self.starts = np.asarray([flag for flags in boundaries for flag in flags], dtype=bool)

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        start, end = self.offsets[index], self.offsets[index + 1]
        seq, starts = self.tokens[start:end], self.starts[start:end]
        ids = seq.astype(np.int64)
        labels = np.full(len(ids), -100, dtype=np.int64)
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)
        interior = np.flatnonzero(starts[1:-1]) + 1
        chosen = [int(position) for position in interior if rng.random() < self.mask_prob]
        spans: list[int] = []
        for position in chosen:
            spans.append(position)
            while position + 1 < len(ids) - 1 and not starts[position + 1]:
                position += 1
                spans.append(position)
            if rng.random() < self.span_prob and position + 1 < len(ids) - 1:
                position += 1
                spans.append(position)
                while position + 1 < len(ids) - 1 and not starts[position + 1]:
                    position += 1
                    spans.append(position)
        for position in dict.fromkeys(spans):
            labels[position] = ids[position]
            draw = rng.random()
            if draw < 0.8:
                ids[position] = self.mask_id
            elif draw < 0.9 and self.vocab_size > 6:
                ids[position] = rng.randint(5, self.vocab_size - 1)
        return (torch.from_numpy(ids), torch.from_numpy(labels))


def _pad_batch(batch, pad_id: int):
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (ids, target) in enumerate(batch):
        input_ids[row, :len(ids)] = ids
        labels[row, :len(target)] = target
        attention[row, :len(ids)] = 1
    return input_ids, attention, labels


def pretrain(tokenizer: Tokenizer, texts: list[str], output: Path, epochs: int = 2,
             batch_size: int = 128, max_length: int = 128, lr: float = 5e-4,
             device: str | None = None, arch: dict[str, Any] | None = None) -> dict[str, Any]:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = JevConfig.from_dict({**DEFAULT_CONFIG, **(arch or {}),
                                  "vocab_size": tokenizer.get_vocab_size()})
    model = JevChoiceModel(config, list(FIELD_DESCRIPTIONS))  # encoder is what trains here
    encoder = model.encoder.to(target)
    head = torch.nn.Linear(config.embed_dim, config.vocab_size, bias=False).to(target)
    head.weight = encoder.embed.weight  # tied
    # Hidden width is 256, embeddings are 128; project before the tied head.
    mlm_proj = torch.nn.Linear(config.hidden, config.embed_dim).to(target)

    started = time.perf_counter()
    data = MaskedDataset(texts, tokenizer, max_length=max_length)
    print(json.dumps({"masked_rows": len(data),
                      "tokenise_seconds": round(time.perf_counter() - started, 1)}),
          flush=True)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=True,
                        collate_fn=lambda batch: _pad_batch(batch, data.pad))
    # head.weight is encoder.embed.weight (tied), so it must not be added twice.
    parameters = list(encoder.parameters()) + list(mlm_proj.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    total = max(1, len(loader) * epochs)
    warmup = int(total * 0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / max(1, warmup) if step < warmup else
        max(0.05, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))))

    history = []
    report_every = max(1, len(loader) // 12)
    for epoch in range(1, epochs + 1):
        data.set_epoch(epoch)
        encoder.train()
        start = time.perf_counter()
        running = 0.0
        for step, (input_ids, attention, labels) in enumerate(loader, start=1):
            input_ids, attention = input_ids.to(target), attention.to(target)
            labels = labels.to(target)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=target.type == "cuda"):
                hidden = encoder(input_ids, attention)
                logits = head(mlm_proj(hidden.float()))
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                                       labels.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            if step % report_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "of": len(loader),
                                  "running_loss": round(running / step, 4),
                                  "elapsed": round(time.perf_counter() - start, 1)}),
                      flush=True)
        record = {"epoch": epoch, "masked_loss": running / max(1, len(loader)),
                  "seconds": round(time.perf_counter() - start, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    save_jev(model, tokenizer, output, extra={"stage": "pretrained",
                                              "pretrain_history": history})
    # Keep the language-only weights too: a later fine-tune should be able to
    # start from the pretrained encoder without re-running pretraining.
    save_jev(model, tokenizer, output / "pretrained",
             extra={"stage": "pretrained", "pretrain_history": history})
    return {"rows": len(data), "history": history,
            "parameters": model.parameter_count()}


# --------------------------------------------------------------------------- #
# Choice fine-tuning
# --------------------------------------------------------------------------- #

class ChoiceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Tokenizer, labels: list[str],
                 question: str = DEFAULT_QUESTION, max_state: int = 127,
                 max_option: int = 24):
        self.index = {label: position for position, label in enumerate(labels)}
        self.tokenizer = tokenizer
        self.question_ids = [*tokenizer.encode(question).ids, tokenizer.token_to_id(SEP)]
        self.rows = []
        cls = tokenizer.token_to_id(CLS)
        sep = tokenizer.token_to_id(SEP)
        for row in rows:
            if row["label"] not in self.index:
                continue
            state = tokenizer.encode((row["text"] or " ").lower()).ids[:max_state]
            ids = [cls, *state, sep, *self.question_ids]
            # Pool the state tokens only, exactly as the served engine does.
            mask = [0] + [1] * len(state) + [0] * (len(ids) - 1 - len(state))
            self.rows.append((ids, mask, self.index[row["label"]]))
        options = tokenizer.encode_batch([FIELD_DESCRIPTIONS.get(label, label)
                                          for label in labels])
        width = min(max(len(item.ids) for item in options), max_option)
        self.option_ids = torch.zeros(len(options), width, dtype=torch.long)
        self.option_mask = torch.zeros(len(options), width, dtype=torch.long)
        for position, item in enumerate(options):
            length = min(len(item.ids), width)
            self.option_ids[position, :length] = torch.tensor(item.ids[:length])
            self.option_mask[position, :length] = 1
        self.pad = tokenizer.token_to_id(PAD)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        ids, mask, label = self.rows[index]
        return torch.tensor(ids), torch.tensor(mask), label


def _choice_collate(batch, pad_id: int):
    width = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    state_mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for row, (ids, mask, label) in enumerate(batch):
        input_ids[row, :len(ids)] = ids
        attention[row, :len(ids)] = 1
        state_mask[row, :len(mask)] = mask
        labels[row] = label
    return input_ids, attention, state_mask, labels


@torch.no_grad()
def evaluate_choice(model: JevChoiceModel, dataset: ChoiceDataset, device: torch.device,
                    temperature: float = 1.0) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        collate_fn=lambda batch: _choice_collate(batch, dataset.pad))
    model.eval()
    logits_all, targets_all = [], []
    option_ids, option_mask = dataset.option_ids.to(device), dataset.option_mask.to(device)
    for input_ids, attention, state_mask, labels in loader:
        logits = model(input_ids.to(device), attention.to(device), state_mask.to(device),
                       option_ids, option_mask)
        logits_all.append(logits.float().cpu())
        targets_all.append(labels)
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all).numpy()
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    confidence, prediction = probs.max(dim=-1)
    prediction = prediction.numpy()
    confidence = confidence.numpy()
    correct = prediction == targets
    bins = 10
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & ((confidence < high) if index < bins - 1 else confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return {
        "rows": int(len(targets)),
        "accuracy": float(accuracy_score(targets, prediction)),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, prediction, average="weighted", zero_division=0)),
        "ece_10bin": ece,
        "mean_confidence": float(confidence.mean()),
        "logits": logits,
        "targets": targets,
    }


def _menu_loss(logits: torch.Tensor, labels: torch.Tensor,
               allowed: torch.Tensor | None, smoothing: float = 0.05,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy over the options a caller actually offered.

    Masked options are set to a large finite negative so they carry no
    probability mass, and the label-smoothing term is spread only across the
    surviving menu: smoothing over impossible options would swamp the loss with
    the mask value itself.
    """
    masked = logits if allowed is None else logits.masked_fill(~allowed, -1e4)
    log_probs = torch.log_softmax(masked.float(), dim=-1)
    per_row = -(log_probs.gather(1, labels.unsqueeze(1)).squeeze(1))
    if smoothing > 0.0:
        mass = torch.ones_like(log_probs) if allowed is None else allowed.float()
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1.0)
        per_row = (1.0 - smoothing) * per_row + smoothing * (-(log_probs * mass).sum(dim=-1))
    if weights is None:
        return per_row.mean()
    return (per_row * weights).sum() / weights.sum().clamp_min(1e-6)


def _label_counts(dataset: "ChoiceDataset", labels: int) -> np.ndarray:
    counts = np.zeros(labels, dtype=np.float64)
    for _, _, label in dataset.rows:
        counts[label] += 1
    return counts


def _class_sampler(dataset: "ChoiceDataset", labels: int, power: float):
    """Draw rare fields more often, without inventing rows.

    Loss weights re-weight the gradient but the batch a thin class appears in is
    still decided by chance; sampling by ``count ** -power`` decides the batches
    themselves, which is the variant that moves the long tail in practice. The
    sampler keeps the epoch the same size as the dataset, so training time and
    the optimizer's schedule are unchanged.
    """
    from torch.utils.data import WeightedRandomSampler

    counts = _label_counts(dataset, labels).clip(min=1.0)
    weights = (counts / counts.mean()) ** (-power)
    per_row = torch.tensor([float(weights[label]) for _, _, label in dataset.rows],
                           dtype=torch.double)
    generator = torch.Generator().manual_seed(20261019)
    return WeightedRandomSampler(per_row, num_samples=len(per_row), replacement=True,
                                 generator=generator)


def fit_temperature(logits: torch.Tensor, targets: np.ndarray) -> float:
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=60, line_search_fn="strong_wolfe")
    target = torch.as_tensor(targets, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp(0.05, 20.0), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def train_choice(train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict],
                 output: Path, tokenizer: Tokenizer, encoder_checkpoint: Path | None = None,
                 epochs: int = 8, batch_size: int = 128, lr: float = 1e-3,
                 patience: int = 2, option_dropout: float = 0.0,
                 class_weight_power: float = 0.0, sampler_power: float = 0.0,
                 arch: dict[str, Any] | None = None,
                 device: str | None = None) -> dict[str, Any]:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    labels = [label for label in FIELD_DESCRIPTIONS
              if any(row["label"] == label for row in train_rows + validation_rows + test_rows)]
    config = JevConfig.from_dict({**DEFAULT_CONFIG, **(arch or {}),
                                  "vocab_size": tokenizer.get_vocab_size()})
    model = JevChoiceModel(config, labels).to(target_device)
    if encoder_checkpoint is not None and (Path(encoder_checkpoint) / "model.pt").exists():
        checkpoint = torch.load(Path(encoder_checkpoint) / "model.pt", map_location="cpu",
                                weights_only=False)
        # Only shape-compatible tensors carry over: the pretraining pass saved the
        # default label set, and a fold can have a different number of options.
        own = model.state_dict()
        compatible = {key: value for key, value in checkpoint["state_dict"].items()
                      if key in own and own[key].shape == value.shape}
        missing = model.load_state_dict(compatible, strict=False)
        print(json.dumps({"loaded_encoder_from": str(encoder_checkpoint),
                          "transferred": len(compatible),
                          "missing": len(missing.missing_keys),
                          "unexpected": len(missing.unexpected_keys)}), flush=True)

    train = ChoiceDataset(train_rows, tokenizer, labels)
    validation = ChoiceDataset(validation_rows, tokenizer, labels)
    test = ChoiceDataset(test_rows, tokenizer, labels)
    print(json.dumps({"labels": len(labels), "train": len(train),
                      "validation": len(validation), "test": len(test),
                      "parameters": model.parameter_count()}), flush=True)

    loader = DataLoader(train, batch_size=batch_size, shuffle=sampler_power <= 0.0,
                        num_workers=0, sampler=_class_sampler(train, len(labels), sampler_power)
                        if sampler_power > 0.0 else None,
                        collate_fn=lambda batch: _choice_collate(batch, train.pad))
    if sampler_power > 0.0:
        print(json.dumps({"sampler_power": sampler_power,
                          "rarest": [[labels[index], int(count)]
                                     for index, count in sorted(enumerate(_label_counts(
                                         train, len(labels))), key=lambda item: item[1])[:5]]}),
              flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = max(1, len(loader) * epochs)
    warmup = int(total * 0.06)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / max(1, warmup) if step < warmup else
        max(0.05, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))))
    option_ids = train.option_ids.to(target_device)
    option_mask = train.option_mask.to(target_device)
    smoothing = 0.05

    history, best = [], -1.0
    best_epoch, best_state = 0, None
    class_weights = None
    if class_weight_power > 0.0:
        # Rare fields are exactly the ones a reader needs (attendance policy,
        # disability services). Plain cross-entropy spends its capacity on
        # course_code; inverse-frequency weights buy the long tail back.
        counts = torch.zeros(len(labels), dtype=torch.float64)
        for _, _, label in train.rows:
            counts[label] += 1
        safe = counts.clamp_min(1.0)
        weights = (safe / safe.mean()).pow(-class_weight_power)
        class_weights = (weights / weights.mean()).float().to(target_device)
        print(json.dumps({"class_weight_power": class_weight_power,
                          "rarest": [[labels[index], int(counts[index]),
                                      round(float(weights[index]), 2)]
                                     for index in counts.argsort()[:5].tolist()]}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        running = 0.0
        for input_ids, attention, state_mask, labels_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
                logits = model(input_ids.to(target_device), attention.to(target_device),
                               state_mask.to(target_device), option_ids, option_mask)
                allowed = None
                if option_dropout > 0.0:
                    # Jev's point is choosing between described criteria, not 42
                    # fixed classes: hide a random slice of the menu each step so
                    # the score has to be earned by the description, and so the
                    # model stays usable when a caller passes a shorter menu.
                    allowed = torch.rand(logits.shape, device=logits.device) >= option_dropout
                    allowed.scatter_(1, labels_batch.to(target_device).unsqueeze(1), True)
                batch_weights = None if class_weights is None else                     class_weights[labels_batch.to(target_device)]
                loss = _menu_loss(logits, labels_batch.to(target_device), allowed, smoothing,
                                  batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        metrics = evaluate_choice(model, validation, target_device)
        record = {"epoch": epoch, "loss": running / max(1, len(loader)),
                  "validation_accuracy": metrics["accuracy"],
                  "validation_macro_f1": metrics["macro_f1"],
                  "seconds": round(time.perf_counter() - start, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            output.mkdir(parents=True, exist_ok=True)
            save_jev(model, tokenizer, output, extra={"stage": "choice", "question": DEFAULT_QUESTION})
        elif epoch - best_epoch >= patience:
            print(json.dumps({"early_stop_at": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_metrics = evaluate_choice(model, validation, target_device)
    temperature = fit_temperature(validation_metrics.pop("logits"), validation_metrics.pop("targets"))
    test_metrics = evaluate_choice(model, test, target_device, temperature)
    test_metrics.pop("logits")
    test_metrics.pop("targets")

    result = {
        "stage": "choice",
        "base_architecture": (f"ELECTRA-shaped (Jev/TinyDecide): {config.layers}x{config.hidden}, "
                              f"{config.heads} heads, ffn {config.ffn}"),
        "parameter_count": model.parameter_count(),
        "labels": labels,
        "question": DEFAULT_QUESTION,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "temperature": temperature,
        "option_dropout": option_dropout,
        "class_weight_power": class_weight_power,
        "sampler_power": sampler_power,
        "arch": arch or {},
        "validation": {key: value for key, value in validation_metrics.items()
                       if key not in {"logits", "targets"}},
        "test": test_metrics,
        "device": str(target_device),
        "train_seconds": round(sum(item["seconds"] for item in history), 1),
    }
    save_jev(model, tokenizer, output, temperature=temperature,
             extra={"stage": "choice", "question": DEFAULT_QUESTION,
                    "metrics": {key: value for key, value in result.items()
                                if key not in {"labels", "history"}}})
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the Jev-shaped field decision model")
    parser.add_argument("--stage", choices=("tokenizer", "pretrain", "choice", "all"),
                        default="all")
    parser.add_argument("--output", default="artifacts/jev-field")
    parser.add_argument("--encoder", default=None,
                        help="Checkpoint directory to initialise the choice stage from "
                             "(defaults to --output; pass .../pretrained to fine-tune from "
                             "the language-only weights)")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer.json to use (defaults to the one in --output)")
    parser.add_argument("--train", default="data/processed/field_mix/train.jsonl")
    parser.add_argument("--validation", default="data/processed/field_mix/validation.jsonl")
    parser.add_argument("--test", default="data/processed/field_mix/test.jsonl")
    parser.add_argument("--unlabelled", nargs="+",
                        default=["data/processed/field_uncapped/corpus.jsonl"])
    parser.add_argument("--pretrain-rows", type=int, default=400_000)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--option-dropout", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--sampler-power", type=float, default=0.0,
                        help="Draw label i with weight count(i) ** -power; 0 keeps shuffling")
    parser.add_argument("--pretrain-max-length", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--max-state", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    arch = {key: value for key, value in (("hidden", args.hidden), ("layers", args.layers),
                                          ("ffn", args.ffn), ("heads", args.heads),
                                          ("max_state_tokens", args.max_state))
            if value is not None}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else output / "tokenizer.json"
    if args.stage in ("tokenizer", "all"):
        tokenizer = train_tokenizer(args.unlabelled, output)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if args.stage in ("pretrain", "all"):
        texts = raw_texts(args.unlabelled, args.pretrain_rows)
        print(json.dumps({"pretrain_texts": len(texts)}), flush=True)
        print(json.dumps(pretrain(tokenizer, texts, output, epochs=args.pretrain_epochs,
                                  max_length=args.pretrain_max_length, arch=arch,
                                  device=args.device)), flush=True)

    if args.stage in ("choice", "all"):
        train_rows = load_jsonl(args.train)
        validation_rows = load_jsonl(args.validation)
        test_rows = load_jsonl(args.test)
        encoder = Path(args.encoder) if args.encoder else output
        result = train_choice(train_rows, validation_rows, test_rows, output, tokenizer,
                              encoder_checkpoint=encoder, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr,
                              patience=args.patience,
                              option_dropout=args.option_dropout,
                              class_weight_power=args.class_weight_power,
                              sampler_power=args.sampler_power, arch=arch,
                              device=args.device)
        print(json.dumps({key: result[key] for key in
                          ("parameter_count", "best_epoch", "temperature", "test")},
                         indent=2))


if __name__ == "__main__":
    main()
