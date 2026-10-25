"""Compact encoder classifier with a typed, task-scoped output vocabulary."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .labels import TASK_LABELS

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 160
CLASS_NAMES = [f"{task}::{label}" for task, labels in TASK_LABELS.items() for label in labels]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


class PaceClassifier(nn.Module):
    """One shared pretrained text encoder and a single typed classification head."""

    def __init__(self, base_model: str = BASE_MODEL, dropout: float = 0.15, config_dir: str | None = None,
                 num_classes: int | None = None):
        super().__init__()
        if config_dir:
            config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(base_model)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = int(num_classes or len(CLASS_NAMES))
        self.classifier = nn.Linear(hidden, self.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


def task_class_ids(task: str) -> list[int]:
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_LABELS)}")
    return [CLASS_TO_ID[f"{task}::{label}"] for label in TASK_LABELS[task]]


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[PaceClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_rel = checkpoint.get("encoder_config")
    config_dir = str((Path(path).parent / config_rel).resolve()) if config_rel else None
    model = PaceClassifier(checkpoint["base_model"], config_dir=config_dir,
                           num_classes=checkpoint.get("num_classes"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
