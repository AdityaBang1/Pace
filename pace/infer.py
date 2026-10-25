"""Local calibrated inference and hierarchical long-document orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .labels import TASK_LABELS
from .model import MAX_LENGTH, CLASS_TO_ID, load_checkpoint
from .text_structure import split_sections


class PaceEngine:
    def __init__(self, checkpoint: str | Path, calibration: str | Path | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_checkpoint(str(checkpoint), self.device)
        tokenizer_dir = self.meta.get("tokenizer_dir")
        local_tokenizer = Path(checkpoint).parent / tokenizer_dir if tokenizer_dir else None
        if local_tokenizer and local_tokenizer.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer), local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.meta["base_model"])
        calibration = Path(calibration) if calibration else Path(checkpoint).with_name("calibration.json")
        self.temperature = json.loads(calibration.read_text(encoding="utf-8"))["temperature"] if calibration.exists() else 1.0
        self.threshold = 0.72

    @torch.inference_mode()
    def classify(self, text: str, task: str) -> dict[str, Any]:
        if task not in TASK_LABELS: raise ValueError(f"unsupported task: {task}")
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                                 padding="max_length")
        logits = self.model(encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device))[0]
        ids = [CLASS_TO_ID[f"{task}::{label}"] for label in TASK_LABELS[task]]
        probs = torch.softmax(logits[ids] / max(float(self.temperature), 0.05), dim=-1)
        confidence, index = probs.max(-1)
        label = TASK_LABELS[task][int(index)]
        score = float(confidence)
        return {"task": task, "label": label if score >= self.threshold else "unknown",
                "proposed_label": label, "confidence": score,
                "abstained": score < self.threshold, "source_text": text}

    def analyze_document(self, text: str) -> dict[str, Any]:
        sections = split_sections(text)
        document = self.classify(text[:2000], "document")
        interpreted = []
        for section in sections:
            section_input = f"{section['title']}: {section['text']}"
            interpreted.append({"title": section["title"], **self.classify(section_input, "section")})
        return {"schema_version": "1.0", "document_type": document,
                "sections": interpreted, "source_characters": len(text),
                "requires_review": document["abstained"] or any(s["abstained"] for s in interpreted),
                "commit_policy": "proposals_only_user_confirmation_required"}
