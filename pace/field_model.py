"""Real-text academic field classifier.

A single shared pretrained encoder with a flat softmax head over a large,
explicit vocabulary of academic-document field types. The head is trained on
real text (course catalogs, syllabi, course schedules) and reports calibrated
per-type probabilities so callers can act on a percentage rather than a
hard label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Grouping used for reporting and for the abstain path: a caller that only wants
# "when is this class?" can collapse the fine-grained vocabulary onto the task.
# The keys are exactly the corpus vocabulary in ``pace.field_corpus.FIELD_TYPES``.
FIELD_TASKS: dict[str, str] = {
    # Course identity
    "course_title": "course_identity",
    "course_code": "course_identity",
    "course_description": "course_identity",
    "program": "course_identity",
    "course_level": "course_identity",
    "class_type": "course_identity",
    # Instructor & contact
    "instructor": "instructor_contact",
    "office_hours": "instructor_contact",
    "contact_information": "instructor_contact",
    # Enrollment
    "credits": "enrollment",
    "capacity": "enrollment",
    "class_section": "enrollment",
    "entry_requirements": "enrollment",
    "restrictions": "enrollment",
    "confirmation": "enrollment",
    # Structure and content
    "prerequisites": "course_structure",
    "course_content": "course_structure",
    "learning_outcomes": "course_structure",
    "teaching_form": "course_structure",
    "language_of_instruction": "course_structure",
    # Materials & assessment
    "materials": "materials_assessment",
    "weekly_readings": "materials_assessment",
    "assignments": "materials_assessment",
    "due_date": "materials_assessment",
    "grading_policy": "materials_assessment",
    "grades": "materials_assessment",
    "assessment": "materials_assessment",
    "course_evaluation": "materials_assessment",
    # Schedule
    "meeting_time": "schedule",
    "meeting_days": "schedule",
    "class_meeting_room": "schedule",
    "course_modality": "schedule",
    "course_schedule": "schedule",
    "exam_schedule": "schedule",
    "session_dates": "schedule",
    "term": "schedule",
    "drop_add_deadline": "schedule",
    # Policy
    "attendance_policy": "policy",
    "academic_integrity": "policy",
    "course_policies": "policy",
    "disability_services": "policy",
    # Navigation / catch-all
    "additional_information": "navigation",
    "other": "navigation",
}

FIELD_LABELS: list[str] = list(FIELD_TASKS)
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(FIELD_LABELS)}


class FieldClassifierModel(nn.Module):
    """Shared encoder plus a flat field-type head."""

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, num_classes: int | None = None,
                 dropout: float = 0.1, config_dir: str | None = None):
        super().__init__()
        if config_dir:
            config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(base_model)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = int(num_classes or len(FIELD_LABELS))
        self.classifier = nn.Linear(hidden, self.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


def load_field_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a trained field classifier plus its tokenizer, labels and temperature.

    ``path`` may be the checkpoint file itself or the directory that holds it.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_rel = checkpoint.get("encoder_config")
    config_dir = str((path.parent / config_rel).resolve()) if config_rel else None
    model = FieldClassifierModel(
        checkpoint["base_model"], num_classes=checkpoint.get("num_classes"),
        config_dir=config_dir,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / checkpoint.get("tokenizer_dir", "tokenizer"))
    return model, checkpoint, tokenizer


class FieldEngine:
    """Inference wrapper returning calibrated per-type percentages."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu",
                 max_length: int | None = None):
        self.model, self.checkpoint, self.tokenizer = load_field_checkpoint(checkpoint, device)
        self.device = torch.device(device)
        self.max_length = int(max_length or self.checkpoint.get("max_length", 160))
        self.temperature = float(self.checkpoint.get("temperature", 1.0))
        self.labels: list[str] = list(self.checkpoint.get("labels") or FIELD_LABELS)
        self.label_tasks = {label: FIELD_TASKS.get(label, "other") for label in self.labels}

    @torch.inference_mode()
    def probabilities(self, texts: Iterable[str]) -> list[dict[str, float]]:
        items = [t if t.strip() else " " for t in texts]
        if not items:
            return []
        encoded = self.tokenizer(items, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        logits = self.model(encoded["input_ids"], encoded["attention_mask"]).float()
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return [dict(zip(self.labels, row)) for row in probs.cpu().tolist()]

    def classify(self, text: str, top_k: int = 5) -> dict[str, Any]:
        """Classify one text; returns the label, percentage and full ranking."""
        probs = self.probabilities([text])[0]
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        label, confidence = ranked[0]
        return {
            "label": label,
            "task": self.label_tasks.get(label, "other"),
            "confidence": confidence,
            "percent": round(confidence * 100, 2),
            "top": [{"label": l, "task": self.label_tasks.get(l, "other"),
                     "probability": p, "percent": round(p * 100, 2)}
                    for l, p in ranked[:top_k]],
            "probabilities": {l: p for l, p in ranked},
            "model": self.checkpoint.get("base_model"),
            "temperature": self.temperature,
        }

    def classify_many(self, texts: Iterable[str], top_k: int = 5) -> list[dict[str, Any]]:
        """Vectorised classification returning one record per input text."""
        items = list(texts)
        prob_rows = self.probabilities(items)
        out = []
        for text, probs in zip(items, prob_rows):
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            label, confidence = ranked[0]
            out.append({
                "text": text,
                "label": label,
                "task": self.label_tasks.get(label, "other"),
                "confidence": confidence,
                "percent": round(confidence * 100, 2),
                "top": [{"label": l, "task": self.label_tasks.get(l, "other"),
                         "probability": p, "percent": round(p * 100, 2)}
                        for l, p in ranked[:top_k]],
            })
        return out


def classify_section(heading: str, value: str, heading_engine: "FieldEngine",
                     context_engine: "FieldEngine | None" = None,
                     threshold: float = 0.8, max_chars: int = 400) -> dict[str, Any]:
    """Classify a document section from its heading and the text under it.

    The heading is the field's identity, so it is asked first. Only when the
    heading is unfamiliar -- the engine says so with a low calibrated
    percentage -- does the context model, which reads the heading together
    with the value, take over. Measured on 446 real labelled sections this
    reaches 0.8879 accuracy at 0.888 macro-F1 with a 7.4% fallback rate.
    """
    heading = (heading or "").strip()
    value = (value or "").strip()
    heading_result = heading_engine.classify(heading) if heading else None
    if context_engine is None or not value:
        if heading_result is None:
            return {"label": None, "task": None, "confidence": 0.0, "percent": 0.0,
                    "source": "none", "fallback": False, "heading": None}
        return {
            "label": heading_result["label"], "task": heading_result["task"],
            "confidence": heading_result["confidence"], "percent": heading_result["percent"],
            "source": "heading",
            "fallback": False,
            "heading": heading_result,
        }
    combined = f"{heading}: {value[:max_chars]}" if heading else value[:max_chars]
    context_result = context_engine.classify(combined)
    use_context = (heading_result is None
                   or heading_result["confidence"] < threshold)
    chosen = context_result if use_context else heading_result
    return {
        "label": chosen["label"],
        "task": chosen["task"],
        "confidence": chosen["confidence"],
        "percent": chosen["percent"],
        "source": "heading+value" if use_context else "heading",
        "fallback": use_context,
        "threshold": threshold,
        "heading": heading_result,
        "heading+value": context_result,
    }
