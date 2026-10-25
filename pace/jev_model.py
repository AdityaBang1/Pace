"""A small Jev-shaped decision model, trained backwards onto academic text.

Jev (TypeSafe AI) is not a classifier with a fixed head: it reads a *state*
(the text), a typed *question* ("which field is this?"), and a menu of *options*
with descriptions, and returns a probability for each option. TinyDecide — the
open 10.4 M build of that idea — is an ELECTRA-small backbone (12 layers,
d=256, 4 heads, FFN 768, 128-d embeddings, 16 K vocabulary) that answers
several questions in one encoder pass and fits in 8 MB of flash.

This module implements that shape at the same size, for one question with a
fixed menu of 42 academic field types:

    q  = pool(encoder("[CLS] state [SEP] question [SEP]"))
    oᵢ = pool(option_tower(descriptionᵢ))
    p  = softmax(scale · cos(q, oᵢ) + biasᵢ)

Options are *descriptions*, not ids, so a field can be renamed, given aliases or
re-described without retraining the text tower, and the 42 option vectors can be
computed once and cached on a phone. The task is the choice API from the Jev
family; the corpus is this project's.

The model is deliberately small enough to run offline on a phone: ~10 M
parameters, ~40 MB fp32, ~10 MB after int8.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .field_model import FIELD_TASKS

#: The question asked of every state. Kept close to the choice API's wording so
#: the model is trained on the same sentence it is served with.
DEFAULT_QUESTION = "Which academic field does this text belong to?"

#: One plain-language description per field type, used as the option menu.
#: These are criteria in the Jev sense: what a text must be about for the option
#: to apply, written so a person could grade an answer against them.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "course_title": "The name of the course, as printed.",
    "course_code": "The course number or code, such as CHEM 1A or MATH 2210.",
    "course_description": "A general description of what the course covers.",
    "program": "The degree, program or department the course belongs to.",
    "course_level": "The level of the course: introductory, upper division, graduate.",
    "class_type": "The kind of class: lecture, lab, seminar, discussion section.",
    "instructor": "The name of the person teaching the course.",
    "office_hours": "When and where the instructor is available to students, such as \"Wednesdays 2-4pm in room 312\".",
    "contact_information": "An email address, phone number or office location.",
    "credits": "How many credits or units the course is worth.",
    "capacity": "How many students may enrol, or how many seats remain.",
    "class_section": "The section number or section identifier of a class.",
    "entry_requirements": "What a student must have to be admitted to the course.",
    "restrictions": "Who may or may not take the course: majors, standing, consent.",
    "confirmation": "Whether the student is registered, admitted or confirmed.",
    "prerequisites": "Courses or knowledge required before taking this course.",
    "course_content": "The topics, schedule of topics or material covered in class.",
    "learning_outcomes": "What a student will be able to do after the course.",
    "teaching_form": "How the course is taught: in person, online, hybrid.",
    "language_of_instruction": "The language the course is taught in.",
    "materials": "Required textbooks, software, equipment or supplies for the course.",
    "weekly_readings": "Readings assigned for particular weeks or classes.",
    "assignments": "Homework, papers, projects or other graded work.",
    "due_date": "The date something is due or an exam is scheduled.",
    "grading_policy": "How the grade is calculated, including weights, percentages and cut-offs.",
    "grades": "Letter grades, grade scales or grade point values.",
    "assessment": "How students are evaluated: exams, quizzes, participation.",
    "course_evaluation": "How students evaluate the course or instructor, such as an end-of-term survey.",
    "meeting_time": "The clock time a class meets.",
    "meeting_days": "The days of the week a class meets.",
    "class_meeting_room": "The room or building where a class meets.",
    "course_modality": "Whether attendance is required or the course is remote.",
    "course_schedule": "A calendar of sessions, topics or events in the course.",
    "exam_schedule": "When exams, midterms or the final exam take place, including dates and durations.",
    "session_dates": "The start and end dates of the term or session.",
    "term": "The academic term or semester, such as Fall 2024.",
    "drop_add_deadline": "The last day to drop or add the course.",
    "attendance_policy": "The rules about attending class and the consequences of absence, such as excused absences or a participation requirement.",
    "academic_integrity": "The rules about cheating, plagiarism and collaboration.",
    "course_policies": "Other course rules: late work, electronics, conduct.",
    "disability_services": "Accommodations for a disability and the office that arranges them, such as extended time on exams.",
    "additional_information": "Other administrative notes about the course.",
    "other": "Text that is not any academic field: navigation, ads, page furniture.",
}

# TinyDecide's dimensions, so the comparison is honest at the architecture level.
DEFAULT_CONFIG = {
    "vocab_size": 16000,
    "embed_dim": 128,
    "hidden": 256,
    "layers": 12,
    "heads": 4,
    "ffn": 768,
    "max_positions": 512,
    "max_state_tokens": 127,
    "max_option_tokens": 24,
    "dropout": 0.1,
}

PAD, CLS, SEP, UNK, MASK = "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"


@dataclass
class JevConfig:
    vocab_size: int = 16000
    embed_dim: int = 128
    hidden: int = 256
    layers: int = 12
    heads: int = 4
    ffn: int = 768
    max_positions: int = 512
    max_state_tokens: int = 127
    max_option_tokens: int = 24
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JevConfig":
        known = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        return cls(**known)


class TransformerEncoder(nn.Module):
    """Pre-LN encoder at ELECTRA-small width, with a shared token embedding."""

    def __init__(self, config: JevConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.position = nn.Embedding(config.max_positions, config.embed_dim)
        self.embed_norm = nn.LayerNorm(config.embed_dim)
        self.project = nn.Linear(config.embed_dim, config.hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden, nhead=config.heads, dim_feedforward=config.ffn,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        embedded = self.embed(input_ids) + self.position(positions)
        hidden = self.project(self.embed_norm(embedded))
        pad = attention_mask == 0
        hidden = self.blocks(hidden, src_key_padding_mask=pad)
        return self.norm(hidden)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class JevChoiceModel(nn.Module):
    """Encode a state and score it against a menu of described options."""

    def __init__(self, config: JevConfig, labels: list[str],
                 descriptions: dict[str, str] | None = None):
        super().__init__()
        self.config = config
        self.labels = list(labels)
        if descriptions is None:
            descriptions = FIELD_DESCRIPTIONS
        if isinstance(descriptions, dict):
            self.descriptions = [descriptions.get(label, label.replace("_", " "))
                                 for label in self.labels]
        else:
            # A checkpoint stores the rendered descriptions, already in label order.
            self.descriptions = list(descriptions)
        self.encoder = TransformerEncoder(config)
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
        )
        self.option_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden), nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
        )
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.option_bias = nn.Parameter(torch.zeros(len(self.labels)))

    def encode_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     state_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        if state_mask is None:
            state_mask = attention_mask
        return nn.functional.normalize(self.state_head(mean_pool(hidden, state_mask)), dim=-1)

    def encode_options(self, option_ids: torch.Tensor,
                       option_mask: torch.Tensor) -> torch.Tensor:
        """Score descriptions from their own embedding bag (no transformer pass)."""
        weights = option_mask.unsqueeze(-1).to(torch.float32)
        embedded = self.encoder.embed(option_ids)
        pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return nn.functional.normalize(self.option_head(pooled), dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                state_mask: torch.Tensor | None = None,
                option_ids: torch.Tensor | None = None,
                option_mask: torch.Tensor | None = None) -> torch.Tensor:
        state = self.encode_state(input_ids, attention_mask, state_mask)
        if option_ids is None or option_mask is None:
            raise ValueError("option_ids and option_mask are required")
        options = self.encode_options(option_ids, option_mask)
        scale = self.log_scale.exp().clamp(1.0, 50.0)
        logits = scale * (state @ options.transpose(0, 1))
        if self.option_bias.numel() == options.size(0):
            # The bias is a per-trained-label calibration term. A caller's own
            # menu has no such history, so it is simply not added there.
            logits = logits + self.option_bias
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_jev(model: JevChoiceModel, tokenizer, output: str | Path,
             temperature: float = 1.0, extra: dict[str, Any] | None = None) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"))
    checkpoint = {
        "config": asdict(model.config),
        "labels": model.labels,
        "descriptions": model.descriptions,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "temperature": float(temperature),
        "parameter_count": model.parameter_count(),
        **(extra or {}),
    }
    torch.save(checkpoint, output / "model.pt")
    (output / "config.json").write_text(
        json.dumps({"config": asdict(model.config), "labels": model.labels,
                    "descriptions": model.descriptions, "temperature": temperature},
                   indent=2) + "\n", encoding="utf-8")


def load_jev(path: str | Path, device: str | torch.device = "cpu"):
    from tokenizers import Tokenizer

    path = Path(path)
    checkpoint = torch.load(path / "model.pt" if path.is_dir() else path,
                            map_location="cpu", weights_only=False)
    config = JevConfig.from_dict(checkpoint["config"])
    model = JevChoiceModel(config, checkpoint["labels"], checkpoint["descriptions"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    return model, checkpoint, tokenizer


class JevEngine:
    """Serve a trained Jev choice model on arbitrary text."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu",
                 question: str = DEFAULT_QUESTION):
        self.model, self.checkpoint, self.tokenizer = load_jev(checkpoint, device)
        self.device = torch.device(device)
        self.labels: list[str] = list(self.model.labels)
        self.label_tasks = {label: FIELD_TASKS.get(label, "other") for label in self.labels}
        self.question = self.checkpoint.get("question", question)
        self.temperature = float(self.checkpoint.get("temperature", 1.0))
        self.max_state = int(self.model.config.max_state_tokens)
        self.option_ids, self.option_mask = self._pack_options(self.model.descriptions)

    def _pack_options(self, descriptions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise option descriptions into one padded (options, tokens) batch."""
        encoded = self.tokenizer.encode_batch(list(descriptions))
        width = min(max(len(item.ids) for item in encoded),
                    self.model.config.max_option_tokens)
        option_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        option_mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for index, item in enumerate(encoded):
            ids = item.ids[:width]
            option_ids[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            option_mask[index, :len(ids)] = 1
        return option_ids.to(self.device), option_mask.to(self.device)

    def _encode(self, texts: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed, state_masks = [], []
        sep = self.tokenizer.token_to_id(SEP)
        cls = self.tokenizer.token_to_id(CLS)
        pad = self.tokenizer.token_to_id(PAD)
        if sep is None or cls is None or pad is None:
            raise ValueError("tokenizer is missing [CLS]/[SEP]/[PAD]")
        question_ids = self.tokenizer.encode(self.question).ids
        for text in texts:
            state = self.tokenizer.encode((text or " ").strip().lower()).ids[: self.max_state]
            ids = [cls, *state, sep, *question_ids, sep]
            packed.append(ids)
            state_masks.append([0] + [1] * len(state) + [0] * (len(ids) - 1 - len(state)))
        width = max(len(ids) for ids in packed)
        input_ids = torch.full((len(packed), width), pad, dtype=torch.long)
        attention = torch.zeros((len(packed), width), dtype=torch.long)
        state_mask = torch.zeros((len(packed), width), dtype=torch.long)
        for row, ids in enumerate(packed):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, :len(ids)] = 1
            state_mask[row, :len(ids)] = torch.tensor(state_masks[row], dtype=torch.long)
        return input_ids.to(self.device), attention.to(self.device), state_mask.to(self.device)

    @torch.inference_mode()
    def probabilities(self, texts: Iterable[str]) -> list[dict[str, float]]:
        items = [text if (text or "").strip() else " " for text in texts]
        if not items:
            return []
        input_ids, attention, state_mask = self._encode(items)
        logits = self.model(input_ids, attention, state_mask, self.option_ids, self.option_mask)
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return [dict(zip(self.labels, row)) for row in probs.cpu().tolist()]

    @torch.inference_mode()
    def score_against(self, texts: Iterable[str],
                      descriptions: dict[str, str] | list[str]) -> list[dict[str, float]]:
        """Score text against a menu the caller writes, not the one trained on.

        This is the whole point of describing the options instead of learning one
        output neuron per field: a caller can ask for a criterion the model never
        saw as a class -- "this value is a thesis submission deadline", say --
        and the text tower scores it from its description.
        """
        pairs = list(descriptions.items()) if isinstance(descriptions, dict)             else [(text, text) for text in descriptions]
        labels = [label for label, _ in pairs]
        option_ids, option_mask = self._pack_options([text for _, text in pairs])
        items = [text if (text or "").strip() else " " for text in texts]
        if not items:
            return []
        input_ids, attention, state_mask = self._encode(items)
        logits = self.model(input_ids, attention, state_mask, option_ids, option_mask)
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return [dict(zip(labels, row)) for row in probs.cpu().tolist()]

    def classify(self, text: str, top_k: int = 5) -> dict[str, Any]:
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
            "probabilities": dict(ranked),
            "model": "jev-small",
            "temperature": self.temperature,
        }

    def classify_many(self, texts: Iterable[str], top_k: int = 5) -> list[dict[str, Any]]:
        items = list(texts)
        rows = self.probabilities(items)
        out = []
        for text, probs in zip(items, rows):
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
