"""Offline field and relation inference for the USC real-catalog models (PyTorch and ONNX INT8)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .extract import extract_candidates
from .model import MAX_LENGTH, PaceClassifier
from .real_relations import RealRelationEngine, parse_structured_value

THAI_SCRIPT_RE = re.compile(r"[\u0E00-\u0E7F]")
HTML_TABLE_RE = re.compile(r"<table\b", re.I)
EXPLICIT_FIELD_PREFIXES = {
    "class name": "course_title",
    "catalogue": "course_description",
    "units": "credit_units",
    "prerequisite": "prerequisites",
    "prerequisites": "prerequisites",
    "restriction": "restrictions",
    "restrictions": "restrictions",
    "time": "meeting_time",
    "days": "meeting_days",
    "class type": "class_type",
}
EXPLICIT_FIELD_PREFIX_RE = re.compile(
    r"^\s*(class\s+name|catalogue|units|prerequisites?|restrictions?|time|days|class\s+type)\s*:",
    re.I,
)


class RealCatalogEngine:
    def __init__(
        self,
        checkpoint: str | Path = "artifacts/usc-miniLM/model.pt",
        device: str | None = None,
        relation_checkpoint: str | Path | None = "artifacts/usc-relations/model.pt",
        onnx_path: str | Path | None = None,
    ):
        self.path = Path(checkpoint)
        self.meta = torch.load(self.path, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.labels: list[str] = list(self.meta["labels"])
        self.max_length: int = int(self.meta.get("max_length", MAX_LENGTH))
        self.temperature: float = float(self.meta.get("temperature", 1.0))
        self.ambiguous_values: dict[str, list[str]] = dict(self.meta.get("ambiguous_values", {}))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path.parent / self.meta["tokenizer_dir"]),
            local_files_only=True,
        )
        self.onnx_session = None
        if onnx_path is not None:
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.model = None
        else:
            self.model = PaceClassifier(
                self.meta["base_model"],
                config_dir=str(self.path.parent / self.meta["encoder_config"]),
                num_classes=len(self.labels),
            )
            self.model.load_state_dict(self.meta["state_dict"])
            self.model.to(self.device).eval()

        self.relation_engine: RealRelationEngine | None = None
        if relation_checkpoint and Path(relation_checkpoint).exists():
            self.relation_engine = RealRelationEngine(relation_checkpoint, device=str(self.device))

    @torch.inference_mode()
    def classify_field(self, text: str, self_course: str | None = None) -> dict[str, Any]:
        from .real_data import normalize_value

        prefix_match = EXPLICIT_FIELD_PREFIX_RE.match(text)
        if prefix_match:
            prefix = re.sub(r"\s+", " ", prefix_match.group(1).strip()).casefold()
            label = EXPLICIT_FIELD_PREFIXES[prefix]
            value_text = text[prefix_match.end():].strip()
            relations = (
                self.relation_engine.extract_relations(text, self_course=self_course)
                if self.relation_engine is not None
                and label in {"course_description", "prerequisites", "restrictions"}
                else []
            )
            return {
                "task": "catalog_field",
                "label": label,
                "proposed_label": label,
                "confidence": 1.0,
                "confidence_semantics": "explicit_source_field_prefix_match_not_model_confidence",
                "confidence_calibrated": False,
                "probabilities_calibrated": False,
                "source_text": text,
                "proposed_only": True,
                "abstained": False,
                "reason": "explicit_source_field_prefix",
                "structured_value": parse_structured_value(label, value_text),
                "course_relations": relations,
            }

        normalized = normalize_value(text)
        if normalized in self.ambiguous_values:
            return {
                "task": "catalog_field",
                "label": "unknown",
                "proposed_label": None,
                "confidence": 0.0,
                "source_text": text,
                "proposed_only": True,
                "abstained": True,
                "reason": "ambiguous_literal_has_multiple_source_field_labels",
                "structured_value": {},
                "course_relations": [],
            }
        if THAI_SCRIPT_RE.search(text) or HTML_TABLE_RE.search(text):
            return {
                "task": "catalog_field",
                "label": "unknown",
                "proposed_label": None,
                "confidence": 0.0,
                "source_text": text,
                "proposed_only": True,
                "abstained": True,
                "reason": "out_of_distribution_script_or_html_table",
                "structured_value": {},
                "course_relations": [],
            }

        if self.onnx_session is not None:
            enc = self.tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            logits_np = self.onnx_session.run(
                ["logits"],
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0][0]
            logits = torch.from_numpy(logits_np)
        else:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            logits = self.model(
                inputs["input_ids"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )[0].cpu()

        probs = torch.softmax(logits / max(self.temperature, 0.05), dim=-1)
        score, idx = probs.max(dim=-1)
        label = self.labels[int(idx)]
        structured = parse_structured_value(label, text)
        relations = (
            self.relation_engine.extract_relations(text, self_course=self_course)
            if self.relation_engine is not None
            and label in {"course_description", "prerequisites", "restrictions"}
            else []
        )
        return {
            "task": "catalog_field",
            "label": label,
            "confidence": float(score),
            "source_text": text,
            "proposed_only": True,
            "abstained": False,
            "confidence_semantics": "temperature_scaled_top_label_probability_for_usc_catalog_cell_classification",
            "confidence_calibrated": True,
            "probabilities_calibrated": True,
            "calibration_scope": "usc_course_catalog_cell_field_classification_with_subject_group_split",
            "calibration_temperature": self.temperature,
            "model_evidence": "usc_course_catalog_subject_held_out_validation",
            "structured_value": structured,
            "course_relations": relations,
        }

    def analyze_course_cells(
        self,
        cells: list[str],
        self_course: str | None = None,
    ) -> dict[str, Any]:
        """Classify a list of raw course cells, parse attributes, and extract grounded course relations."""
        cell_results = [self.classify_field(c, self_course=self_course) for c in cells if c and c.strip()]
        all_relations = [rel for r in cell_results for rel in r.get("course_relations", [])]
        full_text = "\n".join(c for c in cells if c and c.strip())
        return {
            "schema_version": "1.0",
            "self_course": self_course,
            "cells": cell_results,
            "course_relations": all_relations,
            "candidate_entities": extract_candidates(full_text),
            "requires_review": True,
            "commit_policy": "proposals_only_user_confirmation_required",
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run local catalog field & relation inference.")
    p.add_argument("text")
    p.add_argument("--checkpoint", default="artifacts/usc-miniLM/model.pt")
    p.add_argument("--onnx", default=None, help="Optional ONNX model path (e.g., artifacts/usc-miniLM/onnx/model_int8.onnx)")
    p.add_argument("--self-course", default=None)
    a = p.parse_args()
    engine = RealCatalogEngine(a.checkpoint, onnx_path=a.onnx)
    print(json.dumps(engine.classify_field(a.text, self_course=a.self_course), indent=2))
