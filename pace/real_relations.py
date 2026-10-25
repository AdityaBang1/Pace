"""Grounded course-mention relation classification and structured parsing on real USC catalog rows."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

from .data import document_hash, save_jsonl
from .model import BASE_MODEL, MAX_LENGTH, PaceClassifier
from .real_data import split_real_groups

COURSE_MENTION_RE = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{2,4}[A-Za-z]?)\b")
UNITS_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*(?:-|to)\s*(\d+(?:\.\d+)?))?\s*units?(?:\s*,\s*max\s*(\d+(?:\.\d+)?))?",
    re.I,
)
TIME_RANGE_RE = re.compile(
    r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*(am|pm)",
    re.I,
)
DAY_TOKEN_RE = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun|MWF|TTh|MW|TuTh)\b", re.I)

RELATION_LABELS = [
    "corequisite_of",
    "duplicate_credit_of",
    "prerequisite_of",
    "recommended_for",
    "related_reference",
]

HISTORICAL_SUBJECTS = {"GEOG", "MDA", "SWMS", "MDES", "CLAS", "LING"}


def canonical_course_code(subject: str, number: str) -> str:
    return f"{subject.upper()} {number.upper()}"


def base_course_code(subject: str, number: str) -> str:
    digits = re.match(r"\d+", number)
    return f"{subject.upper()} {digits.group(0) if digits else number.upper()}"


def collect_valid_subjects(rows: list[dict[str, Any]]) -> set[str]:
    subjects = set(HISTORICAL_SUBJECTS)
    for row in rows:
        prog = (row.get("Program Code") or "").strip().upper()
        if prog:
            subjects.add(prog)
        cid = (row.get("Class ID") or "").strip()
        match = COURSE_MENTION_RE.search(cid)
        if match:
            subjects.add(match.group(1).upper())
    return subjects


def _clause_around(text: str, start: int, end: int) -> str:
    left_dot = text.rfind(".", 0, start)
    left_semi = text.rfind(";", 0, start)
    left = max(0, max(left_dot, left_semi) + 1)
    right_candidates = [idx for idx in (text.find(".", end), text.find(";", end)) if idx != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right].strip()


def _marked_clause(text: str, start: int, end: int, span_text: str) -> tuple[str, str]:
    left_dot = text.rfind(".", 0, start)
    left_semi = text.rfind(";", 0, start)
    left = max(0, max(left_dot, left_semi) + 1)
    right_candidates = [idx for idx in (text.find(".", end), text.find(";", end)) if idx != -1]
    right = min(right_candidates) if right_candidates else len(text)
    clause = text[left:right].strip()
    marked_clause = f"{text[left:start]}[TARGET] {span_text} [/TARGET]{text[end:right]}".strip()
    return clause, marked_clause


def _infer_relation_label(clause: str, field_name: str, mention_base: str, prereq_bases: set[str]) -> str:
    low = clause.casefold()
    if any(k in low for k in ("corequisite", "co-requisite", "concurrent enrollment", "concurrent:")):
        return "corequisite_of"
    if any(
        k in low
        for k in (
            "duplicates credit",
            "duplicate credit",
            "cross-listed",
            "same as",
            "not open to students with credit",
            "may not register",
        )
    ):
        return "duplicate_credit_of"
    if "recommended preparation" in low:
        return "recommended_for"
    if (
        field_name == "Prerequisites"
        or any(
            k in low
            for k in (
                "prerequisite",
                "qualifying courses",
                "successfully completed",
                "continuation of",
                "completion of",
            )
        )
        or mention_base in prereq_bases
    ):
        return "prerequisite_of"
    return "related_reference"


def load_usc_relation_examples(path: str | Path) -> list[dict[str, Any]]:
    """Build grounded course-mention relation examples from real USC course records."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    valid_subjects = collect_valid_subjects(raw_rows)
    seen: set[tuple[str, str, str, int, int]] = set()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        cid = (row.get("Class ID") or "").strip().rstrip(":")
        self_match = COURSE_MENTION_RE.search(cid)
        if not self_match:
            continue
        self_subj = self_match.group(1).upper()
        self_code = canonical_course_code(self_subj, self_match.group(2))
        prereq_raw = (row.get("Prerequisites") or "").strip()
        prereq_bases = {
            base_course_code(m.group(1), m.group(2))
            for m in COURSE_MENTION_RE.finditer(prereq_raw)
            if m.group(1).upper() in valid_subjects
        }
        for field_name in ("Catalogue", "Prerequisites", "Restrictions"):
            raw_text = (row.get(field_name) or "").strip()
            if not raw_text or raw_text.casefold() in {"not available", "n/a", "none", "null", "nan"}:
                continue
            clean_text = re.sub(r"^\s*(?:Prerequisite|Restriction)s?:\s*", "", raw_text, flags=re.I).strip()
            for match in COURSE_MENTION_RE.finditer(clean_text):
                m_subj, m_num = match.group(1).upper(), match.group(2).upper()
                if m_subj not in valid_subjects:
                    continue
                target_code = canonical_course_code(m_subj, m_num)
                if target_code == self_code:
                    continue
                dedup_key = (self_code, field_name, clean_text, match.start(), match.end())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                clause, marked_text = _marked_clause(clean_text, match.start(), match.end(), match.group(0))
                label = _infer_relation_label(
                    clause=clause,
                    field_name=field_name,
                    mention_base=base_course_code(m_subj, m_num),
                    prereq_bases=prereq_bases,
                )
                examples.append(
                    {
                        "text": marked_text,
                        "unmarked_text": clean_text,
                        "clause": clause,
                        "source_course": self_code,
                        "target_course": target_code,
                        "target_span": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "task": "course_relation",
                        "label": label,
                        "group_id": f"usc-subject:{self_subj}",
                        "source_id": f"usc-2024-spring:{index}:{self_code}",
                        "source_field": field_name,
                        "source": "USC/USC-Course-Catalog",
                        "text_sha256": document_hash(marked_text),
                    }
                )
    return examples


def parse_structured_value(label: str, text: str) -> dict[str, Any]:
    """Deterministically parse structured attributes from classified catalog cell text."""
    stripped = text.strip()
    if label == "credit_units":
        match = UNITS_RANGE_RE.search(stripped)
        if match:
            min_u = float(match.group(1))
            max_u = float(match.group(2)) if match.group(2) else min_u
            repeatable_max = float(match.group(3)) if match.group(3) else None
            return {
                "min_units": min_u,
                "max_units": max_u,
                "repeatable_max_units": repeatable_max,
            }
    elif label == "meeting_time":
        match = TIME_RANGE_RE.search(stripped)
        if match:
            return {
                "start_time": match.group(1),
                "end_time": match.group(2),
                "meridiem": match.group(3).lower(),
            }
    elif label == "meeting_days":
        days = [m.group(0) for m in DAY_TOKEN_RE.finditer(stripped)]
        if days:
            return {"days": days}
    return {}


class RelationDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, labels: list[str], max_length: int):
        self.y = torch.tensor([labels.index(r["label"]) for r in rows], dtype=torch.long)
        self.x = tokenizer(
            [r["text"] for r in rows],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.x["input_ids"][i],
            "attention_mask": self.x["attention_mask"][i],
            "labels": self.y[i],
        }


def _predict_logits(model: PaceClassifier, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    logits_list = []
    ys: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float().cpu()
            logits_list.append(out)
            ys.extend(batch["labels"].tolist())
    return torch.cat(logits_list), np.array(ys)


class RealRelationEngine:
    """Offline course-mention relation extractor backed by the fine-tuned MiniLM relation checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path = "artifacts/usc-relations/model.pt",
        device: str | None = None,
    ):
        self.path = Path(checkpoint)
        self.meta = torch.load(self.path, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PaceClassifier(
            self.meta["base_model"],
            config_dir=str(self.path.parent / self.meta["encoder_config"]),
            num_classes=len(self.meta["labels"]),
        )
        self.model.load_state_dict(self.meta["state_dict"])
        self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path.parent / self.meta["tokenizer_dir"]),
            local_files_only=True,
        )
        self.labels: list[str] = list(self.meta["labels"])
        self.max_length: int = int(self.meta.get("max_length", MAX_LENGTH))
        self.temperature: float = float(self.meta.get("temperature", 1.0))
        self.valid_subjects: set[str] = set(self.meta.get("valid_subjects", []))

    @torch.inference_mode()
    def extract_relations(self, text: str, self_course: str | None = None) -> list[dict[str, Any]]:
        left_trimmed = text.lstrip()
        offset_base = len(text) - len(left_trimmed)
        clean_text = left_trimmed
        prefix = re.match(r"(?:Prerequisite|Restriction)s?:\s*", clean_text, flags=re.I)
        if prefix:
            clean_text = clean_text[prefix.end():]
            offset_base += prefix.end()
        before_strip = clean_text
        clean_text = clean_text.strip()
        offset_base += len(before_strip) - len(before_strip.lstrip())
        if not clean_text:
            return []
        self_norm = None
        if self_course:
            m_self = COURSE_MENTION_RE.search(self_course)
            if m_self:
                self_norm = canonical_course_code(m_self.group(1), m_self.group(2))
        mentions = []
        for match in COURSE_MENTION_RE.finditer(clean_text):
            subj, num = match.group(1).upper(), match.group(2).upper()
            if self.valid_subjects and subj not in self.valid_subjects:
                continue
            target_code = canonical_course_code(subj, num)
            if self_norm and target_code == self_norm:
                continue
            clause, marked = _marked_clause(clean_text, match.start(), match.end(), match.group(0))
            mentions.append((match, target_code, marked, clause))
        if not mentions:
            return []
        encoded = self.tokenizer(
            [m[2] for m in mentions],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        logits = self.model(encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device))
        probs = torch.softmax(logits / max(self.temperature, 0.05), dim=-1).cpu()
        results = []
        for idx, (match, target_code, _, clause) in enumerate(mentions):
            score, pred_id = probs[idx].max(dim=-1)
            label = self.labels[int(pred_id)]
            results.append(
                {
                    "target_course": target_code,
                    "target_span": match.group(0),
                    "start": offset_base + match.start(),
                    "end": offset_base + match.end(),
                    "relation": label,
                    "confidence": float(score),
                    "confidence_semantics": "temperature_scaled_class_score_against_weakly_supervised_usc_relation_labels",
                    "confidence_calibrated": False,
                    "probabilities_calibrated": False,
                    "calibration_scope": "temperature_fit_on_usc_relation_validation_heuristic_labels_only",
                    "calibration_temperature": self.temperature,
                    "model_evidence": "usc_course_catalog_heuristic_relation_labels",
                    "source_text": text,
                    "evidence_clause": clause,
                    "proposed_only": True,
                }
            )
        return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train and evaluate real USC course-mention relation classifiers.")
    p.add_argument("--input", default="data/raw/usc_course_data.csv")
    p.add_argument("--output", default="artifacts/usc-relations")
    p.add_argument("--split-assignment", default="artifacts/usc-real/split_assignment.json")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--seed", type=int, default=20261019)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with Path(args.input).open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    valid_subjects = sorted(collect_valid_subjects(raw_rows))

    examples = load_usc_relation_examples(args.input)
    fixed_assignment = json.loads(Path(args.split_assignment).read_text(encoding="utf-8"))
    # Filter to groups present in examples
    present_groups = {r["group_id"] for r in examples}
    subset_assignment = {g: fixed_assignment[g] for g in present_groups if g in fixed_assignment}
    splits = split_real_groups(examples, seed=args.seed, fixed_assignment=subset_assignment)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        save_jsonl(out / f"{name}.jsonl", part)

    classes = sorted({r["label"] for r in examples})

    # 1. Sparse character + word TF-IDF baseline on marked mention text
    t0 = time.perf_counter()
    sparse_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", random_state=args.seed)),
        ]
    )
    sparse_pipe.fit([r["text"] for r in splits["train"]], [r["label"] for r in splits["train"]])
    sparse_fit_sec = time.perf_counter() - t0
    sparse_preds = sparse_pipe.predict([r["text"] for r in splits["test"]]).tolist()
    sparse_true = [r["label"] for r in splits["test"]]
    joblib.dump(sparse_pipe, out / "relation_tfidf.joblib")

    # Also evaluate Catalogue-only test subset where field prefix alone cannot tell the relation
    cat_test_indices = [i for i, r in enumerate(splits["test"]) if r["source_field"] == "Catalogue"]
    cat_true = [sparse_true[i] for i in cat_test_indices]
    cat_sparse_preds = [sparse_preds[i] for i in cat_test_indices]

    # 2. Fine-tuned MiniLM target-span encoder
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(out / "tokenizer")
    AutoConfig.from_pretrained(args.base_model).save_pretrained(out / "encoder_config")

    ds = {name: RelationDataset(part, tok, classes, args.max_length) for name, part in splits.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        name: DataLoader(d, batch_size=args.batch_size, shuffle=(name == "train"), num_workers=0)
        for name, d in ds.items()
    }

    # Class-balanced weights for loss so minority relations (corequisite_of, related_reference) are learned well
    train_counts = Counter(r["label"] for r in splits["train"])
    weights = torch.tensor(
        [len(splits["train"]) / (len(classes) * max(train_counts[c], 1)) for c in classes],
        dtype=torch.float32,
        device=device,
    ).clamp(max=30.0)

    model = PaceClassifier(args.base_model, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    best_vf = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        losses = []
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            z = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = torch.nn.functional.cross_entropy(z, batch["labels"].to(device), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        vz, vy = _predict_logits(model, loaders["validation"], device)
        vp = vz.argmax(-1).numpy()
        vf = float(f1_score(vy, vp, average="macro", zero_division=0))
        va = float(accuracy_score(vy, vp))
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_accuracy": va,
            "validation_macro_f1": vf,
            "seconds": time.perf_counter() - t_ep,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if vf >= best_vf:
            best_vf = vf
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    vz, vy = _predict_logits(model, loaders["validation"], device)
    tz, ty = _predict_logits(model, loaders["test"], device)

    # Validation temperature scaling
    temperature = torch.nn.Parameter(torch.ones(()))
    temp_opt = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def _cal():
        temp_opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(vz / temperature.clamp(0.05, 20.0), torch.tensor(vy, dtype=torch.long))
        loss.backward()
        return loss

    temp_opt.step(_cal)
    temp = float(temperature.detach().clamp(0.05, 20.0))

    pred_ids = tz.argmax(-1).numpy()
    pred_labels = [classes[int(i)] for i in pred_ids]
    cat_neural_preds = [pred_labels[i] for i in cat_test_indices]

    ckpt = {
        "state_dict": best_state,
        "base_model": args.base_model,
        "num_classes": len(classes),
        "encoder_config": "encoder_config",
        "tokenizer_dir": "tokenizer",
        "labels": classes,
        "max_length": args.max_length,
        "seed": args.seed,
        "temperature": temp,
        "valid_subjects": valid_subjects,
        "selected_epoch": max(history, key=lambda x: x["validation_macro_f1"])["epoch"],
    }
    torch.save(ckpt, out / "model.pt")

    metrics = {
        "task": "grounded course-mention relation classification",
        "dataset": "USC/USC-Course-Catalog",
        "source_group": "course subject prefixes; exact shared split with usc-real",
        "total_examples": len(examples),
        "rows_by_split": {k: len(v) for k, v in splits.items()},
        "subjects_by_split": {k: len({r["group_id"] for r in v}) for k, v in splits.items()},
        "labels": classes,
        "label_counts": dict(Counter(r["label"] for r in examples)),
        "field_counts": dict(Counter(r["source_field"] for r in examples)),
        "sparse_baseline": {
            "model": "char_wb 3-5 gram TF-IDF + balanced logistic regression",
            "fit_seconds": sparse_fit_sec,
            "serialized_bytes": (out / "relation_tfidf.joblib").stat().st_size,
            "test_accuracy": float(accuracy_score(sparse_true, sparse_preds)),
            "test_macro_f1": float(f1_score(sparse_true, sparse_preds, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(sparse_true, sparse_preds, average="weighted", zero_division=0)),
            "test_errors": int(sum(a != b for a, b in zip(sparse_true, sparse_preds))),
            "catalogue_only_test_rows": len(cat_test_indices),
            "catalogue_only_accuracy": float(accuracy_score(cat_true, cat_sparse_preds)),
            "catalogue_only_macro_f1": float(f1_score(cat_true, cat_sparse_preds, average="macro", zero_division=0)),
            "catalogue_only_errors": int(sum(a != b for a, b in zip(cat_true, cat_sparse_preds))),
        },
        "minilm_relation_encoder": {
            "model": args.base_model,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes": (out / "model.pt").stat().st_size,
            "validation_temperature": temp,
            "test_accuracy": float(accuracy_score(sparse_true, pred_labels)),
            "test_macro_f1": float(f1_score(sparse_true, pred_labels, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(sparse_true, pred_labels, average="weighted", zero_division=0)),
            "test_errors": int(sum(a != b for a, b in zip(sparse_true, pred_labels))),
            "catalogue_only_test_rows": len(cat_test_indices),
            "catalogue_only_accuracy": float(accuracy_score(cat_true, cat_neural_preds)),
            "catalogue_only_macro_f1": float(f1_score(cat_true, cat_neural_preds, average="macro", zero_division=0)),
            "catalogue_only_errors": int(sum(a != b for a, b in zip(cat_true, cat_neural_preds))),
            "classification_report": classification_report(
                sparse_true, pred_labels, labels=classes, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(sparse_true, pred_labels, labels=classes).tolist(),
            "test_error_examples": [
                {
                    "true": sparse_true[i],
                    "predicted": pred_labels[i],
                    "source_course": splits["test"][i]["source_course"],
                    "target_course": splits["test"][i]["target_course"],
                    "source_field": splits["test"][i]["source_field"],
                    "clause": splits["test"][i]["clause"],
                }
                for i in range(len(sparse_true))
                if sparse_true[i] != pred_labels[i]
            ],
            "history": history,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "minilm_relation_encoder"}, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics["minilm_relation_encoder"].items()
                if k not in {"classification_report", "confusion_matrix"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
