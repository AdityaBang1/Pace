"""How much does ordinary OCR damage cost the field reader?

The corruption here is character-level and derived from how scanned academic text
actually fails: ``l``/``1`` and ``O``/``0`` confusions, ``rn`` read as ``m``,
``,`` read as ``.``, lost spaces at line ends, duplicated lines and flattened
paragraph breaks. Nothing is dropped from the test set, so the accuracy change
is a real degradation rather than a change of question.

Corruptions are applied per row with a fixed seed, so the same text is corrupted
identically for every checkpoint compared.

    python scripts/measure_ocr_robustness.py --rows data/processed/transfer/unseen-syllabus/test.jsonl \
        --checkpoints artifacts/jev-field-v2 artifacts/jev-field-v3
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

_LOOKALIKES = [("l", "1"), ("O", "0"), ("o", "0"), ("I", "l"), ("S", "5"),
               ("B", "8"), ("rn", "m"), ("cl", "d"), ("ii", "u")]


def _swap(text: str, rng: random.Random, rate: float) -> str:
    out = []
    for char in text:
        if rng.random() < rate:
            for source, target in _LOOKALIKES:
                if char == source:
                    char = target
                    break
        out.append(char)
    return "".join(out)


def _join_words(text: str, rng: random.Random, rate: float) -> str:
    words = text.split()
    out = []
    for word in words:
        if out and rng.random() < rate:
            out[-1] += word
        else:
            out.append(word)
    return " ".join(out)


def _duplicate_lines(text: str, rng: random.Random, rate: float) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        out.append(line)
        if line.strip() and rng.random() < rate:
            out.append(line)
    return "\n".join(out)


def _punctuation(text: str, rng: random.Random, rate: float) -> str:
    return re.sub(r",", lambda _match: "." if rng.random() < rate else ",",
                  re.sub(r"\.", lambda _match: "," if rng.random() < rate else ".", text))


def _flatten(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text)


CORRUPTIONS = {
    "lookalike_chars": lambda text, rng: _swap(text, rng, 0.06),
    "joined_words": lambda text, rng: _join_words(text, rng, 0.25),
    "duplicated_lines": lambda text, rng: _duplicate_lines(text, rng, 0.3),
    "punctuation": lambda text, rng: _punctuation(text, rng, 0.4),
    "flattened_layout": lambda text, rng: _flatten(text),
    "all_combined": lambda text, rng: _flatten(_punctuation(
        _duplicate_lines(_join_words(_swap(text, rng, 0.06), rng, 0.25), rng, 0.3),
        rng, 0.4)),
}


def score(engine, rows, texts) -> dict:
    index = {label: position for position, label in enumerate(engine.labels)}
    kept = [(text, row["label"]) for text, row in zip(texts, rows) if row["label"] in index]
    outputs = engine.classify_many([text for text, _ in kept], top_k=1)
    truth = [label for _, label in kept]
    prediction = [item["label"] for item in outputs]
    return {
        "rows": len(kept),
        "accuracy": float(accuracy_score(truth, prediction)),
        # Exactly pace.jev_eval's definition (sklearn infers the label set from
        # truth and prediction), so these numbers sit in the same table as the
        # benchmark scores rather than being 0.03 off from them.
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Measure OCR robustness")
    parser.add_argument("--rows", default="data/processed/transfer/unseen-syllabus/test.jsonl")
    parser.add_argument("--checkpoints", nargs="+",
                        default=["artifacts/jev-field-v2", "artifacts/jev-field-v3"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", default="artifacts/transfer-eval/ocr_robustness.json")
    args = parser.parse_args(argv)

    from pace.field_train import load_jsonl
    from pace.jev_model import JevEngine

    rows = load_jsonl(args.rows)
    results: dict[str, dict] = {}
    for checkpoint in args.checkpoints:
        engine = JevEngine(checkpoint, device=args.device or "cpu")
        per_condition = {"clean": score(engine, rows, [row["text"] for row in rows])}
        for name, corrupt in CORRUPTIONS.items():
            rng = random.Random(f"{args.seed}:{name}")
            texts = [corrupt(row["text"], rng) for row in rows]
            per_condition[name] = score(engine, rows, texts)
        clean = per_condition["clean"]
        for name, values in per_condition.items():
            values["accuracy_drop"] = round(clean["accuracy"] - values["accuracy"], 4)
            values["macro_f1_drop"] = round(clean["macro_f1"] - values["macro_f1"], 4)
        results[Path(checkpoint).name] = per_condition
        print(json.dumps({Path(checkpoint).name: {name: {"accuracy": values["accuracy"],
                                                         "macro_f1": values["macro_f1"]}
                                                  for name, values in per_condition.items()}},
                         indent=2), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": str(args.rows), "results": results},
                                 indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
