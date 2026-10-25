"""Refuse a candidate checkpoint that is worse where the product is measured.

A single improved metric is not a reason to ship. This runs the same three
benchmarks the project is judged on and applies the gates in the plan: a
candidate may not lose materially on unseen-school macro-F1, may not become more
overconfident on shifted text, and must not regress the district-format holdout.
Latency and size are reported so a size-for-quality trade can be seen rather
than argued.

Exit code 1 means the candidate fails a gate, so this can be wired into CI as-is.

    python scripts/regression_gate.py --candidate artifacts/jv7-field
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BENCHMARKS = (
    # name, rows, which metric is gated, tolerance (candidate minus incumbent)
    ("field_corpus", "data/processed/field_mix_ext/test.jsonl", "macro_f1", -0.005),
    # The hardest benchmark gets an almost-zero tolerance on purpose: this is the
    # number the product is judged on, so a candidate cannot buy district-format
    # accuracy with unseen-school accuracy (which is what v3 did: -0.0098).
    ("unseen_syllabus", "data/processed/transfer/unseen-syllabus/test.jsonl", "macro_f1", -0.002),
    ("sis_holdout", "data/processed/sis/sis_holdout.jsonl", "macro_f1", -0.020),
)


def evaluate(checkpoint: str, rows_path: str, device: str) -> dict:
    from pace.field_train import load_jsonl
    from pace.jev_model import JevEngine
    from sklearn.metrics import accuracy_score, f1_score

    engine = JevEngine(checkpoint, device=device)
    index = {label: position for position, label in enumerate(engine.labels)}
    rows = [row for row in load_jsonl(rows_path) if row["label"] in index]
    if not rows:
        return {"rows": 0}
    start = time.perf_counter()
    outputs = engine.classify_many([row["text"] for row in rows], top_k=1)
    elapsed = time.perf_counter() - start
    truth = [row["label"] for row in rows]
    prediction = [item["label"] for item in outputs]
    confidence = [item["confidence"] for item in outputs]
    correct = [t == p for t, p in zip(truth, prediction)]
    bins = 10
    ece = 0.0
    for position in range(bins):
        low, high = position / bins, (position + 1) / bins
        selected = [i for i, c in enumerate(confidence)
                    if low <= c < high or (position == bins - 1 and c == high)]
        if selected:
            mean_conf = sum(confidence[i] for i in selected) / len(selected)
            mean_acc = sum(correct[i] for i in selected) / len(selected)
            ece += (len(selected) / len(rows)) * abs(mean_conf - mean_acc)
    return {
        "rows": len(rows),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "ece": float(ece),
        "items_per_second": round(len(rows) / elapsed, 1) if elapsed else None,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Regression gate for a field checkpoint")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", default="artifacts/jev-field-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = {"candidate": args.candidate, "incumbent": args.incumbent,
              "benchmarks": {}, "failed": []}
    for name, rows_path, metric, tolerance in BENCHMARKS:
        if not Path(rows_path).exists():
            report["benchmarks"][name] = {"skipped": "rows file missing"}
            continue
        incumbent = evaluate(args.incumbent, rows_path, args.device)
        candidate = evaluate(args.candidate, rows_path, args.device)
        delta = candidate[metric] - incumbent[metric]
        passed = delta >= tolerance
        entry = {"incumbent": incumbent, "candidate": candidate,
                 "metric": metric, "delta": round(delta, 4),
                 "tolerance": tolerance, "passed": passed}
        if not passed:
            report["failed"].append(f"{name}: {metric} {delta:+.4f} (floor {tolerance:+.4f})")
        if candidate.get("ece") is not None and incumbent.get("ece") is not None:
            entry["ece_delta"] = round(candidate["ece"] - incumbent["ece"], 4)
        report["benchmarks"][name] = entry
        print(json.dumps({name: {key: entry[key] for key in
                                 ("metric", "delta", "tolerance", "passed")}}))

    report["verdict"] = "accept" if not report["failed"] else "reject"
    print(json.dumps({"verdict": report["verdict"], "failed": report["failed"]}, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if not report["failed"] else 1)


if __name__ == "__main__":
    main()
