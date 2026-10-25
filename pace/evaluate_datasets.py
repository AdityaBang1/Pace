"""Small task-separated evaluations over CU-DocBench and synthetic PerQA samples.

This is not a schedule-scanner benchmark: it neither trains a model nor changes inference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

CU_DATASET = "iristun/CU-DocBench"
CU_REVISION = "c00e12b958ad01c0c6a5ae10d8f407160fbd0c20"
CU_INPUT_SHA256 = "657965beddc424193cde29c0463aedb5146478db21c18a61021856d0815c1873"
CU_LAYOUT_SYSTEM = "gemini-3-pro-preview"
PERQA_DATASET = "pchristm/PerQA"
PERQA_REVISION = "ff40fdce3875a6fc262e04d2cdd5de610b2211a5"
PERQA_SAMPLE_SHA256 = "c564a5f5751f2cb88b5a94534f71f4e50096fff3e184896aac7dd9b22e4de151"
PERQA_SAMPLE_SIZE = 20
PERQA_CANONICAL_FETCH_SIZE = 20


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_cu_layout(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a pinned CU-DocBench prediction included with the dataset against gold block types."""
    if not data:
        raise ValueError("CU-DocBench layout evaluation requires at least one document")

    gold_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    gold_blocks = predicted_blocks = matched_blocks = type_correct = 0
    missing_blocks = extra_blocks = error_pages = pages_with_predictions = 0

    for document in data:
        predictions = document.get("predictions", {}).get(CU_LAYOUT_SYSTEM, {})
        predicted_pages = {}
        for page in predictions.get("pages", []):
            page_number = page.get("page_number")
            if page_number in predicted_pages:
                raise ValueError(f"Duplicate predicted page number {page_number}")
            predicted_pages[page_number] = page
        gold_pages = document.get("ground_truth", {}).get("pages", [])
        seen_gold_pages: set[Any] = set()
        for gold_page in gold_pages:
            page_number = gold_page.get("page_number")
            if page_number in seen_gold_pages:
                raise ValueError(f"Duplicate gold page number {page_number}")
            seen_gold_pages.add(page_number)
            gold_list = gold_page.get("blocks", [])
            gold = {block.get("id"): block for block in gold_list}
            if any(block.get("id") is None for block in gold_list):
                raise ValueError(f"Missing gold block ID on page {page_number}")
            if len(gold) != len(gold_list):
                raise ValueError(f"Duplicate gold block IDs on page {page_number}")
            gold_blocks += len(gold)
            for block in gold.values():
                gold_counts[str(block.get("type", "unknown"))] += 1

            prediction = predicted_pages.get(page_number)
            if not prediction or prediction.get("is_error"):
                error_pages += 1
                missing_blocks += len(gold)
                for block_id, block in gold.items():
                    confusion[(str(block.get("type", "unknown")), "missing")] += 1
                continue

            pages_with_predictions += 1
            predicted_list = prediction.get("blocks", [])
            predicted = {block.get("id"): block for block in predicted_list}
            if any(block.get("id") is None for block in predicted_list):
                raise ValueError(f"Missing predicted block ID on page {page_number}")
            if len(predicted) != len(predicted_list):
                raise ValueError(f"Duplicate predicted block IDs on page {page_number}")
            predicted_blocks += len(predicted)
            for block in predicted.values():
                predicted_counts[str(block.get("type", "unknown"))] += 1

            for block_id, gold_block in gold.items():
                predicted_block = predicted.get(block_id)
                if predicted_block is None:
                    missing_blocks += 1
                    confusion[(str(gold_block.get("type", "unknown")), "missing")] += 1
                    continue
                matched_blocks += 1
                gold_type = str(gold_block.get("type", "unknown"))
                predicted_type = str(predicted_block.get("type", "unknown"))
                confusion[(gold_type, predicted_type)] += 1
                if gold_type == predicted_type:
                    type_correct += 1
                    true_positive[gold_type] += 1
            extra_blocks += len(predicted.keys() - gold.keys())
        if set(predicted_pages) - seen_gold_pages:
            raise ValueError("CU-DocBench predictions contain pages without corresponding gold pages")

    labels = sorted(set(gold_counts) | set(predicted_counts))
    per_type = {}
    for label in labels:
        tp = true_positive[label]
        fp = predicted_counts[label] - tp
        fn = gold_counts[label] - tp
        per_type[label] = {
            "support": gold_counts[label],
            "predicted": predicted_counts[label],
            "true_positive": tp,
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        }

    if gold_blocks == 0:
        raise ValueError("CU-DocBench layout evaluation requires at least one gold block")

    false_positives = predicted_blocks - type_correct
    false_negatives = gold_blocks - type_correct
    return {
        "task": "document_block_layout_type_classification",
        "dataset": CU_DATASET,
        "dataset_revision": CU_REVISION,
        "license": "MIT declared by dataset card; original source-document rights may differ",
        "prediction_source": f"precomputed dataset prediction: {CU_LAYOUT_SYSTEM}",
        "protocol": "match page-local block IDs; missing predictions count as false negatives; extra and mistyped blocks count as false positives",
        "documents": len(data),
        "gold_pages": sum(len(doc.get("ground_truth", {}).get("pages", [])) for doc in data),
        "pages_with_predictions": pages_with_predictions,
        "prediction_error_pages": error_pages,
        "gold_blocks": gold_blocks,
        "predicted_blocks": predicted_blocks,
        "matched_blocks": matched_blocks,
        "missing_blocks": missing_blocks,
        "extra_blocks": extra_blocks,
        "type_correct": type_correct,
        "coverage": _safe_ratio(matched_blocks, gold_blocks),
        "type_accuracy_over_gold": _safe_ratio(type_correct, gold_blocks),
        "micro_precision": _safe_ratio(type_correct, type_correct + false_positives),
        "micro_recall": _safe_ratio(type_correct, type_correct + false_negatives),
        "micro_f1": _safe_ratio(2 * type_correct, 2 * type_correct + false_positives + false_negatives),
        "per_type": per_type,
        "confusion": {
            gold_type: {
                predicted_type: count
                for (row_type, predicted_type), count in sorted(confusion.items())
                if row_type == gold_type
            }
            for gold_type in sorted(gold_counts)
        },
        "scope_limit": "Evaluates supplied VLM layout predictions, not Pace schedule extraction or a model trained by this command.",
    }


def _parse_json_column(value: Any, *, row_id: Any, field: str, expected: type) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"PerQA row {row_id!r} has non-string {field}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"PerQA row {row_id!r} has invalid JSON in {field}") from exc
    if not isinstance(parsed, expected):
        raise ValueError(f"PerQA row {row_id!r} has unexpected {field} JSON type")
    return parsed


def _valid_temporal_value(field: str, value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    try:
        if field.endswith("_date"):
            dt.date.fromisoformat(value)
        else:
            dt.time.fromisoformat(value)
    except ValueError:
        return False
    return True


def evaluate_perqa_structured_events(
    observable_rows: list[dict[str, Any]],
    canonicalized_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Smoke-check structured event JSON and temporal fields on a bounded PerQA test sample."""
    canonical = {row.get("id"): row for row in canonicalized_rows}
    if len(canonical) != len(canonicalized_rows):
        raise ValueError("PerQA canonicalized sample contains duplicate event IDs")
    observable_ids = [row.get("id") for row in observable_rows]
    if len(set(observable_ids)) != len(observable_ids):
        raise ValueError("PerQA observable sample contains duplicate row IDs")
    if any(event_id is None for event_id in canonical) or any(row_id is None for row_id in observable_ids):
        raise ValueError("PerQA sampled rows must have event IDs")
    if any(row.get("split") != "test" for row in observable_rows + canonicalized_rows):
        raise ValueError("PerQA smoke-test inputs must all belong to the test split")
    if any(row.get("event_source") != "observable" for row in observable_rows):
        raise ValueError("PerQA observable sample contains a non-observable event_source")
    if any(row.get("event_source") != "canonicalized" for row in canonicalized_rows):
        raise ValueError("PerQA canonical sample contains a non-canonicalized event_source")
    if len(observable_rows) != PERQA_SAMPLE_SIZE:
        raise ValueError(f"PerQA smoke test requires exactly {PERQA_SAMPLE_SIZE} observable rows")
    canonical_ids = {row.get("id") for row in canonicalized_rows}
    requested_canonical_ids = {row.get("canonicalized_event_id") for row in observable_rows}
    if requested_canonical_ids - canonical_ids:
        raise ValueError("PerQA canonicalized rows are missing one or more events linked by the sample")
    if canonical_ids - requested_canonical_ids:
        raise ValueError("PerQA canonicalized sample includes events not linked by this observable sample")
    for row in observable_rows:
        linked = canonical[row.get("canonicalized_event_id")]
        # Observable events can be consolidated or retyped during canonicalization;
        # only their stable split/persona linkage is asserted here.
        if not _valid_temporal_value("start_date", row.get("start_date")):
            raise ValueError(f"PerQA row {row.get('id')!r} has an invalid start_date")
        if not _valid_temporal_value("end_date", row.get("end_date")):
            raise ValueError(f"PerQA row {row.get('id')!r} has an invalid end_date")
        if not _valid_temporal_value("start_time", row.get("start_time")):
            raise ValueError(f"PerQA row {row.get('id')!r} has an invalid start_time")
        if not _valid_temporal_value("end_time", row.get("end_time")):
            raise ValueError(f"PerQA row {row.get('id')!r} has an invalid end_time")
        if not _valid_temporal_value("start_date", linked.get("start_date")):
            raise ValueError(f"PerQA canonical event {linked.get('id')!r} has an invalid start_date")
        if not _valid_temporal_value("end_date", linked.get("end_date")):
            raise ValueError(f"PerQA canonical event {linked.get('id')!r} has an invalid end_date")
        if not _valid_temporal_value("start_time", linked.get("start_time")):
            raise ValueError(f"PerQA canonical event {linked.get('id')!r} has an invalid start_time")
        if not _valid_temporal_value("end_time", linked.get("end_time")):
            raise ValueError(f"PerQA canonical event {linked.get('id')!r} has an invalid end_time")

    event_data_valid = properties_valid = 0
    temporal_values = temporal_values_valid = 0
    surfaced_temporal_fields = surfaced_temporal_fields_valid = 0
    matched_event_rows = 0
    event_types: Counter[str] = Counter()
    temporal_fields = ("start_date", "start_time", "end_date", "end_time")

    for row in observable_rows:
        row_id = row.get("id")
        linked = canonical.get(row.get("canonicalized_event_id"))
        if row.get("persona_id") != linked.get("persona_id") or row.get("split") != linked.get("split"):
            raise ValueError(f"PerQA observable row {row_id!r} links across persona or split")
        matched_event_rows += 1
        event_types[str(row.get("event_type", "unknown"))] += 1
        _parse_json_column(row.get("event_data"), row_id=row_id, field="event_data", expected=dict)
        event_data_valid += 1
        properties = _parse_json_column(
            row.get("properties_mentioned"), row_id=row_id,
            field="properties_mentioned", expected=list,
        )
        if any(not isinstance(item, str) for item in properties):
            raise ValueError(f"PerQA row {row_id!r} has non-string properties_mentioned entries")
        properties_valid += 1
        for field in temporal_fields:
            value = row.get(field)
            if value is not None:
                temporal_values += 1
                temporal_values_valid += _valid_temporal_value(field, value)
            if field in properties:
                surfaced_temporal_fields += 1
                surfaced_temporal_fields_valid += value is not None and _valid_temporal_value(field, value)

    return {
        "task": "synthetic_structured_temporal_event_field_smoke_test",
        "dataset": PERQA_DATASET,
        "dataset_revision": PERQA_REVISION,
        "license": "CC-BY-4.0 declared by dataset card; attribute Philipp Christmann and Gerhard Weikum",
        "config": "observable_events + canonicalized_events",
        "split": "test",
        "observable_rows": len(observable_rows),
        "canonicalized_rows_evaluated": len(canonicalized_rows),
        "linked_observable_rows": matched_event_rows,
        "distinct_linked_canonical_events": len({row.get("canonicalized_event_id") for row in observable_rows}),
        "event_data_json_valid": event_data_valid,
        "properties_mentioned_json_valid": properties_valid,
        "synthetic_rows_confirmed": len(observable_rows) + len(canonicalized_rows),
        "non_null_temporal_values": temporal_values,
        "valid_non_null_temporal_values": temporal_values_valid,
        "non_null_temporal_value_valid_rate": _safe_ratio(temporal_values_valid, temporal_values),
        "surface_declared_temporal_fields": surfaced_temporal_fields,
        "valid_surface_declared_temporal_fields": surfaced_temporal_fields_valid,
        "surface_declared_temporal_field_valid_rate": _safe_ratio(
            surfaced_temporal_fields_valid, surfaced_temporal_fields
        ),
        "event_types": dict(sorted(event_types.items())),
        "reference_date": "PerQA questions use 2024-11-25; question SQL is not executed by this smoke test",
        "scope_limit": "Structured synthetic event/schema smoke test only; not natural-language syllabus extraction or schedule-scanner calibration.",
    }



def load_perqa_sample(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the checked-in PerQA sample only when its exact pinned digest matches."""
    content = Path(path).read_bytes()
    actual_digest = hashlib.sha256(content).hexdigest()
    if actual_digest != PERQA_SAMPLE_SHA256:
        raise ValueError(
            "PerQA sample SHA-256 does not match the pinned sample; "
            f"expected {PERQA_SAMPLE_SHA256}, got {actual_digest}"
        )
    sample = json.loads(content.decode("utf-8"))
    if sample.get("dataset") != PERQA_DATASET or sample.get("dataset_revision") != PERQA_REVISION:
        raise ValueError("PerQA sample dataset or revision does not match the pinned evaluation protocol")
    if sample.get("license") != "CC-BY-4.0":
        raise ValueError("PerQA sample license metadata must remain CC-BY-4.0")
    if sample.get("attribution") != ["Philipp Christmann", "Gerhard Weikum"]:
        raise ValueError("PerQA sample attribution metadata is missing or unexpected")
    configs = sample.get("configs", {})
    rows_by_config = {}
    for config, expected_length in (
        ("observable_events", PERQA_SAMPLE_SIZE),
        ("canonicalized_events", PERQA_CANONICAL_FETCH_SIZE),
    ):
        entry = configs.get(config, {})
        rows = entry.get("rows")
        if entry.get("split") != "test" or entry.get("offset") != 0:
            raise ValueError(f"PerQA {config} sample must use offset 0 of the test split")
        if entry.get("requested_length") != expected_length or entry.get("returned_rows") != expected_length:
            raise ValueError(f"PerQA {config} requested/returned length must be {expected_length}")
        if not isinstance(rows, list) or len(rows) != expected_length:
            raise ValueError(f"PerQA {config} sample must contain exactly {expected_length} rows")
        rows_by_config[config] = rows
    return rows_by_config["observable_events"], rows_by_config["canonicalized_events"]


def load_cu_docbench(path: str | Path) -> list[dict[str, Any]]:
    """Load the CU-DocBench snapshot only when its exact pinned content digest matches."""
    content = Path(path).read_bytes()
    actual_digest = hashlib.sha256(content).hexdigest()
    if actual_digest != CU_INPUT_SHA256:
        raise ValueError(
            "CU-DocBench input SHA-256 does not match the pinned dataset snapshot "
            f"({CU_REVISION}); expected {CU_INPUT_SHA256}, got {actual_digest}"
        )
    data = json.loads(content.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("Pinned CU-DocBench input must be a JSON document list")
    return data


def run_evaluation(
    cu_path: str | Path = "data/raw/cu_docbench.json",
    perqa_sample_path: str | Path = "data/samples/perqa_test_sample.json",
) -> dict[str, Any]:
    """Run both task-specific checks entirely offline from pinned local source snapshots."""
    cu_data = load_cu_docbench(cu_path)
    observable_rows, canonical_rows = load_perqa_sample(perqa_sample_path)
    linked_ids = {row.get("canonicalized_event_id") for row in observable_rows}
    canonical_by_id = {row.get("id"): row for row in canonical_rows}
    missing = linked_ids - canonical_by_id.keys()
    if missing:
        raise ValueError(f"PerQA sample does not contain linked canonicalized event IDs: {sorted(missing)}")
    canonical_rows = [canonical_by_id[event_id] for event_id in sorted(linked_ids)]
    return {
        "schema_version": "pace-multidataset-evaluation/0.1",
        "perqa_sample_provenance": {
            "dataset_revision": PERQA_REVISION,
            "sample_sha256": PERQA_SAMPLE_SHA256,
            "observable_events": {"split": "test", "offset": 0, "requested_length": PERQA_SAMPLE_SIZE},
            "canonicalized_events": {
                "split": "test",
                "offset": 0,
                "requested_length": PERQA_CANONICAL_FETCH_SIZE,
                "evaluated_linked_unique_ids_only": True,
            },
        },
        "tasks_are_separate": True,
        "metrics_are_not_aggregated_across_tasks": True,
        "cu_docbench_input_sha256": CU_INPUT_SHA256,
        "cu_docbench_layout": evaluate_cu_layout(cu_data),
        "perqa_structured_temporal_smoke": evaluate_perqa_structured_events(observable_rows, canonical_rows),
        "schedule_scanner": {
            "trained_or_calibrated_on_these_datasets": False,
            "inference_confidence_changed": False,
            "real_syllabus_calibration": "blocked_pending_independent_human_labeled_licensed_task_matched_data",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run separate offline CU-DocBench layout and bounded synthetic PerQA event-field evaluations."
    )
    parser.add_argument("--cu-input", default="data/raw/cu_docbench.json")
    parser.add_argument("--perqa-sample", default="data/samples/perqa_test_sample.json")
    parser.add_argument("--output", default="artifacts/multi-dataset/evaluation.json")
    args = parser.parse_args(argv)
    result = run_evaluation(args.cu_input, args.perqa_sample)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "cu_docbench_layout": {
            key: result["cu_docbench_layout"][key]
            for key in ("gold_blocks", "matched_blocks", "coverage", "type_accuracy_over_gold", "micro_f1")
        },
        "perqa_structured_temporal_smoke": {
            key: result["perqa_structured_temporal_smoke"][key]
            for key in (
                "observable_rows", "linked_observable_rows", "event_data_json_valid",
                "valid_non_null_temporal_values", "non_null_temporal_values",
            )
        },
        "schedule_scanner": result["schedule_scanner"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
