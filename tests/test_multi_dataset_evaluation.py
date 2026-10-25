import copy
import hashlib
import json
from pathlib import Path

import pytest

from pace import evaluate_datasets as evaluation


def test_cu_input_loader_requires_pinned_digest(tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "data" / "raw" / "cu_docbench.json"
    content = source.read_bytes()
    assert hashlib.sha256(content).hexdigest() == evaluation.CU_INPUT_SHA256
    assert len(evaluation.load_cu_docbench(source)) == 23

    wrong_revision = tmp_path / "different-revision.json"
    wrong_revision.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        evaluation.load_cu_docbench(wrong_revision)


def test_cu_layout_rejects_empty_documents_and_empty_gold_blocks():
    with pytest.raises(ValueError, match="at least one document"):
        evaluation.evaluate_cu_layout([])

    empty_gold = {
        "ground_truth": {"pages": [{"page_number": 1, "blocks": []}]},
        "predictions": {evaluation.CU_LAYOUT_SYSTEM: {"pages": [{"page_number": 1, "blocks": []}]}},
    }
    with pytest.raises(ValueError, match="at least one gold block"):
        evaluation.evaluate_cu_layout([empty_gold])


def test_cu_layout_scoring_counts_type_errors_missing_and_extra():
    documents = [
        {
            "metadata": {"file_name": "layout.pdf"},
            "ground_truth": {
                "pages": [
                    {
                        "page_number": 1,
                        "blocks": [
                            {"id": "a", "type": "title"},
                            {"id": "b", "type": "text"},
                            {"id": "c", "type": "image"},
                        ],
                    }
                ]
            },
            "predictions": {
                evaluation.CU_LAYOUT_SYSTEM: {
                    "pages": [
                        {
                            "page_number": 1,
                            "blocks": [
                                {"id": "a", "type": "title"},
                                {"id": "b", "type": "title"},
                                {"id": "extra", "type": "text"},
                            ],
                        }
                    ]
                }
            },
        }
    ]

    result = evaluation.evaluate_cu_layout(documents)

    assert result["task"] == "document_block_layout_type_classification"
    assert result["gold_blocks"] == 3
    assert result["predicted_blocks"] == 3
    assert result["matched_blocks"] == 2
    assert result["missing_blocks"] == 1
    assert result["extra_blocks"] == 1
    assert result["type_correct"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["micro_precision"] == pytest.approx(1 / 3)
    assert result["micro_recall"] == pytest.approx(1 / 3)
    assert result["micro_f1"] == pytest.approx(1 / 3)
    assert result["per_type"]["title"]["support"] == 1
    assert result["per_type"]["title"]["predicted"] == 2
    assert result["per_type"]["text"]["support"] == 1


def test_cu_layout_missing_and_error_pages_count_gold_blocks_as_false_negatives():
    documents = [
        {
            "ground_truth": {
                "pages": [
                    {"page_number": 1, "blocks": [{"id": "missing", "type": "text"}]},
                    {"page_number": 2, "blocks": [{"id": "errored", "type": "title"}]},
                    {"page_number": 3, "blocks": [{"id": "ok", "type": "image"}]},
                ]
            },
            "predictions": {
                evaluation.CU_LAYOUT_SYSTEM: {
                    "pages": [
                        {"page_number": 2, "is_error": True, "blocks": [{"id": "ignored", "type": "text"}]},
                        {"page_number": 3, "blocks": [{"id": "ok", "type": "image"}]},
                    ]
                }
            },
        }
    ]

    result = evaluation.evaluate_cu_layout(documents)

    assert result["gold_pages"] == 3
    assert result["pages_with_predictions"] == 1
    assert result["prediction_error_pages"] == 2
    assert result["gold_blocks"] == 3
    assert result["predicted_blocks"] == 1
    assert result["matched_blocks"] == 1
    assert result["missing_blocks"] == 2
    assert result["extra_blocks"] == 0
    assert result["type_correct"] == 1
    assert result["micro_recall"] == pytest.approx(1 / 3)
    assert result["confusion"]["text"]["missing"] == 1
    assert result["confusion"]["title"]["missing"] == 1


def _perqa_rows():
    canonical = [
        {
            "id": 100,
            "split": "test",
            "persona_id": "test_persona_0",
            "event_source": "canonicalized",
            "start_date": "2024-01-02",
            "start_time": "09:30:00",
            "end_date": None,
            "end_time": None,
        }
    ]
    observable = [
        {
            "id": index,
            "canonicalized_event_id": 100,
            "split": "test",
            "persona_id": "test_persona_0",
            "event_source": "observable",
            "event_type": "meeting",
            "event_data": '{"location":"Room 2"}',
            "properties_mentioned": '["start_date", "start_time"]',
            "start_date": "2024-01-02",
            "start_time": "09:30:00",
            "end_date": None,
            "end_time": None,
        }
        for index in range(evaluation.PERQA_SAMPLE_SIZE)
    ]
    return observable, canonical


def test_perqa_structured_smoke_is_deterministic_and_task_limited():
    observable, canonical = _perqa_rows()

    first = evaluation.evaluate_perqa_structured_events(observable, canonical)
    second = evaluation.evaluate_perqa_structured_events(observable, canonical)

    assert first == second
    assert first["task"] == "synthetic_structured_temporal_event_field_smoke_test"
    assert first["observable_rows"] == evaluation.PERQA_SAMPLE_SIZE
    assert first["linked_observable_rows"] == evaluation.PERQA_SAMPLE_SIZE
    assert first["event_data_json_valid"] == evaluation.PERQA_SAMPLE_SIZE
    assert first["properties_mentioned_json_valid"] == evaluation.PERQA_SAMPLE_SIZE
    assert first["non_null_temporal_values"] == 2 * evaluation.PERQA_SAMPLE_SIZE
    assert first["valid_non_null_temporal_values"] == 2 * evaluation.PERQA_SAMPLE_SIZE
    assert first["surface_declared_temporal_fields"] == 2 * evaluation.PERQA_SAMPLE_SIZE
    assert first["surface_declared_temporal_field_valid_rate"] == 1.0
    assert "not natural-language syllabus extraction" in first["scope_limit"]


def test_perqa_structured_smoke_rejects_malformed_json_and_bad_links():
    observable, canonical = _perqa_rows()
    malformed = copy.deepcopy(observable)
    malformed[0]["event_data"] = "not json"
    with pytest.raises(ValueError, match="invalid JSON"):
        evaluation.evaluate_perqa_structured_events(malformed, canonical)

    unlinked = copy.deepcopy(observable)
    unlinked[0]["canonicalized_event_id"] = -1
    with pytest.raises(ValueError, match="missing one or more"):
        evaluation.evaluate_perqa_structured_events(unlinked, canonical)

    invalid_date = copy.deepcopy(observable)
    invalid_date[0]["start_date"] = "2024-02-30"
    with pytest.raises(ValueError, match="invalid start_date"):
        evaluation.evaluate_perqa_structured_events(invalid_date, canonical)


def test_perqa_sample_loader_enforces_pinned_protocol(tmp_path: Path):
    sample_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "perqa_test_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    observable, canonical = evaluation.load_perqa_sample(sample_path)
    assert len(observable) == evaluation.PERQA_SAMPLE_SIZE
    assert len(canonical) == evaluation.PERQA_CANONICAL_FETCH_SIZE

    sample["dataset_revision"] = "unpinned"
    invalid_path = tmp_path / "invalid-sample.json"
    invalid_path.write_text(json.dumps(sample), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation.load_perqa_sample(invalid_path)

    tampered = json.loads(sample_path.read_text(encoding="utf-8"))
    tampered["configs"]["observable_events"]["rows"][0]["start_date"] = "2031-12-31"
    tampered_path = tmp_path / "tampered-sample.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        evaluation.run_evaluation(perqa_sample_path=tampered_path)


def test_multi_dataset_cli_writes_separate_offline_artifact(tmp_path: Path, monkeypatch, capsys):
    output = tmp_path / "metrics" / "evaluation.json"
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    evaluation.main(["--output", str(output)])

    summary = json.loads(capsys.readouterr().out)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert summary["output"] == str(output)
    assert result["tasks_are_separate"] is True
    assert result["metrics_are_not_aggregated_across_tasks"] is True
    assert result["cu_docbench_layout"]["gold_blocks"] == 1065
    assert result["perqa_structured_temporal_smoke"]["observable_rows"] == 20
    assert result["schedule_scanner"]["trained_or_calibrated_on_these_datasets"] is False
    assert result["schedule_scanner"]["inference_confidence_changed"] is False
