from pathlib import Path
import pytest


@pytest.mark.skipif(not Path("artifacts/usc-miniLM/model.pt").exists(), reason="trained local artifact is not present")
def test_saved_real_model_runs_offline_and_returns_source_evidence():
    from pace.real_infer import RealCatalogEngine

    engine = RealCatalogEngine(device="cpu")
    text = "BIO 101 with a grade of C or higher is required for enrollment."
    result = engine.classify_field(text)
    assert result["task"] == "catalog_field"
    assert result["label"] == "course_description"
    assert 0 <= result["confidence"] <= 1
    assert result["source_text"] == text
    assert result["proposed_only"] is True
    assert result["abstained"] is False
    assert result["confidence_calibrated"] is True
    assert result["probabilities_calibrated"] is True
    assert result["calibration_scope"] == "usc_course_catalog_cell_field_classification_with_subject_group_split"


def test_explicit_catalog_field_prefix_bypasses_ambiguous_text_classifier():
    from pace.real_infer import RealCatalogEngine

    engine = RealCatalogEngine(device="cpu", relation_checkpoint=None)
    result = engine.classify_field("Class Type: Lecture")
    assert result["label"] == "class_type"
    assert result["confidence"] == 1.0
    assert result["confidence_calibrated"] is False
    assert result["confidence_semantics"] == "explicit_source_field_prefix_match_not_model_confidence"
    assert result["reason"] == "explicit_source_field_prefix"


def test_all_supported_explicit_catalog_prefixes_return_expected_label():
    from pace.real_infer import RealCatalogEngine

    engine = RealCatalogEngine(device="cpu", relation_checkpoint=None)
    fields = {
        "Class Name: Cell Biology": "course_title",
        "Catalogue: A study of cells.": "course_description",
        "Units: 3.0 units": "credit_units",
        "Prerequisite: BIO 101": "prerequisites",
        "Restrictions: Juniors only": "restrictions",
        "Time: 2:00-3:20pm": "meeting_time",
        "Days: Mon, Wed": "meeting_days",
        "Class Type: Lecture": "class_type",
    }
    for text, expected in fields.items():
        assert engine.classify_field(text)["label"] == expected


@pytest.mark.skipif(not Path("artifacts/usc-miniLM/model.pt").exists(), reason="trained local artifact is not present")
def test_real_model_identifies_simple_field_examples_and_parses_structured_values():
    from pace.real_infer import RealCatalogEngine

    engine = RealCatalogEngine(device="cpu")
    units = engine.classify_field("(4.0 units, max 6)")
    assert units["label"] == "credit_units"
    assert units["structured_value"] == {"min_units": 4.0, "max_units": 4.0, "repeatable_max_units": 6.0}

    days = engine.classify_field("Mon, Wed")
    assert days["label"] == "meeting_days"
    assert days["structured_value"] == {"days": ["Mon", "Wed"]}

    time_res = engine.classify_field("2:00-3:20pm")
    assert time_res["label"] == "meeting_time"
    assert time_res["structured_value"] == {"start_time": "2:00", "end_time": "3:20", "meridiem": "pm"}

    ambiguous = engine.classify_field("TBA")
    assert ambiguous["label"] == "unknown" and ambiguous["abstained"] is True

    ood_thai = engine.classify_field("## ประมวลรายวิชา (Course Syllabus)")
    assert ood_thai["label"] == "unknown" and ood_thai["abstained"] is True
    assert ood_thai["reason"] == "out_of_distribution_script_or_html_table"


@pytest.mark.skipif(not Path("artifacts/usc-relations/model.pt").exists(), reason="trained relation artifact is not present")
def test_real_relation_engine_extracts_prerequisite_and_duplicate_credit():
    from pace.real_infer import RealCatalogEngine

    engine = RealCatalogEngine(device="cpu")
    desc = (
        "Introduction to algorithms. Prerequisite: CSCI 104 and CSCI 170. "
        "Recommended preparation: MATH 225. Duplicates credit in former CSCI 271."
    )
    res = engine.classify_field(desc, self_course="CSCI 270")
    assert res["label"] == "course_description"
    by_target = {r["target_course"]: r for r in res["course_relations"]}
    assert by_target["CSCI 104"]["relation"] == "prerequisite_of"
    assert by_target["CSCI 104"]["confidence_calibrated"] is False
    assert "weakly_supervised" in by_target["CSCI 104"]["confidence_semantics"]
    assert by_target["CSCI 104"]["source_text"] == desc
    assert by_target["CSCI 170"]["relation"] == "prerequisite_of"
    assert by_target["MATH 225"]["relation"] == "recommended_for"
    assert by_target["CSCI 271"]["relation"] == "duplicate_credit_of"


@pytest.mark.skipif(not Path("artifacts/usc-miniLM/onnx/model_int8.onnx").exists(), reason="ONNX INT8 artifact is not present")
def test_onnx_int8_engine_matches_pytorch_predictions():
    from pace.real_infer import RealCatalogEngine

    pt_engine = RealCatalogEngine(device="cpu")
    onnx_engine = RealCatalogEngine(onnx_path="artifacts/usc-miniLM/onnx/model_int8.onnx")
    for sample in ["(4.0 units, max 6)", "Mon, Wed", "2:00-3:20pm", "Lecture"]:
        assert pt_engine.classify_field(sample)["label"] == onnx_engine.classify_field(sample)["label"]
