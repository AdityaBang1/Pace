"""Focused regression coverage for the synthetic-only PyTorch multitask experiment."""
import json
import re
from pathlib import Path

import pytest

from pace.synthetic_academic import DOCUMENT_TYPES, generate_dataset, split_by_template


def test_syllabus_grade_scale_letters_are_not_marks_and_a_later_code_is_not_a_prerequisite():
    """A bare letter in a grade scale is not a mark; a code after a cue is a code."""
    rows = [row for row in generate_dataset(600, 31, 0.0) if row["document_type"] == "syllabus"]
    assert rows
    letters = {"A", "B", "C", "D", "F"}
    saw_scale = False
    saw_code_after_a_prerequisite = False
    for row in rows:
        text = row["text"]
        if re.search(r"grade(?:s| scale)|grading scale|final grades", text, re.I):
            saw_scale = True
        for entity in row["entities"]:
            # The scale is context, not an awarded mark.
            assert entity["type"] != "MARK"
            assert entity["text"] not in letters
            prerequisite_cue = re.search(r"prerequisite|required prior course|must complete|required before enrollment", text[: entity["start"]], re.I)
            if entity["type"] == "COURSE_CODE" and prerequisite_cue:
                saw_code_after_a_prerequisite = True
    assert saw_scale
    assert saw_code_after_a_prerequisite


def test_times_and_dates_are_gold_entities_wherever_they_appear():
    """A time of day and a calendar date are entities in any line, not only on a line that names an assessment."""
    rows = generate_dataset(600, 41, 0.0)
    saw_prose_time = False
    saw_prose_date = False
    for row in rows:
        times = [e for e in row["entities"] if e["type"] == "TIME"]
        dates = [e for e in row["entities"] if e["type"] == "DATE"]
        for entity in times + dates:
            # The span must cover the value exactly, with no trailing space.
            assert entity["text"] == entity["text"].strip()
            assert row["text"][entity["start"]:entity["end"]] == entity["text"]
        for entity in times:
            assert re.search(r"\d{1,2}:\d{2}", entity["text"])
            line = row["text"][entity["start"]:entity["end"]]
            assert line == entity["text"]
        if any("Office hours" in row["text"] or "Student services open" in row["text"] for _ in [0]):
            prose_times = {e["text"] for e in times}
            assert prose_times, row["text"]
            saw_prose_time = True
        if "Course materials were revised" in row["text"] or "department website was updated" in row["text"]:
            prose_dates = {e["text"] for e in dates}
            assert any(re.match(r"[A-Z][a-z]+ \d{1,2}, \d{4}$", value) for value in prose_dates), row["text"]
            saw_prose_date = True
    assert saw_prose_time
    assert saw_prose_date


def test_course_codes_come_from_the_world_and_not_from_a_code_shaped_shape():
    """Prose that merely looks like a code is not a course code."""
    rows = [row for row in generate_dataset(900, 47, 0.0) if row["document_type"] != "transcript"]
    saw_prose_lookalike = False
    for row in rows:
        forms = {form.upper() for form in row["gold"]["course_code_forms"]}
        for entity in row["entities"]:
            if entity["type"] == "COURSE_CODE":
                assert entity["text"].upper() in forms, entity["text"]
        if "AT 14" in row["text"] or "TERM 2" in row["text"].upper():
            saw_prose_lookalike = True
            assert not [e for e in row["entities"] if e["type"] == "COURSE_CODE" and e["text"].upper() in {"AT 14", "TERM 2"}]
    assert saw_prose_lookalike


def test_every_assessment_name_the_document_prints_is_a_gold_assessment():
    """The names of dated work come from the world state, so all of them are covered."""
    rows = [row for row in generate_dataset(900, 53, 0.0) if row["document_type"] != "transcript"]
    saw_assessment = False
    for row in rows:
        names = [entry["name"] for entry in row["gold"]["assessments"]] + [entry["name"] for entry in row["gold"]["deliverables"]]
        printed = [name for name in names if name in row["text"]]
        assessments = {e["text"] for e in row["entities"] if e["type"] == "ASSESSMENT"}
        for name in printed:
            assert name in assessments, (row["text"], assessments)
        if printed:
            saw_assessment = True
    assert saw_assessment


def test_gradebook_weights_are_gold_for_every_category_it_prints():
    """Every category the generator prints as "Category: N%" is a gold weight, so the tagger is not penalised for finding it."""
    rows = [row for row in generate_dataset(900, 43, 0.0) if row["document_type"] == "gradebook"]
    assert rows
    for row in rows:
        weights = {e["text"] for e in row["entities"] if e["type"] == "WEIGHT"}
        printed = set(re.findall(r"\b\d{1,3}%\s*$", row["text"], flags=re.M))
        for value in printed:
            assert value.strip() in weights, (row["text"], weights)


def test_a_value_that_merely_ends_at_a_window_edge_is_not_treated_as_cut():
    from pace.synthetic_doc_model import is_word_cut

    text = "Room: Science 2.\nNext"
    # "Science 2" ends where the window ends, but its word is complete.
    assert is_word_cut(text, 6, 15, 0, 15) is False
    # A window that stopped inside the word "Science" did read a fragment.
    assert is_word_cut(text, 6, 11, 0, 11) is True
    # A value that starts at the window start is complete when the window starts
    # at the document start.
    assert is_word_cut(text, 0, 4, 0, 4) is False
    # A value whose word continues past the window edge was read as a fragment.
    assert is_word_cut(text, 6, 13, 0, 13) is False
    assert is_word_cut("Sciences are here", 0, 7, 0, 7) is True


def test_synthetic_split_holds_out_schools_and_layout_families():
    splits = split_by_template(generate_dataset(5000, 17, 0.02), 17)
    school_sets = {name: {row["world_school_id"] for row in rows} for name, rows in splits.items()}
    assert not (school_sets["train"] & school_sets["validation"])
    assert not (school_sets["train"] & school_sets["test"])
    assert not (school_sets["validation"] & school_sets["test"])
    family_sets = {name: {row["template_family"] for row in rows} for name, rows in splits.items()}
    assert not (family_sets["train"] & family_sets["validation"])
    assert not (family_sets["train"] & family_sets["test"])
    assert not (family_sets["validation"] & family_sets["test"])
    for rows in splits.values():
        assert {row["document_type"] for row in rows} == set(DOCUMENT_TYPES)


def test_synthetic_entity_spans_are_literal_and_prereq_role_is_correct():
    rows = generate_dataset(800, 29, 0.0)
    saw_prereq = False
    for row in rows:
        for entity in row["entities"]:
            assert row["text"][entity["start"]:entity["end"]] == entity["text"]
        for entity in row["entities"]:
            if entity["type"] == "PREREQUISITE":
                saw_prereq = True
                assert "prerequisite" in row["text"].casefold() or "required before enrollment" in row["text"].casefold() or "must complete" in row["text"].casefold()
                assert entity["text"] in row["gold"]["prerequisites"]
    assert saw_prereq


def test_jev_choice_confidence_uses_distribution_sharpness():
    from pace.synthetic_doc_model import choice_confidence

    assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="sum to 1"):
        choice_confidence([0.7, 0.4])
    with pytest.raises(ValueError, match="at least two"):
        choice_confidence([1.0])


def test_entity_type_distribution_is_normalized_over_entity_types():
    import torch

    from pace.synthetic_doc_model import (
        ENTITY_LABELS,
        ENTITY_TO_ID,
        ENTITY_TYPES,
        SINGLE_TOKEN_ENTITY_TYPES,
        entity_type_distribution,
    )

    token_probabilities = torch.full((2, len(ENTITY_LABELS)), 0.001)
    token_probabilities[0, ENTITY_TO_ID["B-CREDIT"]] = 0.8
    token_probabilities[1, ENTITY_TO_ID["I-CREDIT"]] = 0.8
    # A two-token span cannot be a single-token value, so those types are excluded.
    probabilities = entity_type_distribution(token_probabilities, [0, 1])
    assert set(probabilities) == set(ENTITY_TYPES) - SINGLE_TOKEN_ENTITY_TYPES
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities, key=probabilities.get) == "CREDIT"

    single = entity_type_distribution(token_probabilities[:1], [0])
    assert set(single) == set(ENTITY_TYPES)
    assert sum(single.values()) == pytest.approx(1.0)


@pytest.mark.skipif(not Path("artifacts/synthetic-multitask/model.pt").exists(), reason="trained synthetic PyTorch artifact is unavailable")
def test_synthetic_pytorch_cli_serializes_choice_distributions_and_proposal_policy(tmp_path, capsys):
    from transformers import AutoTokenizer

    from pace.cli import main
    from pace.synthetic_doc_model import ENTITY_TYPES, SINGLE_TOKEN_ENTITY_TYPES, choice_confidence

    tokenizer = AutoTokenizer.from_pretrained("artifacts/synthetic-multitask/tokenizer")

    fixture = "Syllabus\n\nInstructor: Dr. Avery Morgan\nBIO 210 Cell Biology.\nPrerequisite: CHEM 118\nCredits: 3.0 credits\nGrades: A, B, C, D, F"
    source = tmp_path / "fixture.txt"
    # Offsets in the result index this exact text; the platform's default line
    # ending would make the file longer than the string being compared against.
    source.write_text(fixture, encoding="utf-8", newline="\n")
    main(["analyze", str(source), "--checkpoint", "artifacts/synthetic-multitask", "--synthetic-pytorch"])
    result = json.loads(capsys.readouterr().out)

    document = result["document_type"]
    assert set(document["probabilities"]) == set(DOCUMENT_TYPES)
    assert sum(document["probabilities"].values()) == pytest.approx(1.0)
    assert document["label"] == max(document["probabilities"], key=document["probabilities"].get)
    assert document["confidence"] == pytest.approx(choice_confidence(document["probabilities"].values()))
    assert document["confidence_semantics"] == "jev_choice_sharpness"
    assert document["confidence_calibrated"] is False
    assert document["probabilities_calibrated"] is False
    assert result["requires_review"] is True
    assert result["commit_policy"] == "proposals_only_user_confirmation_required"
    assert result["model_scope"] == "synthetic_text_only_unvalidated_on_real_documents"
    assert "uncalibrated" in result["confidence_semantics"]
    assert any(entity["type"] == "PREREQUISITE" for entity in result["entities"])
    assert any(entity["type"] == "LITERAL_CANDIDATE" for entity in result["entities"])
    fallback_texts = {entity["text"] for entity in result["entities"] if entity["type"] == "LITERAL_CANDIDATE"}
    assert fallback_texts == {"A", "B", "C", "D", "F"}
    for entity in result["entities"]:
        assert fixture[entity["start"]:entity["end"]] == entity["source_text"]
        assert entity["requires_review"] is True
        assert entity["confidence_calibrated"] is False
        assert entity["probabilities_calibrated"] is False
        if entity["type"] == "LITERAL_CANDIDATE":
            assert entity["confidence_semantics"] == "literal_pattern_match_not_model_confidence"
            continue
        # A value that is more than one token cannot be a single-token entity type, so
        # those types are excluded from the distribution for a multi-token span.
        token_count = len(tokenizer.tokenize(entity["source_text"]))
        expected_types = set(ENTITY_TYPES) if token_count <= 1 else set(ENTITY_TYPES) - SINGLE_TOKEN_ENTITY_TYPES
        assert set(entity["type_probabilities"]) == expected_types
        assert sum(entity["type_probabilities"].values()) == pytest.approx(1.0)
        assert entity["type"] == max(entity["type_probabilities"], key=entity["type_probabilities"].get)
        assert entity["confidence"] == pytest.approx(choice_confidence(entity["type_probabilities"].values()))
        assert entity["confidence_semantics"] == "jev_choice_sharpness"
        assert entity["decision_scope"] == "conditional_on_proposed_span_boundary"


@pytest.mark.skipif(not Path("artifacts/synthetic-multitask/model.pt").exists(), reason="trained synthetic PyTorch artifact is unavailable")
def test_sliding_windows_collapse_competing_boundaries_and_report_truncation():
    from pace.synthetic_doc_model import SyntheticDocumentEngine

    engine = SyntheticDocumentEngine(device="cpu", max_windows=8)
    block = (
        "Course Syllabus\nBIO 210 Cell Biology. Instructor: Dr. Avery Morgan. "
        "Prerequisite: CHEM 118. Credits: 3.0 credits. Room: Science 2.\n"
    )
    text = block * 8
    result = engine.analyze_document(text)
    teachers = [entity for entity in result["entities"] if entity["type"] == "TEACHER"]
    assert len(teachers) == 8
    assert all(entity["text"] == "Dr. Avery Morgan" for entity in teachers)
    assert all(text[e["start"]:e["end"]] == e["source_text"] for e in result["entities"])
    assert all(
        (a["start"], a["end"]) != (b["start"], b["end"])
        for index, a in enumerate(teachers)
        for b in teachers[index + 1:]
    )

    truncated = engine.analyze_document("word " * 5000)
    assert truncated["window_count"] == 8
    assert truncated["window_limit_reached"] is True


def test_empty_document_is_rejected_by_synthetic_engine():
    from pace.synthetic_doc_model import SyntheticDocumentEngine

    engine = SyntheticDocumentEngine(device="cpu")
    for text in ("", " ", "\n\t"):
        with pytest.raises(ValueError, match="must not be empty"):
            engine.analyze_document(text)


@pytest.mark.skipif(not Path("artifacts/synthetic-multitask/model.pt").exists(), reason="trained synthetic PyTorch artifact is unavailable")
def test_engine_drops_punctuation_only_and_line_spanning_proposals():
    """A hyphen inside a masked phone number and a span running into the next row are not values."""
    from pace.synthetic_doc_model import SyntheticDocumentEngine

    engine = SyntheticDocumentEngine(device="cpu")
    text = "Lepid, Arthur\n3160 Harbor Mtn Road\nPhone: (907) xxx-xxxx\n"
    result = engine.analyze_document(text)
    for entity in result["entities"]:
        assert any(char.isalnum() for char in entity["text"]), entity
        assert "\n" not in entity["text"] and "\r" not in entity["text"], entity
