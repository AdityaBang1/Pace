"""Regression coverage for the transcript path: parser, model bridge and evaluation."""
import hashlib
import re
from pathlib import Path

import pytest

from pace.evaluate_transcript import gold_entities_from_parser, span_scores
from pace.transcript_extract import parse_transcript, summarize

MOCK_TABLE = Path("data/samples/mock_transcript_lepid.txt")
MOCK_ROWWISE = Path("data/samples/mock_transcript_lepid_rowwise.txt")
MOCK_SYLLABUS = Path("data/samples/mock_syllabus.txt")


@pytest.mark.skipif(not MOCK_TABLE.exists(), reason="mock transcript fixture is unavailable")
def test_parser_reads_the_mock_transcript_rows_sections_and_credits():
    parsed = parse_transcript(MOCK_TABLE.read_text(encoding="utf-8"))
    summary = summarize(parsed)

    assert summary["student_name"] == "Lepid, Arthur"
    assert summary["school"] == "Pacific High School"
    assert summary["course_records"] == 38
    assert summary["assessment_records"] == 3
    assert summary["document_kind"] == "academic_history"
    assert summary["looks_like_transcript"] is True
    assert summary["warnings"] == []
    assert len(summary["sections"]) == 11
    # The printed credit column of the mock adds up to the total its own footer
    # states, so the parser confirms the document instead of reporting a gap.
    assert summary["computed_total_credits"] == 13.25
    assert summary["reported_total_credits"] == 13.25
    assert summary["credit_sum_matches_reported_total"] is True
    assert parsed["checks"]["credit_sum_difference"] == 0.0
    # The document states no GPA value, so none may be invented.
    assert summary["gpa"] is None
    assert parsed["checks"]["records_without_title"] == 0
    assert parsed["checks"]["records_without_section"] == 0
    assert parsed["checks"]["unattached_performance_levels"] == 0


@pytest.mark.skipif(not MOCK_SYLLABUS.exists(), reason="mock syllabus fixture is unavailable")
def test_a_syllabus_is_not_passed_off_as_a_transcript_with_zero_rows():
    parsed = parse_transcript(MOCK_SYLLABUS.read_text(encoding="utf-8"))
    summary = summarize(parsed)

    assert parsed["checks"]["looks_like_transcript"] is False
    assert parsed["document_kind"] == "unrecognized_text"
    assert summary["looks_like_transcript"] is False
    assert len(summary["warnings"]) == 1
    assert "does not look like a transcript" in summary["warnings"][0]


@pytest.mark.skipif(not MOCK_ROWWISE.exists(), reason="mock transcript fixture is unavailable")
def test_row_wise_two_column_ocr_of_the_same_transcript_keeps_the_totals():
    parsed = parse_transcript(MOCK_ROWWISE.read_text(encoding="utf-8"))
    summary = summarize(parsed)

    assert summary["column_interleaving_detected"] is True
    assert summary["course_records"] == 38
    assert summary["assessment_records"] == 3
    assert summary["computed_total_credits"] == 13.25
    assert summary["credit_sum_matches_reported_total"] is True
    assert sorted(summary["sections"]) == sorted(summarize(parse_transcript(MOCK_TABLE.read_text(encoding="utf-8")))["sections"])


@pytest.mark.skipif(not MOCK_ROWWISE.exists(), reason="mock transcript fixture is unavailable")
def test_two_column_rows_keep_the_subject_area_of_their_own_column():
    """A heading in the right column must not capture the left column's rows."""
    aligned = parse_transcript(MOCK_TABLE.read_text(encoding="utf-8"))
    rowwise = parse_transcript(MOCK_ROWWISE.read_text(encoding="utf-8"))

    def rows(parsed):
        return sorted(
            (record["section"], record["date"], record["title"].strip(), record["mark"], round(float(record["credit"]), 3))
            for record in parsed["course_records"]
        )

    assert rows(rowwise) == rows(aligned)
    assert parsed_sections(rowwise) == parsed_sections(aligned)


def parsed_sections(parsed):
    return sorted(parsed["checks"]["records_per_subject_area"].items())


def test_credit_column_that_contradicts_the_footer_is_reported_not_hidden():
    """A transcript whose rows disagree with its own total must be flagged."""
    text = "\n".join([
        "ACADEMIC HISTORY",
        "",
        "Pacific High School",
        "509 Lincoln Street",
        "Sitka, AK 99835",
        "",
        "Lepid, Arthur                           Sex  Male",
        "",
        "DATE         GL  TITLE                        MARK  CREDIT",
        "",
        "ELECTIVE",
        "03/20/04     09  Elective Intensive          A     0.250",
        "",
        "ACADEMIC STANDING",
        "As of:          Total Credits       GPA",
        "12/20/05        13.25",
        "",
    ])
    parsed = parse_transcript(text)
    summary = summarize(parsed)
    assert summary["course_records"] == 1
    assert summary["computed_total_credits"] == 0.25
    assert summary["reported_total_credits"] == 13.25
    assert summary["credit_sum_matches_reported_total"] is False
    # The difference is signed: computed minus reported.
    assert parsed["checks"]["credit_sum_difference"] == -13.0


@pytest.mark.skipif(not MOCK_TABLE.exists(), reason="mock transcript fixture is unavailable")
def test_parser_evidence_spans_are_literal_substrings_of_the_source():
    text = MOCK_TABLE.read_text(encoding="utf-8")
    parsed = parse_transcript(text)
    for entity in gold_entities_from_parser(parsed):
        assert entity["start"] < entity["end"]
        assert text[entity["start"]:entity["end"]].strip()


@pytest.mark.skipif(not MOCK_TABLE.exists(), reason="mock transcript fixture is unavailable")
def test_gold_span_scores_reward_exact_agreement_only():
    gold = [{"type": "DATE", "start": 10, "end": 18}]
    perfect = span_scores(gold, [{"type": "DATE", "start": 10, "end": 18}])
    assert perfect["f1"] == 1.0
    wrong_type = span_scores(gold, [{"type": "COURSE", "start": 10, "end": 18}])
    assert wrong_type["f1"] == 0.0
    wrong_bounds = span_scores(gold, [{"type": "DATE", "start": 10, "end": 17}])
    assert wrong_bounds["f1"] == 0.0


def test_entity_type_distribution_excludes_single_token_types_for_long_spans():
    import torch

    from pace.synthetic_doc_model import (
        ENTITY_LABELS,
        ENTITY_TO_ID,
        ENTITY_TYPES,
        SINGLE_TOKEN_ENTITY_TYPES,
        entity_type_distribution,
    )

    probabilities = torch.full((3, len(ENTITY_LABELS)), 0.001)
    probabilities[0, ENTITY_TO_ID["B-CREDIT"]] = 0.8
    probabilities[1, ENTITY_TO_ID["I-CREDIT"]] = 0.8
    probabilities[2, ENTITY_TO_ID["I-CREDIT"]] = 0.8

    long_span = entity_type_distribution(probabilities, [0, 1, 2])
    assert max(long_span, key=long_span.get) == "CREDIT"
    assert not (set(long_span) & SINGLE_TOKEN_ENTITY_TYPES)
    assert sum(long_span.values()) == pytest.approx(1.0)

    single_span = entity_type_distribution(probabilities[:1], [0])
    assert set(single_span) == set(ENTITY_TYPES)
    assert sum(single_span.values()) == pytest.approx(1.0)


def test_span_decoding_recovers_an_continuation_tag_that_opens_an_entity():
    from pace.synthetic_doc_model import ENTITY_TO_ID, spans_from_iob

    offsets = [(0, 0), (0, 2), (3, 5)]
    tags = [0, ENTITY_TO_ID["I-GRADE_LEVEL"], ENTITY_TO_ID["I-GRADE_LEVEL"]]
    spans = spans_from_iob(tags, offsets)
    assert spans == [{"type": "GRADE_LEVEL", "start": 0, "end": 5}]


def test_transcript_ir_bridge_reports_the_trigger_and_keeps_proposals():
    from pace.synthetic_doc_model import attach_transcript_ir

    text = MOCK_TABLE.read_text(encoding="utf-8") if MOCK_TABLE.exists() else ""
    if not text:
        pytest.skip("mock transcript fixture is unavailable")
    result = {"document_type": {"label": "transcript"}}
    attached = attach_transcript_ir(result, text)
    assert attached["transcript"]["triggered_by"] == "both"
    assert attached["transcript"]["parser"] == "deterministic_rule_based_no_model"
    assert attached["transcript"]["summary"]["course_records"] == 38
    assert attached["transcript"]["parsed"]["requires_review"] is True
    assert attached["transcript"]["parsed"]["commit_policy"] == "proposals_only_user_confirmation_required"


def test_transcript_ir_bridge_stays_silent_on_a_non_transcript_document():
    from pace.synthetic_doc_model import attach_transcript_ir

    result = {"document_type": {"label": "syllabus"}}
    attached = attach_transcript_ir(result, "Syllabus\nBIO 210 Cell Biology\nInstructor: Dr. Avery Morgan")
    assert "transcript" not in attached


def _course_key(record):
    return (
        record["date"],
        record["grade_level"],
        record["title"].strip(),
        record["mark"],
        round(float(record["credit"]), 3),
    )


def test_parser_recovers_every_generated_transcript_layout():
    from pace.synthetic_academic import generate_academic_world
    from pace.synthetic_transcript import TRANSCRIPT_FAMILIES, render_transcript_document

    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    for family in TRANSCRIPT_FAMILIES:
        for seed in (91, 137, 613):
            world = generate_academic_world(4242 + seed)
            document = render_transcript_document(
                world, seed, family, "Cedar Ridge High School", "cedar-ridge-high-school", digest
            )
            parsed = parse_transcript(document["text"])
            gold = document["gold"]

            assert len(parsed["course_records"]) == len(gold["course_records"]), (family, seed)
            assert len(parsed["assessment_records"]) == len(gold["assessment_records"]), (family, seed)
            assert {_course_key(record) for record in parsed["course_records"]} == {
                _course_key(record) for record in gold["course_records"]
            }, (family, seed)
            assert {(record["date"], record["grade_level"], record["title"].strip(), record["score"])
                    for record in parsed["assessment_records"]} == {
                (record["date"], record["grade_level"], record["title"].strip(), record["score"])
                for record in gold["assessment_records"]
            }, (family, seed)
            assert parsed["checks"]["computed_total_credits"] == pytest.approx(gold["total_credits"]), (family, seed)
            assert len(parsed["sections"]) == len({
                record["section"] for record in gold["course_records"] + gold["assessment_records"]
            }), (family, seed)
            for record in parsed["assessment_records"]:
                assert record["performance_level"] is not None, (family, seed)
                assert record["performance_label"], (family, seed)


def test_generated_transcripts_cover_both_gpa_and_empty_gpa_footers():
    from pace.synthetic_academic import generate_academic_world
    from pace.synthetic_transcript import render_transcript_document

    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    with_gpa = 0
    without_gpa = 0
    for index in range(40):
        world = generate_academic_world(1000 + index)
        document = render_transcript_document(
            world, 500 + index, "transcript_table", "Cedar Ridge High School", "cedar-ridge-high-school", digest
        )
        parsed = parse_transcript(document["text"])
        standing = parsed["academic_standing"]
        assert standing["total_credits"] is not None
        if document["gold"]["gpa"] is None:
            without_gpa += 1
            assert standing["gpa"] is None
        else:
            with_gpa += 1
            assert standing["gpa"]["value"] == pytest.approx(document["gold"]["gpa"])
    assert with_gpa > 0
    assert without_gpa > 0


TERM_TABLE = Path("data/samples/transcript_term_marks.txt")
GRADE_REPORT = Path("data/samples/grade_report.txt")


def _grade_report_text():
    return GRADE_REPORT.read_text(encoding="utf-8")


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_a_course_block_grade_report_reads_every_course_its_grade_and_its_tasks():
    """A page with no dates, no credit and no weight is still a table of courses."""
    parsed = parse_transcript(_grade_report_text())
    summary = summarize(parsed)
    rows = [(record["course_code"], record["section_number"], record["title"],
             record["instructor"], record["period"], record["mark"]) for record in parsed["course_records"]]

    assert rows == [
        ("10145y", "101", "Literature & Composition II", "Fireall, Ashley N", "9 WEEK", "89"),
        ("220745y", "302", "AP Precalculus", "Shaw, Christian Evans", "9 WEEK", "82"),
        ("32145y", "501", "Chemistry I Honors", "Ash, Christopher John", "9 WEEK", "96"),
        ("42135y", "402", "AP World History", "Berry, Gavin Scott", "9 WEEK", "92"),
        ("50725y", "701", "Spanish III", "Elison, Robert Ross", "9 WEEK", "93"),
        ("700695y", "601", "AP Business", "Moody, Christopher M", "9 WEEK", "96"),
        ("728305y", "202", "AP Computer Science Principles",
         "de la Rosa, Jacquelyn Gabriellj", "9 WEEK", "97"),
    ]
    assert summary["layout"] == "course_task_grade"
    # The page names its own mark column, and that name is reported as printed.
    assert summary["grade_columns"] == ["S1"]
    assert summary["looks_like_transcript"] is True
    assert summary["warnings"] == []
    # No credit column is printed, so the compact view reports none rather than zero.
    assert summary["computed_total_credits"] is None
    assert parsed["checks"]["records_without_grade"] == 0
    assert [record["mark_percent"] for record in parsed["course_records"]] == [89, 82, 96, 92, 93, 96, 97]
    # The header row is not a subject-area heading, and the attendance table is not
    # the school name printed above the body.
    assert summary["sections"] == []
    assert summary["school"] is None


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_a_grade_report_invents_no_credit_it_never_printed():
    parsed = parse_transcript(_grade_report_text())
    assert [record["credit"] for record in parsed["course_records"]] == [None] * 7
    assert [record["weight"] for record in parsed["course_records"]] == [None] * 7
    assert parsed["checks"]["credits_printed"] is False
    assert parsed["checks"]["credit_sum_matches_reported_total"] is None
    # The compact view says "no credit column", not "zero credits earned".
    assert summarize(parsed)["computed_total_credits"] is None


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_each_graded_task_stays_with_the_course_block_above_it():
    parsed = parse_transcript(_grade_report_text())
    tasks = [(record["course_code"], record["title"], record["score"])
             for record in parsed["assessment_records"]]

    assert tasks[:4] == [("10145y", "Assignment Completion", 2),
                         ("10145y", "Interpersonal Skills", 2),
                         ("10145y", "Participation", 2),
                         ("10145y", "Responsibility", 2)]
    assert len(tasks) == 25
    assert sum(1 for code, _title, _score in tasks if code == "10145y") == 4
    # The last course's tasks are printed after its own wrapped header line, so the
    # wrap must not be read as a new course and must not steal the next block either.
    assert [task for task in tasks if task[0] == "728305y"] == [
        ("728305y", "Assignment Completion", 2)]
    # One label was lost by the scan. Its mark is kept and no label is invented for it.
    assert [(code, score) for code, title, score in tasks if title is None] == [("32145y", 2)]


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_a_course_header_that_wraps_onto_the_next_line_is_one_course():
    """The second printed line finishes the instructor name and carries the grade."""
    text = _grade_report_text()
    parsed = parse_transcript(text)
    last = parsed["course_records"][-1]

    assert last["title"] == "AP Computer Science Principles"
    assert last["instructor"] == "de la Rosa, Jacquelyn Gabriellj"
    assert last["instructor_wrapped"] is True
    assert last["title_wrapped"] is False
    assert (last["period"], last["mark"], last["mark_percent"]) == ("9 WEEK", "97", 97)
    # One value printed on two lines keeps one evidence span per printed line.
    lines = last["evidence"]["instructor_lines"]
    assert [line["text"] for line in lines] == ["de la Rosa,", "Jacquelyn Gabriellj"]
    for line in lines:
        assert text[line["start"]:line["end"]] == line["text"]
    assert last["evidence"]["mark"]["text"] == "97"
    assert "unread" not in last


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_a_course_whose_grade_never_prints_is_refused_with_its_reason():
    text = "\n".join([
        "Grade Report:",
        "Course                                Task                            S1",
        "10145y-101 Literature & Composition II [Fireall, Ashley N]  9 WEEK     89",
        "                Assignment Completion                                    2",
        "32145y-501 Chemistry I Honors [Ash, Christopher John]       9 WEEK",
        "                Assignment Completion                                    2",
    ])
    parsed = parse_transcript(text)

    assert [record["mark"] for record in parsed["course_records"]] == ["89", None]
    missing = parsed["course_records"][1]["unread"]["mark"]
    assert missing["page_reading"] is None
    assert "not printed" in missing["reason"]
    assert [record["score"] for record in parsed["assessment_records"]] == [2, 2]


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_the_attendance_table_above_the_report_is_read_not_dropped():
    parsed = parse_transcript(_grade_report_text())
    rows = [(record["course_code"], record["title"], record["absent"], record["tardy"])
            for record in parsed["attendance_records"]]

    assert len(rows) == 6
    assert rows[0] == ("700695y", "AP Business", 0, 0)
    assert rows[-1] == ("50725y", "Spanish III", 0, 0)
    assert summarize(parsed)["attendance_records"] == 6
    assert parsed["checks"]["attendance_record_count"] == 6


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_the_grade_report_survives_a_text_layer_that_collapsed_its_columns():
    """A PDF text layer that loses the column gaps must give the same rows."""
    aligned = parse_transcript(_grade_report_text())
    collapsed = parse_transcript(re.sub(r"[ \t]{2,}", " ", _grade_report_text()))

    def rows(parsed):
        return [(record["course_code"], record["title"], record["instructor"],
                 record["period"], record["mark"]) for record in parsed["course_records"]]

    assert rows(collapsed) == rows(aligned)
    assert [(record["course_code"], record["title"], record["score"])
            for record in collapsed["assessment_records"]] == [
        (record["course_code"], record["title"], record["score"])
        for record in aligned["assessment_records"]]
    assert [(record["course_code"], record["absent"], record["tardy"])
            for record in collapsed["attendance_records"]] == [
        (record["course_code"], record["absent"], record["tardy"])
        for record in aligned["attendance_records"]]


def test_a_grade_report_whose_column_gaps_were_lost_still_splits_its_codes():
    """OCR glues the code to the title; the code the page prints with a section splits it."""
    text = "\n".join([
        "Course            Absent     Tardy",
        "10145yLiterature&CompositionII            0        0",
        "Grade Report:",
        "Course            Task            s1",
        "10145y-101Literature&CompositionII[Fireall,AshleyN]  6 WEEK            68",
        "Assignment Completion            2",
        "728305y-202APComputerSciencePrinciples[de la Rosa,",
        "Jacquelyn Gabriellj            9 WEEK            97",
        "Assignment Completion            2",
    ])
    parsed = parse_transcript(text)
    rows = [(record["course_code"], record["section_number"], record["instructor"],
             record["period"], record["mark"]) for record in parsed["course_records"]]

    assert rows == [("10145y", "101", "Fireall,AshleyN", "6 WEEK", "68"),
                    ("728305y", "202", "de la Rosa, Jacquelyn Gabriellj", "9 WEEK", "97")]
    assert [record["title"] for record in parsed["course_records"]] == [
        "Literature&CompositionII", "APComputerSciencePrinciples"]
    assert [record["title"] for record in parsed["assessment_records"]] == [
        "Assignment Completion", "Assignment Completion"]
    assert [(record["course_code"], record["absent"], record["tardy"])
            for record in parsed["attendance_records"]] == [("10145y", 0, 0)]


def test_a_letter_marked_task_keeps_its_mark_and_claims_no_percentage():
    text = "\n".join([
        "Grade Report:",
        "Course                    Task                  S1",
        "10145y-101 Literature [Fireall, Ashley N]  9 WEEK    89",
        "                Assignment Completion                  A",
        "                Participation                          2",
    ])
    parsed = parse_transcript(text)

    assert [(record["title"], record["score"], record["mark"])
            for record in parsed["assessment_records"]] == [
        ("Assignment Completion", None, "A"), ("Participation", 2, "2")]
    # The letter is a task mark, not the rest of the course title above it.
    assert [record["title"] for record in parsed["course_records"]] == ["Literature"]


@pytest.mark.skipif(not GRADE_REPORT.exists(), reason="grade report fixture is unavailable")
def test_every_grade_report_value_is_a_literal_slice_of_the_source():
    text = _grade_report_text()
    parsed = parse_transcript(text)
    records = (parsed["course_records"] + parsed["assessment_records"]
               + parsed["attendance_records"])
    for record in records:
        for name, evidence in record["evidence"].items():
            pieces = ([evidence] if isinstance(evidence, dict) else list(evidence or []))
            for piece in pieces:
                assert piece["start"] < piece["end"], (name, record.get("course_code"))
                assert text[piece["start"]:piece["end"]] == piece["text"], (name, piece)


def _term_table_text():
    return TERM_TABLE.read_text(encoding="utf-8")


@pytest.mark.skipif(not TERM_TABLE.exists(), reason="term-block transcript fixture is unavailable")
def test_term_block_reads_codes_marks_weights_and_credits_row_for_row():
    """A COURSE / MARK / WEIGHT / CREDIT block with no date column still reads exactly."""
    summary = summarize(parse_transcript(_term_table_text()))
    parsed = parse_transcript(_term_table_text())
    rows = [(record["course_code"], record["title"], record["mark_percent"],
             record["weight"], record["credit"]) for record in parsed["course_records"]]

    assert rows == [
        ("220725y", "Accel Adv Alg:Concept&Connect", 94, 1.0, 1.0),
        ("200715b", "Accel Geo B:Concept&Connect", 94, 0.5, 0.5),
        ("42075y", "AP Human Geography", 90, 1.0, 1.0),
        ("32115y", "Biology I Honors", 93, 1.0, 1.0),
        ("700605y", "Intro to Business and Tech", 98, 1.0, 1.0),
        ("708325y", "Intro to Software Tech", 100, 1.0, 1.0),
        ("12135y", "Literature & Composition I Hnr", 91, 1.0, 1.0),
        ("50715y", "Spanish II", 97, 1.0, 1.0),
    ]
    assert summary["layout"] == "course_mark_weight_credit"
    assert summary["computed_total_credits"] == 7.5
    assert summary["reported_total_credits"] == 7.5
    assert summary["credit_sum_matches_reported_total"] is True
    assert summary["gpa"] == pytest.approx(4.13333)
    assert parsed["checks"]["computed_total_weight"] == 7.5
    assert parsed["checks"]["records_without_title"] == 0
    assert parsed["checks"]["records_without_term"] == 0


@pytest.mark.skipif(not TERM_TABLE.exists(), reason="term-block transcript fixture is unavailable")
def test_a_wrapped_course_title_is_one_value_printed_on_two_lines():
    """The wrapped half of a title completes it; it never becomes a row of its own."""
    text = _term_table_text()
    parsed = parse_transcript(text)
    assert len(parsed["course_records"]) == 8

    wrapped = [record for record in parsed["course_records"] if record["title_wrapped"]]
    assert [record["title"] for record in wrapped] == [
        "Accel Adv Alg:Concept&Connect",
        "Accel Geo B:Concept&Connect",
        "Intro to Business and Tech",
        "Literature & Composition I Hnr",
    ]
    for record in parsed["course_records"]:
        lines = record["evidence"]["title_lines"]
        assert len(lines) == (2 if record["title_wrapped"] else 1)
        for line in lines:
            assert text[line["start"]:line["end"]] == line["text"]
        assert record["title"] == " ".join(" ".join(line["text"].split()) for line in lines)


@pytest.mark.skipif(not TERM_TABLE.exists(), reason="term-block transcript fixture is unavailable")
def test_every_term_block_value_is_a_literal_slice_of_the_source():
    text = _term_table_text()
    parsed = parse_transcript(text)
    for record in parsed["course_records"]:
        for name in ("course_code", "mark", "weight", "credit", "term"):
            span = record["evidence"][name]
            assert span is not None, name
            assert text[span["start"]:span["end"]] == span["text"]
        assert record["mark"] == str(record["mark_percent"])


@pytest.mark.skipif(not TERM_TABLE.exists(), reason="term-block transcript fixture is unavailable")
def test_the_term_banner_names_the_term_and_not_the_school():
    """The banner above the rows is the term; it is not a school name and not a section."""
    parsed = parse_transcript(_term_table_text())
    summary = summarize(parsed)

    assert summary["school"] is None
    assert summary["sections"] == []
    assert summary["terms"] == ["2025-2026 Grade 09 Term 2"]
    # The page prints the banner twice; the one directly above the rows owns them.
    assert [term["records"] for term in parsed["terms"]] == [0, 8]
    assert {record["term"] for record in parsed["course_records"]} == {"2025-2026 Grade 09 Term 2"}


def test_term_block_without_a_labelled_footer_invents_no_total_and_no_gpa():
    text = "\n".join([
        "2025-2026 Grade 09   Term 2",
        "Course        Mark  Weight  Credit",
        "Algebra 1      94  1.0000       1",
        "",
    ])
    parsed = parse_transcript(text)
    standing = parsed["academic_standing"]
    assert parsed["checks"]["course_record_count"] == 1
    assert standing["total_credits"] is None
    assert standing["gpa"] is None
    assert parsed["checks"]["credit_sum_matches_reported_total"] is None


def test_a_labelled_line_below_a_row_is_not_read_as_the_rest_of_its_title():
    text = "\n".join([
        "2025-2026 Grade 09   Term 2",
        "Course        Mark  Weight  Credit",
        "Algebra 1      94  1.0000       1",
        "Reading: Chapters 1-3",
        "Credit: 1.000   GPA: 4.00000",
        "",
    ])
    parsed = parse_transcript(text)
    assert [record["title"] for record in parsed["course_records"]] == ["Algebra 1"]
    assert parsed["course_records"][0]["title_wrapped"] is False
    assert parsed["academic_standing"]["gpa"]["value"] == pytest.approx(4.0)


@pytest.mark.skipif(not TERM_TABLE.exists(), reason="term-block transcript fixture is unavailable")
def test_the_same_term_block_survives_a_text_layer_that_collapsed_its_columns():
    """A PDF text layer that loses the column gaps must give the same eight rows."""
    aligned = parse_transcript(_term_table_text())
    collapsed = parse_transcript(re.sub(r"[ \t]{2,}", " ", _term_table_text()))
    key = lambda parsed: [
        (record["course_code"], record["title"], record["mark_percent"],
         record["weight"], record["credit"])
        for record in parsed["course_records"]
    ]

    assert len(collapsed["course_records"]) == 8
    assert key(collapsed) == key(aligned)
    assert collapsed["checks"]["credit_sum_matches_reported_total"] is True
    assert collapsed["academic_standing"]["gpa"]["value"] == pytest.approx(4.13333)


def test_a_subject_heading_below_a_row_is_not_read_as_the_rest_of_its_title():
    text = "\n".join([
        "2025-2026 Grade 09   Term 2",
        "Course        Mark  Weight  Credit",
        "Algebra 1      94  1.0000       1",
        "MATH",
        "Geometry       91  1.0000       1",
        "Credit: 2.000   GPA: 3.50000",
        "",
    ])
    parsed = parse_transcript(text)
    assert [record["title"] for record in parsed["course_records"]] == ["Algebra 1", "Geometry"]
    assert parsed["checks"]["computed_total_credits"] == 2.0
