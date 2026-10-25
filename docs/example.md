# Worked example: one syllabus in, one record out

This page is the whole path through the extractor on a single, deliberately
short document: the input file verbatim, the one command that reads it, the
record that command prints verbatim, and a walkthrough of every part of that
record — including the entries that record what the extractor *did not* emit.

It is a demonstration, not a measurement. See
[what this example does and does not show](#what-this-example-does-and-does-not-show)
at the end.

## The input

[`data/samples/example_syllabus.txt`](../data/samples/example_syllabus.txt) is a
short synthetic syllabus (18 non-blank lines). It is a near neighbour of the longer
[`mock_syllabus.txt`](../data/samples/mock_syllabus.txt) used elsewhere in the
docs: same shape, fewer sections, and two deliberate repeats — a duplicated
grading row and a second Office Hours section — so the record's loss ledger has
something real to show.

```text
MATH 1A: Calculus
Spring 2027 — Section 003

Instructor: Dr. Lena Ortiz
Email: lortiz@example.edu

This is a 4 credit hours course.

Class Meetings
Lecture: TuTh 11:00-12:30 in Evans 10.

Grading
Problem Sets 30%
Midterm Exam 30%
Final Exam 40%
Problem Sets 30%

Office Hours
Mondays 1-2 PM in Evans 330.

Office Hours
Fridays 2-3 PM in Evans 330.

Attendance
Attendance is not graded, but problem sets are due at the start of section.
```

## The command

Run this from the repository root, with the trained field checkpoint present:

```bash
python -m pace.field_extract data/samples/example_syllabus.txt \
  --checkpoint artifacts/field-classifier-minilm --device cpu
```

It reads the text file, cleans it (a no-op here — it is plain text, not HTML),
types its sections with the checkpoint, and prints one JSON record to stdout.
`--output path/to/record.json` writes the same record to a file instead. The
command is deterministic on CPU: the same input and checkpoint print the same
record.

## The output

This is that command's stdout, verbatim (the only edit is line endings):

```json
{
  "course": {
    "course_title": {
      "value": "MATH 1A: Calculus",
      "field": "course_title",
      "confidence": 0.86,
      "rule": "code_bearing_title",
      "evidence": {
        "text": "MATH 1A: Calculus",
        "start": 0,
        "end": 17
      },
      "occurrences": 1,
      "grounded": true
    },
    "course_code": {
      "value": "MATH 1A",
      "field": "course_code",
      "confidence": 0.93,
      "rule": "code_pattern",
      "evidence": {
        "text": "MATH 1A",
        "start": 0,
        "end": 17
      }
    },
    "credits": {
      "value": 4.0,
      "field": "credits",
      "confidence": 0.95,
      "rule": "credit_pattern",
      "evidence": {
        "text": "4 credit",
        "start": 99,
        "end": 131
      }
    },
    "instructor": {
      "value": "Dr. Lena Ortiz",
      "field": "instructor",
      "confidence": 0.95,
      "rule": "instructor_label",
      "evidence": {
        "text": "Dr. Lena Ortiz",
        "start": 45,
        "end": 71
      }
    }
  },
  "grading": [
    {
      "category": "Problem Sets",
      "weight": 30,
      "confidence": 0.95,
      "evidence": {
        "text": "Problem Sets 30%",
        "start": 196,
        "end": 212
      },
      "weights_sum": 100,
      "weights_sum_consistent": true
    },
    {
      "category": "Midterm Exam",
      "weight": 30,
      "confidence": 0.95,
      "evidence": {
        "text": "Midterm Exam 30%",
        "start": 213,
        "end": 229
      },
      "weights_sum": 100,
      "weights_sum_consistent": true
    },
    {
      "category": "Final Exam",
      "weight": 40,
      "confidence": 0.95,
      "evidence": {
        "text": "Final Exam 40%",
        "start": 230,
        "end": 244
      },
      "weights_sum": 100,
      "weights_sum_consistent": true
    }
  ],
  "meetings": [
    {
      "days": [
        "TU",
        "TH"
      ],
      "start_time": "11:00",
      "end_time": "12:30",
      "room": "Evans 10",
      "alternatives": false,
      "confidence": 0.95,
      "evidence": {
        "text": "Lecture: TuTh 11:00-12:30 in Evans 10.",
        "start": 148,
        "end": 186
      }
    }
  ],
  "policies": {
    "office_hours": {
      "value": "Fridays 2-3 PM in Evans 330",
      "field": "office_hours",
      "confidence": 0.9,
      "evidence": {
        "text": "Office Hours",
        "start": 306,
        "end": 319
      },
      "candidates": [
        {
          "value": "Mondays 1-2 PM in Evans 330",
          "confidence": 0.9,
          "section": "Office Hours",
          "status": "overwritten",
          "reason": null
        },
        {
          "value": "Fridays 2-3 PM in Evans 330",
          "confidence": 0.9,
          "section": "Office Hours",
          "status": "chosen",
          "reason": null
        }
      ]
    },
    "attendance_policy": {
      "value": "Attendance is not graded, but problem sets are due at the start of section",
      "field": "attendance_policy",
      "confidence": 0.8,
      "evidence": {
        "text": "Attendance",
        "start": 349,
        "end": 360
      },
      "candidates": [
        {
          "value": "Attendance is not graded, but problem sets are due at the start of section",
          "confidence": 0.8,
          "section": "Attendance",
          "status": "chosen",
          "reason": null
        }
      ]
    }
  },
  "suppressed": [
    {
      "field": "course_title",
      "value": "Spring 2027 — Section 003",
      "reason": "a session or term line",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Instructor: Dr. Lena Ortiz",
      "reason": "a labelled attribute, not a course name",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Email: lortiz@example.edu",
      "reason": "a labelled attribute, not a course name",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Class Meetings",
      "reason": "a session or term line",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Lecture: TuTh 11:00-12:30 in Evans 10",
      "reason": "a session or term line",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Grading",
      "reason": "a section heading",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Office Hours",
      "reason": "a section heading",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Mondays 1-2 PM in Evans 330",
      "reason": "a contact line",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Office Hours",
      "reason": "a section heading",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "course_title",
      "value": "Fridays 2-3 PM in Evans 330",
      "reason": "a contact line",
      "evidence": "text in the first 800 characters"
    },
    {
      "field": "grading",
      "value": "Problem Sets 30%",
      "reason": "duplicate of 'Problem Sets'",
      "evidence": "Problem Sets 30%"
    },
    {
      "field": "meetings",
      "value": "Midterm Exam 30%",
      "reason": "an exam line, not a class meeting",
      "evidence": "Midterm Exam 30%"
    },
    {
      "field": "meetings",
      "value": "Final Exam 40%",
      "reason": "an exam line, not a class meeting",
      "evidence": "Final Exam 40%"
    },
    {
      "field": "meetings",
      "value": "Office Hours",
      "reason": "an office-hours line, not a class meeting",
      "evidence": "Office Hours"
    },
    {
      "field": "meetings",
      "value": "Office Hours",
      "reason": "an office-hours line, not a class meeting",
      "evidence": "Office Hours"
    },
    {
      "field": "office_hours",
      "value": "Mondays 1-2 PM in Evans 330",
      "reason": "a later section supplied the value",
      "evidence": "Office Hours"
    }
  ],
  "sections_typed": 6,
  "min_confidence": 0.0,
  "grading_total_percent": 100,
  "grading_total_consistent": true,
  "source": "data/samples/example_syllabus.txt",
  "checkpoint": "artifacts/field-classifier-minilm"
}
```

## Reading it field by field

**`course`** holds one object per scalar field, and each one is a small claim
rather than just a string:

- `course_title`: `"MATH 1A: Calculus"`, confidence `0.86`, rule
  `code_bearing_title`. The rule says *why* it won: the line carries a catalogue
  code, which a section heading does not. There is a second title candidate in
  `suppressed` (`"Problem Sets 30%"`, score 2.97) — the grading line's shape is
  apparently title-like to the ranker, and the ledger shows that competition
  instead of hiding it.
- `course_code`: `"MATH 1A"` from a `code_pattern`, confidence `0.93`. Its
  `evidence.text` is the whole title line, because the code occurs inside it;
  the offsets point at that line, not at the six characters.
- `credits`: `4.0`, read from the sentence `"This is a 4 credit hours course."`
  — `evidence.text` is the matched fragment `"4 credit"`.
- `instructor`: `"Dr. Lena Ortiz"`, confidence `0.95`, rule `instructor_label`,
  because the line says `Instructor:`.

`evidence.start` and `evidence.end` are character offsets into the cleaned
text, so any value can be found again and argued with. `course_title` also
carries `occurrences` (1) and `grounded` (true): the returned string must appear
verbatim in the document.

A field can also come back empty **on purpose**, and the record says so rather
than leaving a hole. When no title candidate survives — every title-shaped line
names the document rather than the course, or the best line is not a printed
title — `course` has no `course_title` key, and the `suppressed` list carries one
entry with `"value": null`, a reason (`no course name: ...`) and the printed
line that produced it. The refused labels sit above that entry, so the silence
and its cause are both auditable. This sample has a title, so the record above
shows the emitted case only.

**`grading`** is a list of rows, each with `category`, `weight`, `confidence`,
`evidence`, and two derived fields. `weights_sum` is 100 on every row because
each row is annotated with the total of the subset that went into it;
`weights_sum_consistent` is true, which is the same `100 ± 5` rule that sets
`grading_total_consistent` at the bottom of the record. The input deliberately
repeats `Problem Sets 30%`; only the first is emitted, and the second appears in
`suppressed` as `"duplicate of 'Problem Sets'"`. Nothing else is lost here: the
three emitted weights are 30 + 30 + 40 = 100.

**`meetings`** is a list of class slots. This document has one: Tuesday and
Thursday, `11:00`–`12:30`, room `Evans 10`, `alternatives: false` (the line
names one slot, not an either/or pair), confidence `0.95`, evidence the whole
`Lecture:` line. The four grading and office-hours lines that a schedule reader
might trip over are in `suppressed` with reasons (`an exam line, not a class
meeting`, `an office-hours line, not a class meeting`) — the line-level drops
are visible, not silent.

**`policies`** has one entry per policy field the document actually has. There
are two here, not the five in the schema sketch, because a field the document
does not mention is *absent*, not low-confidence:

- `office_hours`: the chosen value is `"Fridays 2-3 PM in Evans 330"`, the
  second Office Hours section. `candidates` lists both sections read for the
  field — the Monday one with `status: "overwritten"`, the Friday one
  `"chosen"`. The overwritten Monday value is also in `suppressed` with the
  reason `"a later section supplied the value"`. The `evidence.text` on the
  entry is the section heading (`"Office Hours"`), not the body sentence; the
  value itself is the body.
- `attendance_policy`: the body sentence, with the trailing period stripped and
  confidence `0.8`. One candidate, `chosen`.

The `candidates` entries carry `"reason": null` for these two statuses; the
human-readable reason for a loss lives on the matching `suppressed` entry. A
candidate that was read and *rejected* (for example a placeholder body like
`TBD`) carries its reason on the candidate itself.

The title ledger also shows why the field keeps its own value: the line that
gave the instructor (`Instructor: Dr. Lena Ortiz`) and the line that gave the
email are both title-shaped, and both are rejected as labelled attributes rather
than accepted as a course name. The meeting slots are rejected the same way, by
shape, so the office-hours value comes from the policy reader and not from the
title reader's leftovers.

**`suppressed`** is the loss ledger, and every entry has the same three parts:
`field`, `value`, `reason`, plus a short `evidence` string. The 19 entries here
fall into five groups:

| Group | Count | Meaning |
|---|---:|---|
| title pre-filters | 10 | Lines in the first 800 characters that were shape-filtered before ranking: three term/schedule lines, two labelled attributes (`Instructor:`, `Email:`), two meeting slots, three headings. |
| title ranking | 3 | Lines that were real title candidates but scored below the chosen one: three grading rows. |
| grading duplicate | 1 | The repeated `Problem Sets 30%` row. |
| meeting lines | 4 | Grading and office-hours lines that a meeting reader saw and declined, with the reason. |
| office-hours overwrite | 1 | The Monday section replaced by the Friday one. |

**The bottom of the record** is bookkeeping: `sections_typed` (6 sections got a
type from the checkpoint), `min_confidence` (`0.0`, so nothing was filtered by a
threshold — and note that a positive threshold would surface here *and* as
`below min_confidence` candidates), `grading_total_percent` (100) with
`grading_total_consistent` (true), and `source` / `checkpoint` echoing the
arguments so a record can be traced back to the run that produced it.

## What this example does and does not show

It shows the interface end to end: what goes in, what comes out, how evidence
and confidence attach to a value, and where the values the extractor declined
are recorded. It shows that the record is wider than the answer — 3 grading
rows, 1 meeting and 2 policy bodies emitted, against 19 ledger entries.

It does **not** show accuracy. One synthetic document is not a sample: it was
written to exercise the record's shape, it is in neither the development nor the
test split, and no accuracy number is re-derived from it here. The measured
figures live in [REPORT.md](../REPORT.md#task-6-the-extractor--from-a-classified-span-to-a-filled-record)
and are the only ones quoted anywhere: the development number is the one every
fix was tuned on, and the test number is the single one-touch evaluation already
recorded there. This page adds no number of its own. The report does carry one
labelled second touch — the same 20 held-out documents scored once after the
title fix — and says there why that number is contaminated by construction and
is not an independent estimate.

**The output above is checked by the suite.**
`tests/test_field_pipeline.py::test_the_worked_example_in_the_docs_is_the_record_the_extractor_emits`
parses the `json` block on this page, runs the extractor on the same sample, and
fails if the two records differ (ignoring only the echoed `source` and
`checkpoint` path strings, which depend on how the command is invoked). So if a
future rule changes this document's output, the test goes red rather than the
docs quietly going stale.
