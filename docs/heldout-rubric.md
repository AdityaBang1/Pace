# Held-out annotation rubric (written before scoring)

This page is the rubric for a new value-level gold set built to answer a question
the existing numbers cannot: **how accurate is the extractor on documents that
were not used to tune it?** It was written before any extractor output on these
documents was produced. The label file it describes was frozen (SHA-256 recorded
in `data/processed/field/heldout_selection.json`) before `pace.field_heldout_eval`
was run, and no rule in `pace/field_extract.py` was changed for it.

## Which documents, and the caveat that comes with them

The whole syllabus corpus (`data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl`)
has 90 documents. 47 are development, 16 are test (14 scored plus 2 excluded),
and 27 belong to two institutions that are in **neither** split:

- University of California, Berkeley (17 documents)
- Georgia Institute of Technology (10 documents)

They are outside both splits for a reason recorded in
[`extract_split.json`](data/processed/field/extract_split.json): they are the two
schools whose documents were read while writing the **very first** rules, so the
split protocol removes them entirely rather than calling them held-out. They are
therefore not untouched by the project's earliest title/code/credit rules. They
have not been read in this pass, and no rule added after the split was tuned on
them, but **a number on this set is an upper bound, not a clean estimate of
accuracy on never-seen institutions.** The report must say that next to the
number.

To keep annotation cost bounded, the set is every eligible document whose cleaned
text is **at most 15,000 characters** (20 of the 27). The seven longer documents
are excluded; that is a length filter, not a quality or difficulty filter, and it
is a limitation stated with the result. No document was selected by looking at
extractor output.

Selection is written to
[`heldout_selection.json`](data/processed/field/heldout_selection.json) with the
school, character count and SHA-256 of the frozen label file.

## Fields and what counts as a value

Seven fields are labelled, matching the existing value-level gold so the numbers
are comparable:

| Field | A value is | Matching rule (already implemented in `pace/field_gold.py`) |
|---|---|---|
| `course_title` | The document's own name for the course, as printed, without the term or section number. | Substring after normalisation; a title carrying the code still matches a bare name. |
| `course_code` | The catalogue code exactly as printed (`CS 10`, `BIOLOGY 1A`). | Alphanumeric compaction, so `CS10` and `CS 10` agree. |
| `credits` | A printed credit/unit count, as a number. Points are not credits. | Numeric equality. |
| `instructor` | The instructor(s) of record, full name as printed. Teaching assistants are not instructors. | Surname containment. |
| `meetings` | Each regular lecture/lab/discussion slot that prints days and a clock time. Exams, office hours and "TBA" slots are not meetings. | Exact day set and clock times; room compared only when the document prints one. |
| `grading` | Each **top-level** graded component with a single printed percentage (or points out of 100). Nested sub-weights inside a parent category and ranges (`0-1%`, `25% lowest`) are not top-level rows. | Weight equality plus category-token overlap. |
| `office_hours` | Each availability statement: days/times, a place where students can meet the instructor, or "by appointment". A heading alone is not a value. | All gold content tokens must occur in the emitted text. |

`absent` lists the fields among those seven that the document does **not** state
anywhere. An emitted value on an absent field is a false positive. If a field is
stated but hard to pin down, it is labelled with the best reading and explained
in `notes`; it is not marked absent.

## Rules applied while reading

1. Label only what the document states. Do not infer a code from a title, a
   credit count from a programme, or a meeting time from a calendar that is not
   in the text.
2. A value must be findable again in the document. `notes` can quote the line.
3. Section headings are not titles; a room number is not a course code; an
   instructor's office number is not a meeting room.
4. Grading follows the existing published rule: top-level breakdown only. If the
   document prints `Exams 60% (Midterm 30% + Final 30%)`, the gold is the single
   `Exams 60` row, not the two children.
5. A meeting with days and a room but no clock time is **not** labelled as a
   meeting (the extractor's day-only case is reported there, not here).
6. Office hours stated as "TBD", "see Canvas", or only as a heading are not
   labelled.
7. If two readings are defensible, take the literal printed one and note the
   alternative.

## How the labels were kept out of the rules

- The rubric above was written first.
- Documents were read from `clean_html` text only; the extractor was not run on
  them until the label file was complete.
- Every document was labelled in ID order, then re-read once against its own
  text before the file was frozen.
- The freeze hash is recorded before scoring; the scoring command refuses a gold
  file whose hash does not match the one in `heldout_selection.json` unless
  `--allow-changed-gold` is passed (used only while annotating, never for the
  reported run).
- No rule, threshold or cap in `pace/field_extract.py` was edited in this pass.

## What the number can and cannot say

Precision is computed over **emitted** values that the gold can judge; recall is
reported separately against the labelled items. With *N* judged values and zero
errors, the one-sided 95% Clopper–Pearson upper bound on the error rate is
`1 - 0.05^(1/N)`, so the precision lower bound is `0.05^(1/N)`. `N = 149` is the
smallest clean sample whose bound reaches 0.98 (0.98010; at `N = 148` it is
0.97996). If this set produces fewer than 149 judged values, the report states
the count and the bound it actually supports. A result below 0.98 is reported as
measured; nothing is tuned to raise it.

## Corrections made while annotating (no extractor output involved)

Two decisions were fixed in writing while the labels were being read, before the
extractor was run on any of these documents. Neither was informed by extractor
output, and neither changed which documents were eligible.

1. The interval formula above was first written as `1 - 0.05^(1/N)` and called
the *precision* bound. That expression is the error-rate bound; the precision
bound is its complement. The arithmetic (and the sample size it implies) is
corrected above. The correction only renames which side of the interval the
formula describes.
2. The `grading` definition says a value is a component with "a single printed
percentage (or points out of 100)". Three documents print a point scheme out of
a larger total (1000, 405, 400). Because dropping a plainly printed breakdown
would mark a stated field as absent, those components are labelled with the
points as printed (e.g. 180 of 1000), and each such document says so in `notes`.
This is a reading of the document, not of the extractor: it records what is
printed, and it does not widen the extractor's own sum-to-100 gate.

