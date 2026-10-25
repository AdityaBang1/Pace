# Academic field classifier — model card

MiniLM-L6 encoder with a flat 42-way softmax head over academic-document field
types. Trained **only on real institutional text**: university course catalogs,
course schedules, and syllabi from eight public datasets. No synthetic
documents, no paraphrasing, no generated field values.

Every prediction is a proposal with a calibrated percentage and the exact source
text it came from. Nothing is written to a student record.

## Summary

| | |
|---|---|
| Task | classify a piece of academic text into one of 42 academic field types |
| Base model | `sentence-transformers/all-MiniLM-L6-v2` (6 layers) |
| Parameters | 22,729,386 |
| Checkpoint | 90,946,511 B (FP32) |
| Training data | 163,797 real rows / 22,914 real groups / 8 datasets |
| Field types | 42 trained (43rd, `language_of_instruction`, has no real examples) |
| Hardware | 1× RTX 4070, bf16 autocast, 1.66 GB peak GPU memory, 257 s to run 6 epochs (epoch 5 selected) |
| Checkpoints | [model.pt](model.pt) · [metrics.json](metrics.json) · [summary.json](summary.json) |

## Data — every row is real text a university published

| Source | Rows | What it contributes |
|---|---:|---|
| `twangodev/uwcourses` (UW–Madison) | 84,034 | catalog fields + ~700k real class meetings (room, time, type, instructor) |
| `anordkvist/gu-course-syllabus` (Gothenburg) | 46,338 | 11 natural syllabus field passages per course |
| `USC/USC-Course-Catalog` | 15,948 | natural catalog cell values, 13 columns |
| `chaseharmon/MIT_Course_Catalog` | 14,107 | MIT catalog rows: number, title, description, term, level, units, prerequisites, lab |
| `Dev-the-dev91/syllabus-cleaned` | 1,638 | 90 real syllabi from 14 universities with labelled sections and facts |
| `syllabus-finetune` | 907 | span-annotated syllabus entities |
| `iristun/CU-DocBench` | 533 | real document blocks that are **not** field values → the `other` class |
| `alecwatts/umich-course-descriptions` | 292 | UMich course pages incl. the published section table |

Rows whose identical normalised text carries conflicting labels (6,149 of them,
e.g. a bare `TBA`) are **excluded**, because a text-only model cannot recover
the source field from the same string. Every row carries `source`, `source_field`,
`provenance`, `source_id` and a SHA-256 of its text.

**The corpus is capped at 6,000 rows per field type.** 163,797 of 1,405,290
available rows are kept, and 23 of the 42 field types sit exactly on the
ceiling. This keeps the classes trainable, and it is also why the in-distribution
accuracy below should not be read as the accuracy on a naturally distributed
catalog dump: the test set is label-balanced by construction, while the real
distribution is dominated by a handful of catalog columns.

The cap was tested rather than assumed. Uncapped checkpoints (1.14M training
rows, with and without tail oversampling, 3 and 6 epochs) score macro-F1
0.918–0.928 against 0.952 here, and 0.915–0.927 tail accuracy against 0.976,
on identical test rows. Restoring the natural distribution starves the tail
enough to cost more than the extra data buys, so this capped checkpoint remains
the reference model. Raw comparison in
[per_class_comparison.json](../field-calibration/per_class_comparison.json).

Splits are **group-wise**: a group is a course subject, a course, or a document,
so no document contributes to both training and test.

- train 130,257 rows / 18,332 groups
- validation 16,644 rows / 2,291 groups
- test 16,896 rows / 2,291 groups

## Held-out results (in-distribution)

Two seeds, identical recipe; the reported spread is across seeds.

| Metric | Seed 20261019 | Seed 20261020 | Mean ± sd |
|---|---:|---:|---:|
| Test accuracy | 0.98917 | 0.98935 | **0.98926 ± 0.00009** |
| Test macro-F1 | 0.93339 | 0.93180 | **0.93260 ± 0.00079** |
| Test ECE (10 bins) | 0.0024 | 0.0018 | **0.0021** |

Temperature is fitted on the validation split only (T = 0.647), so the
percentages are calibrated while the test split stays untouched.

Risk–coverage on the test split (means across the two seeds): at a 0.90
confidence threshold the model covers 98.97% of rows at 99.37% precision; at
0.99 it covers 94.74% at 99.75%.

## Calibration where it matters: a university the model has never seen

In-distribution calibration is easy to pass and misleading. These are the same
three cross-institution folds reported above, scored with `pace.field_calibration`
(mean confidence against observed accuracy, plus a 95% Clopper–Pearson bound):

| Setting | Rows | Accuracy | Mean confidence | ECE |
|---|---:|---:|---:|---:|
| Held-out groups (in-distribution) | 16,896 | 0.9892 | 0.9898 | **0.0024** |
| Unseen universities, bare value | 1,638 | 0.6484 | 0.9043 | **0.2559** |
| Unseen universities, heading+value | 1,295 | 0.8695 | 0.9703 | **0.1008** |

On unseen universities the bare-value model is overconfident in every bin: the
0.7–0.8 bin claims 75% and delivers 21%. **No threshold reaches 99% precision.**
Its best operating point is 0.995, which accepts 50.3% of rows at 91.5%
precision (95% lower bound 89.7%) and leaves 4.3% of all rows confidently wrong.
The heading+value model at a 0.99 threshold accepts 88.2% of rows at 94.7%
precision (lower bound 93.4%), with 4.7% confidently wrong.

Precision claims have a sample-size floor: zero errors in 300 accepted rows
certifies 99.01% precision; zero errors in 100 certifies only 97.05%.

## Transcripts: not this model's job

A transcript is a table. On 38 parsed course records, bare cells are misread
exactly as the ambiguity predicts — a credit `0.250` becomes `course_code` in
38/38 cases and a grade level `09` becomes `class_section` in 25/38 — and the
vocabulary has no field for a transcript date. Passing the column header fixes
course titles and grade levels but turns `GRADE` into `grading_policy`. Use the
rule-based transcript parser for tabular records; use this model for prose and
headings.

**Read the macro-F1 with its support in mind.** Several field types are genuinely
rare in real documents, and macro-F1 over 42 classes is dominated by the common
ones:

| Field type | test rows | F1 (seed 20261019) |
|---|---:|---:|
| `drop_add_deadline` | 0 | 0.000 |
| `attendance_policy` | 2 | 0.667 |
| `materials` | 4 | 0.727 |
| `exam_schedule` | 5 | 0.600 |
| `course_policies` | 7 | 0.600 |
| `course_content` | 581 | 0.966 |
| `assessment` | 616 | 0.969 |

A class with 0 or 2 test rows is not measured, it is merely enumerated. The
honest summary is: **the model is reliable on the field types that actually
occur often in catalogs and syllabi, and unmeasured on the handful that do
not.**

Most frequent confusions (seed 20261019): `additional_information → course_evaluation` (20),
`learning_outcomes → course_description` (14), `prerequisites → course_code` (8),
`teaching_form → assessment` (8), `teaching_form → course_content` (8).

## The result that matters: it does not generalise to a new university

The in-distribution number above is real but narrow: the model has seen every
source's layout during training. Two harder evaluations were run.

### Leave-one-source-out — a whole dataset removed from training

| Held out | Rows | Accuracy | Macro-F1 | Accuracy on rows whose label had *some* training support |
|---|---:|---:|---:|---:|
| `chaseharmon` (MIT) | 14,107 | 0.6294 | 0.1832 | 0.6294 |
| `USC` | 15,948 | 0.4643 | 0.1672 | 0.7181 |
| `twangodev` (UW) | 84,034 | 0.4104 | 0.1965 | 0.5223 |
| `Dev-the-dev91` (syllabi) | 1,638 | 0.2149 | 0.1329 | 0.2495 |
| `anordkvist` (GU) | 46,338 | 0.1054 | 0.0585 | 0.2903 |

The last column removes rows whose field type had **zero** training examples
after the holdout — for GU that is 29,509 of 46,338 rows (64%), because GU is
the only source of `confirmation`, `course_level`, `entry_requirements`,
`teaching_form`, `assessment`, `grades` and `course_evaluation`. Even with
those removed, accuracy is 29.0%.

**The model learned each institution's layout, not the field semantics.** When
GU is removed, 5,387 `entry_requirements` rows, 5,246 `course_content` rows and
4,812 `teaching_form` rows all get read as `course_description`: the model
learned "long prose = course_description" from the one source that had that
shape, and applies it everywhere.

### Cross-institution — whole universities held out

Universities are dealt into three disjoint folds. Each fold trains its own
model on the remaining universities and is tested only on the held-out ones.
The early-stopping epoch is chosen on a slice of the *training* universities,
never on the held-out ones.

| Fold | Held-out universities | Test rows | Accuracy | Macro-F1 | ECE |
|---|---|---:|---:|---:|---:|
| 0 | Berkeley, Ohio State, MIT, Penn State, UCLA | 592 | 0.7027 | 0.4332 | 0.217 |
| 1 | Georgia Tech, UT Austin, Wisconsin–Madison, Stanford, UNC | 594 | 0.5707 | 0.4059 | 0.329 |
| 2 | Purdue, Illinois, Carnegie Mellon, Michigan | 452 | 0.6792 | 0.4004 | 0.214 |

Pooled: **accuracy 0.6509 ± 0.0575, macro-F1 0.4131 ± 0.0143** over 1,638 real
syllabus rows from universities the model has never seen. Against 0.989
in-distribution, accuracy on **real syllabi from universities the model has
never seen is 0.57–0.70**, and the calibration is much worse (ECE 0.21–0.33
versus 0.002). The percentages the model reports on unfamiliar documents are
overconfident, and that is the single most important limitation here.

## What actually carries the signal: the heading, not the value

On 446 real labelled sections from 83 real syllabi, the section heading itself
names the field. That gives a model-independent truth to measure against.
Three inputs were compared ([context_eval_cascade.json](../field-ood/context_eval_cascade.json)):

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| heading only (`"Office Hours"`) | **0.8812** | 0.7428 |
| body only (`"Wednesdays 3-4 PM"`) | 0.6928 | 0.5006 |
| heading + body, naive concatenation | 0.7578 | 0.5509 |

Two things follow. First, **the heading is the transferable signal**: 88.1%
from the heading alone versus 69.3% from the body alone. Second, **naive
concatenation is worse than the heading alone** (0.7578 < 0.8812), because the
body dominates the token budget and pulls the prediction away from the heading.

A second model was trained on heading-plus-value pairs ("Office Hours:
Wednesdays 3-4 PM") so that it learns to *use* the heading instead of being
dominated by the value ([field-classifier-context](../field-classifier-context)).
It scores 0.8453 / macro-F1 0.6920 on heading+body — clearly better than
naive concatenation with the first model, and it shows the model is reading
the heading. Combining them as a cascade — classify the heading, and fall back
to the context model only when the heading's calibrated confidence is low —
gives:

| Fallback threshold | Fallback rate | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 0.00 (never fall back) | 0.0% | 0.8812 | 0.7428 |
| 0.70 | 4.7% | 0.8901 | 0.7769 |
| **0.80** | **7.4%** | **0.8879** | **0.7788** |
| 0.90 | 10.3% | 0.8946 | 0.7753 |
| 1.01 (always fall back) | 100.0% | 0.8453 | 0.6920 |

**Recommendation: read the document's structure, classify the heading, and use
the body only as a fallback when the heading is unfamiliar.** That is the
deployment recipe the measurements support, and it is better than any single
model.

### It also generalises to unseen universities

A second model was trained on the heading-aware corpus (75,015 rows, every
one carrying the heading that introduces it) and run through the *same* three
cross-institution folds, read from the *same* fold plan
(`data/processed/field/fold_plan.json`):

| Fold | bare-value acc | heading+value acc | bare macro-F1 | heading+value macro-F1 | bare ECE | heading+value ECE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.7027 | **0.8300** | 0.4332 | **0.6483** | 0.217 | **0.147** |
| 1 | 0.5707 | **0.8917** | 0.4059 | **0.6952** | 0.329 | **0.069** |
| 2 | 0.6792 | **0.8886** | 0.4004 | **0.6783** | 0.214 | **0.087** |
| mean | 0.6509 | **0.8701** | 0.4131 | **0.6739** | 0.253 | **0.101** |

Because the two corpora are different sizes (the heading-aware corpus drops
rows with no heading and duplicate heading/value pairs), the comparison was
repeated on exactly the rows they share — 1,326 rows, paired per fold:

| Fold | paired rows | bare acc | heading+value acc | bare macro-F1 | heading+value macro-F1 |
|---|---:|---:|---:|---:|---:|
| 0 | 466 | 0.6545 | **0.8348** | 0.3760 | **0.6508** |
| 1 | 485 | 0.4928 | **0.8928** | 0.3666 | **0.6958** |
| 2 | 375 | 0.6240 | **0.8907** | 0.3494 | **0.7079** |
| **mean** | **1,326** | **0.5904** | **0.8727** | **0.3640** | **0.6849** |

**On identical rows the heading-aware model is +28.2 accuracy points and +32.1
macro-F1 points better on universities it has never seen.** See
[paired_comparison.json](../field-ood/paired_comparison.json).

## Intended use and limits

- **Intended:** labelling the sections of an academic document (syllabus, course
  page, catalog entry) so a human can confirm them; routing a text to the right
  parser; flagging which fields a document is missing.
- **Recommended use:** read the document's structure and classify the
  *heading*, falling back to the heading+value model only when the heading's
  calibrated confidence is below 0.8. `pace.cli classify-section` implements
  exactly that. Do not use this checkpoint to classify loose values on their
  own — that is the case that does not transfer.
- **Not intended:** deciding a student's grade, credit, or admission outcome;
  reading scanned or handwritten documents (no OCR in this pipeline); parsing
  transcripts; anything that requires the field to be right without a human
  check.
- **Calibration scope:** the temperature is fitted on in-distribution
  validation rows. On documents from an unseen university the reported
  percentages are overconfident (ECE 0.21–0.33 for the bare-value model, 0.07–0.15
  for the heading-aware model). Treat a percentage as meaningful only when the
  source document resembles the training sources.
- **Language:** Latin script only. Non-Latin text was filtered out of the
  corpus and is not handled.
- **The `other` class** lets the model decline on real document blocks that are
  not field values, but it is only 533 rows of evidence and is not a reliable
  "this is not a field" detector.
- **Rights:** six of the eight corpus sources declare **no reuse licence**
  (one declares `other`). Public downloadability is not permission. This
  checkpoint and its training corpus are local research artifacts: do not
  publish, redistribute or commercialise them without per-source clearance,
  and strip the personal fields (`instructor`, `contact_information`) before
  any release. See
  [docs/research.md](../../docs/research.md#real-text-field-classifier-corpus-additional-sources-and-their-rights).

## Reproduction

```bash
python -m pace.field_splits --output data/processed/field --rebuild
python -m pace.field_train \
  --train data/processed/field/group/train.jsonl \
  --validation data/processed/field/group/validation.jsonl \
  --test data/processed/field/group/test.jsonl \
  --output artifacts/field-classifier-minilm --epochs 6 --seeds 20261019 20261020
python -m pace.field_ood --checkpoint artifacts/field-classifier-minilm --output artifacts/field-ood
python -m pace.field_cross --output artifacts/field-crossinstitution --folds 3 \
  --fold-plan data/processed/field/fold_plan.json
python -m pace.field_context --output data/processed/field_context
python -m pace.field_context_eval --checkpoint artifacts/field-classifier-minilm \
  --context-checkpoint artifacts/field-classifier-context
python -m pace.field_pair
python -m pace.field_demo --limit 3
```

Inference:

```bash
python -m pace.cli classify-field "Office Hours: Mon 2-4pm" \
  --checkpoint artifacts/field-classifier-minilm
python -m pace.cli classify-section --heading "Office Hours" --value "Wednesdays 3-4 PM" \
  --checkpoint artifacts/field-classifier-minilm \
  --context-checkpoint artifacts/field-classifier-context
```

## Provenance

See [data/inventory.json](../../data/inventory.json),
[data/manifest.json](../../data/manifest.json) and
[docs/research.md](../../docs/research.md) for download provenance, licence
notes per dataset, and the rights caveats that apply to redistributing any of
them. Instructor names and public institutional email addresses appear in the
training rows because they are the published content of a public course guide;
nothing here is a student record.
