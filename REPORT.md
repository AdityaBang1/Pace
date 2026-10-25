# PaceDecide — real-data experiment report

Updated 2026-10-21. Two real-data tracks: the original catalog-cell pipeline (Tasks 1–4 below) and the newer **academic field classifier** (Task 5), which is trained on 163,797 real rows from eight public datasets and evaluated with leak-free group splits, leave-one-source-out and cross-institution protocols.

## Task 5: academic field classifier (real text, 42 field types)

**What it is.** A MiniLM-L6 encoder with a flat 42-way head that labels a piece of academic text with the field it came from — `course_title`, `credits`, `office_hours`, `exam_schedule`, `attendance_policy`, `class_meeting_room`, `academic_integrity` and 35 more — and reports a calibrated percentage for each. The taxonomy is a harmonisation of eight real schemas into one vocabulary, so the model has to learn the field rather than one institution's column name.

Everything in the training corpus is text a university published: course catalogs, class-meeting schedules, and syllabi. No synthetic documents, no paraphrasing, no generated field values.

### Data

| Source | Rows | What it contributes |
|---|---:|---|
| `twangodev/uwcourses` (UW–Madison) | 84,034 | catalog fields + ~700k real class meetings (room, time, type, instructor, capacity, modality) |
| `anordkvist/gu-course-syllabus` (Gothenburg) | 46,338 | 11 natural syllabus field passages per course |
| `USC/USC-Course-Catalog` | 15,948 | natural catalog cell values, 13 columns |
| `chaseharmon/MIT_Course_Catalog` | 14,107 | class number, title, description, term, level, units, prerequisites, lab |
| `Dev-the-dev91/syllabus-cleaned` | 1,638 | 90 real syllabi from 14 universities, labelled sections and facts |
| `syllabus-finetune` | 907 | span-annotated syllabus entities |
| `iristun/CU-DocBench` | 533 | real document blocks that are **not** field values → the `other` class |
| `alecwatts/umich-course-descriptions` | 292 | UMich course pages incl. the published section table |

163,797 rows / 22,914 groups, selected from 1,405,290 available rows in two steps: 6,149 rows are dropped where identical text carries conflicting labels (a bare `TBA` in two columns cannot be resolved from text alone), and each field type is capped at 6,000 rows — 23 of the 42 types sit exactly on that ceiling. That cap is why the in-distribution accuracy below is as high as it is: it is measured on a label-balanced subsample, not on the corpus's natural distribution, which is 1.4M rows dominated by catalog columns. Splits are group-wise — a group is a course subject, a course, or a document — so no document contributes to both training and test: **130,257 / 16,644 / 16,896** rows for train / validation / test.

**Rights.** Six of the eight new sources declare **no licence** (one declares `other`). Public downloadability is not permission. The corpus and checkpoints are local research artifacts: do not publish or commercialise them without clearing rights per source, and strip the personal fields (`instructor`, `contact_information`) before any release. Details in [docs/research.md](docs/research.md#real-text-field-classifier-corpus-additional-sources-and-their-rights). This is a deliberate change from the earlier "inspect only" rule for the GU corpus, recorded so the trade-off is visible.

### In-distribution result (group-held-out test, two seeds)

| Metric | Mean ± sd | 
|---|---:|
| Accuracy | **0.98926 ± 0.00009** |
| Macro-F1 | **0.93260 ± 0.00079** |
| ECE (10 bins) | **0.0021** |

Temperature 0.647, fitted on validation only. At a 0.90 confidence threshold the model covers 98.97% of test rows at 99.37% precision (mean of the two seeds). Model: 22.7M parameters, 90.9 MB checkpoint, 1.66 GB peak GPU memory, 257 s to run 6 epochs on an RTX 4070 (epoch 5 selected by validation macro-F1).

**Read that macro-F1 with its support.** Field types are genuinely long-tailed: `drop_add_deadline` has 0 test rows, `attendance_policy` 2, `materials` 4, `exam_schedule` 5, `course_policies` 7 — those classes are *enumerated*, not measured. The common types are solid (seed-averaged F1 0.965 for `course_content` on 581 rows, 0.967 for `assessment` on 616). The honest statement is that the model is reliable on the fields that actually occur often, and unmeasured on the handful that do not.

### The result that matters: it does not generalise to a new university

The 0.989 above is narrow — every source's layout was in the training set. Two harder protocols were run.

**Leave-one-source-out** (a whole dataset removed from training, then evaluated on it):

| Held out | Rows | Accuracy | Macro-F1 | Accuracy on rows whose label still had training support |
|---|---:|---:|---:|---:|
| `chaseharmon` (MIT) | 14,107 | 0.6294 | 0.1832 | 0.6294 |
| `USC` | 15,948 | 0.4643 | 0.1672 | 0.7181 |
| `twangodev` (UW) | 84,034 | 0.4104 | 0.1965 | 0.5223 |
| `Dev-the-dev91` (syllabi) | 1,638 | 0.2149 | 0.1329 | 0.2495 |
| `anordkvist` (GU) | 46,338 | 0.1054 | 0.0585 | 0.2903 |

The last column drops rows whose field type had *zero* training examples after the holdout — for GU that is 64% of the rows, because GU is the only source of `confirmation`, `course_level`, `entry_requirements`, `teaching_form`, `assessment`, `grades` and `course_evaluation`. Even so, accuracy is 29.0%.

**The model learned each institution's layout, not the field semantics.** With GU removed, 5,387 `entry_requirements` rows, 5,246 `course_content` rows and 4,812 `teaching_form` rows are all read as `course_description`: the model learned "long prose = course_description" from the single source with that shape and applies it everywhere.

**Cross-institution** (whole universities held out; 14 universities dealt into 3 disjoint folds, each fold training its own model, the early-stopping epoch chosen on a slice of the *training* universities):

| Fold | Held-out universities | Test rows | Accuracy | Macro-F1 | ECE |
|---|---|---:|---:|---:|---:|
| 0 | Berkeley, Ohio State, MIT, Penn State, UCLA | 592 | 0.7027 | 0.4332 | 0.217 |
| 1 | Georgia Tech, UT Austin, Wisconsin–Madison, Stanford, UNC | 594 | 0.5707 | 0.4059 | 0.329 |
| 2 | Purdue, Illinois, Carnegie Mellon, Michigan | 452 | 0.6792 | 0.4004 | 0.214 |

Pooled: **accuracy 0.6509 ± 0.0575, macro-F1 0.4131 ± 0.0143 over 1,638 real syllabus rows from universities the model never saw.** ECE rises from 0.002 in-distribution to 0.21–0.33, so the percentages are overconfident on unfamiliar documents. That is the most important limitation in this report.

### What actually carries the signal: the heading, not the value

446 real labelled sections from 83 real syllabi, scored against the truth implied by the section heading itself (a model-independent label):

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| heading only (`"Office Hours"`) | **0.8812** | 0.7428 |
| body only (`"Wednesdays 3-4 PM"`) | 0.6928 | 0.5006 |
| heading + body, naive concatenation | 0.7578 | 0.5509 |

The heading is the transferable signal, and **naive concatenation is worse than the heading alone** because the body dominates the token budget.

A second model was therefore trained on heading-plus-value pairs (`"Office Hours: Wednesdays 3-4 PM"`) so it learns to *use* the heading rather than be dominated by the value. Scored on the same 446 sections it reaches 0.8453 / macro-F1 0.6920 on heading+body — better than naive concatenation, and evidence that it is reading the heading.

**And on unseen universities it is much stronger.** The same three cross-institution folds, same held-out universities, same recipe (the fold plan is a single file both runs read):

| Fold | bare-value acc | heading+value acc | bare macro-F1 | heading+value macro-F1 | bare ECE | heading+value ECE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.7027 | **0.8300** | 0.4332 | **0.6483** | 0.217 | **0.147** |
| 1 | 0.5707 | **0.8917** | 0.4059 | **0.6952** | 0.329 | **0.069** |
| 2 | 0.6792 | **0.8886** | 0.4004 | **0.6783** | 0.214 | **0.087** |
| mean | 0.6509 | **0.8701** | 0.4131 | **0.6739** | 0.253 | **0.101** |

**Reading the document's structure, rather than classifying loose values, is what makes the model transfer to a university it has never seen: +22 accuracy points and +26 macro-F1 points, and calibration improves from ECE 0.25 to 0.10.**

The two corpora are not the same size (the heading-aware corpus drops rows with no heading and duplicate heading/value pairs: 1,295 test rows against 1,638), so the comparison was repeated on **exactly the rows both corpora share** — 1,326 rows, paired per fold, same rows, same models:

| Fold | paired rows | bare acc | heading+value acc | bare macro-F1 | heading+value macro-F1 |
|---|---:|---:|---:|---:|---:|
| 0 | 466 | 0.6545 | **0.8348** | 0.3760 | **0.6508** |
| 1 | 485 | 0.4928 | **0.8928** | 0.3666 | **0.6958** |
| 2 | 375 | 0.6240 | **0.8907** | 0.3494 | **0.7079** |
| **mean** | **1,326** | **0.5904** | **0.8727** | **0.3640** | **0.6849** |

On identical rows: **+28.2 accuracy points and +32.1 macro-F1 points** for the heading-aware model. Raw numbers in [paired_comparison.json](artifacts/field-ood/paired_comparison.json).

### Deployment recipe the measurements support

Classify the heading; fall back to the heading+value model only when the heading's calibrated confidence is low.

| Fallback threshold | Fallback rate | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 0.00 (never) | 0.0% | 0.8812 | 0.7428 |
| 0.70 | 4.7% | 0.8901 | 0.7769 |
| **0.80** | **7.4%** | **0.8879** | **0.7788** |
| 0.90 | 10.3% | 0.8946 | 0.7753 |
| 1.01 (always) | 100.0% | 0.8453 | 0.6920 |

`pace.field_demo` runs this end to end on real syllabi: 27 sections across 3 UC Berkeley CHEM 1A documents, each output carrying the exact source text it was scored from, its predicted field and its percentage. Quoted from [syllabus_demo.json](artifacts/field-ood/syllabus_demo.json), with runs of whitespace collapsed for readability: `Course Materials` → `materials` 99.99%, `Office Hours: Wednesdays 3-4 PM` → `office_hours` 99.99%, `Grading:` → `grading_policy` 99.98%, `Exams:` → `exam_schedule` 99.99%, `Attendance: 75` → `attendance_policy` 99.99%.

The same file shows where it fails, which is why every prediction ships with its source text. `Final Exam` — a heading carrying no course name — goes to `assignments` at 99.84% instead of `exam_schedule`, and the bare inline number `65` under `Homework:` is read as `class_section` at 99.53%. High confidence is not correctness: those are 99.5–99.8%-confident wrong answers, and the ~99% in-distribution accuracy is what hides them.

The same run reports the **heading/body agreement: 0.52** — for 48% of these sections the bare model, given only the text under the heading, names a *different* field than it does when given the heading. That disagreement is the whole story of this report in one number. With the heading-aware checkpoint in the loop the cascade needed **0 fallbacks** on these 27 sections: the headings alone were already confident enough.

```bash
python -m pace.field_demo --limit 3 --context-checkpoint artifacts/field-classifier-context
```

### Do the percentages mean anything?

A percentage is only useful if the number is true. `pace.field_calibration` measures that directly: 10-bin reliability (mean confidence against observed accuracy), ECE, and a threshold sweep with a 95% Clopper–Pearson lower bound, so a precision claim can fail rather than be asserted.

| Setting | Rows | Accuracy | Mean confidence | ECE (10 bins) |
|---|---:|---:|---:|---:|
| In-distribution, held-out groups | 16,896 | 0.9892 | 0.9898 | **0.0024** |
| Unseen universities, bare value | 1,638 | 0.6484 | 0.9043 | **0.2559** |
| Unseen universities, heading+value | 1,295 | 0.8695 | 0.9703 | **0.1008** |

On the familiar distribution the percentages hold. At a 0.90 threshold the model accepts 99.0% of rows at 99.37% precision; at 0.99 it accepts 93.7% at 99.75%, and the 95% lower bound is 99.68% — so **"≥99% precise" is certified there**, with 0.23% of all rows confidently wrong.

On a university it has never seen, the bare-value model's percentages are fiction: it averages 90.4% confidence on 64.8% accuracy, and **no threshold reaches 99% precision**. Its best point is 0.995 — half the rows accepted, 91.5% right (lower bound 89.7%), and 4.3% of all rows confidently wrong. The 0.7–0.8 confidence bin claims 75% and delivers 21%.

The heading-aware model is the only setting that comes close: 87.0% accuracy, ECE 0.101, and at 0.99 it accepts 88.2% of rows at 94.7% precision (lower bound 93.4%), 4.7% confidently wrong. That is the honest ceiling of this checkpoint on a new institution, and it is why the report says *report the number, do not promise 99%*.

Sample size sets the limit, and the bound makes it explicit: zero errors in 300 accepted rows certifies 99.01% precision at 95%; zero errors in 100 certifies only 97.05%. A 99% claim from a 40-row demo means nothing.

### The 6,000-row cap, measured: more data made it worse

The cap was disclosed above as a distortion. It was then tested. The corpus was rebuilt with no ceiling (1,405,290 rows, 26,818 groups) and retrained four ways; every model is scored on the **same 127,596-row test split**, so only the training data differs.

| Model | Train rows | Accuracy | Macro-F1 | Accuracy on classes with ≤200 test rows |
|---|---:|---:|---:|---:|
| Capped (shipped) | 130,257 | 0.9967 | **0.9518** | **0.976** |
| Uncapped, natural distribution | 1,137,159 | 0.9969 | 0.9183 | 0.915 |
| Uncapped + tail oversampled to 3,000, 3 epochs | 1,182,159 | 0.9969 | 0.9283 | 0.927 |
| Uncapped + tail oversampled to 3,000, 6 epochs | 1,182,159 | 0.9972 | 0.9262 | 0.921 |

Restoring the natural distribution floods the head — 232,734 `class_section` rows against 8 `attendance_policy` rows — and the tail F1 falls even with 8.7× the data. Oversampling the tail to 3,000 rows and doubling the schedule recover roughly a third of the loss, not all of it, and they plateau: the 6-epoch run is not better than the 3-epoch one. Overall accuracy barely moves in either direction because the head is easy, which is exactly why accuracy alone hides this.

The capped reference was trained at batch 64 for 6 epochs with two seeds; the uncapped runs used batch 256 (1.14M rows at batch 64 needs ~25 minutes per epoch), one seed, and 3–6 epochs. That is not a perfectly matched recipe, and it is the main caveat on this comparison. It is also why the conclusion rests on three uncapped variants agreeing (0.918, 0.928, 0.926) rather than on one run.

**The cap was doing real balancing work, not hiding an easy win.** It stays, and it is now justified by measurement rather than convenience. The uncapped checkpoints are kept as audit artifacts.

### Transcripts are a different problem, and the classifier fails at them

A transcript is a table, not prose. `pace.field_transcript_audit` takes the parser's own evidence spans — the exact characters each field was read from — and classifies them, bare and again with the column header a person would read above the cell (38 course records, synthetic mock transcript):

| Entity | Expected field | Bare agreement | With column header |
|---|---|---:|---:|
| course title | `course_title` | 0.474 | **1.000** |
| credits | `credits` | 0.000 (`course_code` 38/38) | 0.000 (`program`) |
| mark | `grades` | 0.868 | 0.000 (`grading_policy`) |
| grade level | `course_level` | 0.000 (`class_section` 25/38) | **1.000** |
| date | *no field exists* | — | — |

Two of these are vocabulary defects, not model defects: `grades` holds a mark value while `grading_policy` holds the rules, and a bare `0.250` is indistinguishable from a course number without its column. Headers fix the title and the grade level and break the mark. The honest conclusion is that **the field classifier should not be pointed at lone transcript cells**: the rule-based parser resolves the columns deterministically, keeps evidence spans, and on the sample reconciles the credit sum exactly (13.25 computed = 13.25 reported). Transcript support is delivered by the parser; the field model is for prose and headings.

### Artifacts

| | |
|---|---|
| Model card | [artifacts/field-classifier-minilm/MODEL_CARD.md](artifacts/field-classifier-minilm/MODEL_CARD.md) |
| In-distribution metrics | [artifacts/field-classifier-minilm/metrics.json](artifacts/field-classifier-minilm/metrics.json) |
| Leave-one-source-out report | [artifacts/field-ood/ood_report.json](artifacts/field-ood/ood_report.json) |
| Cross-institution report (bare values) | [artifacts/field-crossinstitution/cross_institution_report.json](artifacts/field-crossinstitution/cross_institution_report.json) |
| Cross-institution report (heading-aware) | [artifacts/field-crossinstitution-context/cross_institution_report.json](artifacts/field-crossinstitution-context/cross_institution_report.json) |
| Paired bare-vs-heading comparison | [artifacts/field-ood/paired_comparison.json](artifacts/field-ood/paired_comparison.json) |
| Heading vs body vs cascade | [artifacts/field-ood/context_eval_cascade.json](artifacts/field-ood/context_eval_cascade.json) |
| Syllabus demo | [artifacts/field-ood/syllabus_demo.json](artifacts/field-ood/syllabus_demo.json) |
| Calibration, held-out groups | [artifacts/field-calibration/capped_test.json](artifacts/field-calibration/capped_test.json) |
| Calibration, unseen universities (bare) | [artifacts/field-calibration/unseen_universities.json](artifacts/field-calibration/unseen_universities.json) |
| Calibration, unseen universities (heading+value) | [artifacts/field-calibration/unseen_universities_context.json](artifacts/field-calibration/unseen_universities_context.json) |
| Capped vs uncapped, same rows | [artifacts/field-calibration/per_class_comparison.json](artifacts/field-calibration/per_class_comparison.json) |
| Transcript coverage audit | [artifacts/field-ood/transcript_coverage.json](artifacts/field-ood/transcript_coverage.json) |
| Shared fold plan | [data/processed/field/fold_plan.json](data/processed/field/fold_plan.json) |

```bash
python -m pace.field_splits --output data/processed/field --rebuild
python -m pace.field_train --train data/processed/field/group/train.jsonl \
  --validation data/processed/field/group/validation.jsonl \
  --test data/processed/field/group/test.jsonl \
  --output artifacts/field-classifier-minilm --epochs 6 --seeds 20261019 20261020
python -m pace.field_ood --checkpoint artifacts/field-classifier-minilm --output artifacts/field-ood
python -m pace.field_context --output data/processed/field_context
python -m pace.field_context_eval --checkpoint artifacts/field-classifier-minilm \
  --context-checkpoint artifacts/field-classifier-context
# both corpora read the same held-out universities from one plan
python -m pace.field_cross --output artifacts/field-crossinstitution --folds 3 \
  --fold-plan data/processed/field/fold_plan.json
python -m pace.field_cross --output artifacts/field-crossinstitution-context --folds 3 \
  --corpus data/processed/field_context/corpus.jsonl \
  --fold-plan data/processed/field/fold_plan.json
python -m pace.field_pair
python -m pace.field_demo --limit 3 --context-checkpoint artifacts/field-classifier-context
# Is the percentage true? Reliability, ECE, and certified operating points
python -m pace.field_calibration --checkpoint artifacts/field-classifier-minilm \
  --data data/processed/field/group/test.jsonl --output artifacts/field-calibration/capped_test.json
python -m pace.field_calibration --folds-root artifacts/field-crossinstitution \
  --corpus data/processed/field/corpus.jsonl --fold-plan data/processed/field/fold_plan.json \
  --output artifacts/field-calibration/unseen_universities.json
python -m pace.field_calibration --folds-root artifacts/field-crossinstitution-context \
  --corpus data/processed/field_context/corpus.jsonl --fold-plan data/processed/field/fold_plan.json \
  --output artifacts/field-calibration/unseen_universities_context.json
# Does more data help? Same rows, several checkpoints
python -m pace.field_compare --data data/processed/field_uncapped/group/test.jsonl \
  --checkpoint capped=artifacts/field-classifier-minilm \
  --checkpoint uncapped=artifacts/field-classifier-uncapped \
  --checkpoint balanced=artifacts/field-classifier-uncapped-balanced
# What the model does with a transcript's cells
python -m pace.field_transcript_audit --context-checkpoint artifacts/field-classifier-context
python -m pace.cli classify-field "Office Hours: Mon 2-4pm"
python -m pace.cli classify-section --heading "Office Hours" --value "Wednesdays 3-4 PM"
```

## Task 6: the extractor — from a classified span to a filled record

**What it is.** The classifier above answers "what field is this text?" for one span at a time. That is not what a person asks for. The extractor is the layer that turns a raw syllabus or course page into a record:

```bash
python -m pace.field_extract path/to/syllabus.txt        # or: - to read stdin
```

It prints, and optionally writes with `--output`, a JSON record:

| Part of the record | Fields |
|---|---|
| `course` | `course_title` (with `grounded`), `course_code`, `credits`, `instructor` |
| `grading` | one row per category: `category`, `weight`, `weights_sum`, `weights_sum_consistent` |
| `meetings` | `days`, `start_time`, `end_time`, `room`, `alternatives` |
| `policies` | `office_hours`, `attendance_policy`, `academic_integrity`, `disability_services`, `materials` |

Every value carries `confidence` and `evidence` — the exact line and character offsets it was read from — so a wrong value can be traced to the text that produced it. The field-typing signal is the existing classifier checkpoint (`artifacts/field-classifier-minilm`, plus `artifacts/field-classifier-context` for heading+body sections); **no new model is trained for this layer**. The model decides *what kind of text this is*; rules decide *where in a line the value sits*.

A complete worked example — one short syllabus, the command that reads it, the record it prints verbatim, and a walkthrough of every emitted field and every suppressed entry — is in [docs/example.md](docs/example.md). It is a demonstration of the interface, not a measurement; the numbers below are unchanged by it.

### The measurement protocol: three gold sets, a development split and a test split

The earlier version of this section quoted 0.875, produced by diagnosing failures on the same institutions it was scored on. Two things changed since. The held-out institutions were split **before any fix**, by institution, and that split is written to [extract_split.json](data/processed/field/extract_split.json); and the evaluation gained a stricter gold set, because the first one could not answer the question the mission actually asks. A third set, on documents from institutions in neither split, was added later to measure accuracy without tuning (its own section below).

| Gold | Documents | What it judges | What it cannot see |
|---|---:|---|---|
| Section-span (published with the corpus) | 14 untouched documents from MIT, Ohio State and UNC | did the value come from the right section | a value on a field the document never states — there is no span, so it is never scored |
| Value-level (read field by field for this pass) | 12 documents from two institutions that were not in the corpus at all | title, code, credits, instructor, meetings (days, times, room), every grading category and weight, office hours — including the fields the document does **not** state | fields outside that named set (policy prose), and grading printed only as points |
| Held-out (read field by field, frozen before scoring) | 20 corpus documents from UC Berkeley and Georgia Tech, in neither split | the same seven fields, on documents no rule added since the split was tuned on | a clean unseen-institution estimate: those schools were read while the first rules were written, so its number is an upper bound |

The second set exists because of a concrete hole. On the 14-document split, 22 of 37 emitted values had no gold and were counted as unjudged; an invented credit count on a document that never prints credits, or a room number returned as a course code, could not be an error because there was nothing to be wrong against. The value-level gold marks absent fields explicitly, so those values are now false positives.

The 12 new documents come from Yale Summer Session and the University of Florida's public syllabus archive. Candidates were sampled evenly from each published listing and kept by a readability gate only (a real text layer, 1,200–20,000 characters) — never by how the extractor later did. Gold was written from the raw document before the extractor was run on it, and no extraction rule has changed since. None of the 12 documents, nor even a nine-word phrase from any of them, appears anywhere in the classifier corpus, so these are the only numbers in this report where neither the rules nor the encoder had seen the text.

Two further corpus documents were removed from the test split because they had already been looked at (`870b9308`, `28359696`); both exclusions are recorded in the split artifact.

```bash
python -m pace.field_extract_eval --write-split                 # the split, before any fix
python -m pace.field_extract_eval --split dev                   # tuning loop, run as often as needed
python -m pace.field_extract_eval --split test                  # run once, after the rules are frozen
python -m pace.field_extract_eval --protocol-report             # the published table
```

### The recorded test number (one touch)

**63 of 66 judged values correct — precision 0.955, 95% lower bound 0.887**, over 12 documents from Yale and Florida. Six emitted values have no gold either way and are counted as unjudged, not as wins. The lower bound is one-sided Clopper–Pearson on the error count. The same sample scored **51/68 = 0.750, bound 0.649** before that run; both runs are on disk, so the change can be re-derived. Rules added in later passes were measured on development only; this table describes the recorded run, not the current code.

| Field | Emitted | Judged | Correct | Precision | Gold items | Found |
|---|---:|---:|---:|---:|---:|---:|
| `course_title` | 8 | 8 | 8 | 1.000 | 12 | 8 |
| `course_code` | 7 | 7 | 7 | 1.000 | 17 | 8 |
| `credits` | 1 | 1 | 1 | 1.000 | 2 | 1 |
| `instructor` | 8 | 8 | 8 | 1.000 | 12 | 8 |
| `meetings` | 2 | 2 | 2 | 1.000 | 12 | 2 |
| `grading` | 33 | 33 | 32 | 0.970 | 45 | 33 |
| `office_hours` | 7 | 7 | 5 | 0.714 | 10 | 5 |

The pattern is not uniform, and the per-field split is the useful part. Every title, code, name, credit and meeting slot it found is correct; `meetings` recall is still 2 of 12 because most of these documents print times in forms the line reader does not pair. `office_hours` went from 0/5 to 5/7: the wrong-body values are gone, and the two that remain are a labelled `F 3:00-4:00` the gold does not accept and a section headed `Class Times` that the classifier typed as office hours. Both are recorded as known failures. This table is the **recorded run**: the office-hours heading and opening-availability rules added in this pass came after it and were measured only on development, so the table describes that run, not the current code, and is not a fresh measurement of the rules added since.

One gold rule still matters: only top-level breakdown items count as grading gold. PHYS S102E's four sub-weights (`written test` 15, `oral test` 5, `online section discussion` 10, `participation` 10) are printed inside parent categories, so the rule scores them as misses. The extractor now keeps them out of the top-level list — a nested breakdown of a category it already returned is suppressed — so this pass prints 33 grading values instead of 36 and the rule is not what separates the two runs.

On the section-span stratum — the narrower documents — the extractor scores **4/7 = precision 0.571, lower bound 0.225**, against 10/15 = 0.667 before. That fall is mostly coverage: the stricter rules emit 21 values there instead of 37, so only 7 still have a span the gold can judge, and four of those are right. A 4/7 sample cannot carry a precise estimate in either direction; the value-level gold is the number this section quotes.

### Reliability: does the percentage mean anything?

| Confidence band | Values | Correct | Precision | 95% lower bound |
|---|---:|---:|---:|---:|
| 0.00–0.70 | 0 | 0 | - | - |
| 0.70–0.90 | 21 | 21 | 1.000 | 0.867 |
| 0.90–0.99 | 45 | 42 | 0.933 | 0.837 |
| 0.99–1.00 | 0 | 0 | - | - |

The bands no longer run *backwards*: before that run the top band was 0.529 and the bottom 0.857, because the percentage was the section-type softmax and it was most sure exactly where it attached the wrong body to a correct heading. Each value's percentage now comes from its own evidence — an explicit `Course Code:`/`Instructor:` label, a code or credit count with the shape the field requires, a day and clock for a meeting, a body whose opening label is not another field — and the ranking follows: **1.000 at 0.70–0.90, 0.933 at 0.90–0.99**.

It is still not monotone, and the numbers say why: the higher band holds 45 values and three errors (`office_hours` ×2 at 0.95, `grading` ×1 at 0.90), while the lower band holds 21 values and none. The two intervals overlap (`0.867` lower bound against `0.837`), so the difference shown is inside what 66 values can resolve. No threshold fixes it either: 0.95 keeps 32 of 66 values and still two of the three errors, and 0.99 keeps nothing at all.

### What a default threshold could buy — and what it cannot

| `min_confidence` | Values kept | Coverage | Precision | 95% lower bound |
|---:|---:|---:|---:|---:|
| 0.000 | 66 | 100% | 0.955 | 0.887 |
| 0.500 | 66 | 100% | 0.955 | 0.887 |
| 0.800 | 57 | 86% | 0.947 | 0.870 |
| 0.900 | 45 | 68% | 0.933 | 0.837 |
| 0.950 | 32 | 48% | 0.938 | 0.816 |
| 0.970 | 1 | 2% | 1.000 | 0.050 |

**No threshold certifies precision 0.95 at 95% confidence on this sample, and no threshold certifies 0.90 either.** The strongest certified bound any setting reaches is **0.887**, at the default of zero. That is a different sentence from the previous pass, where the best bound was 0.649 and every positive threshold measured worse than zero; now the default is the only setting that keeps all 66 judged values, and every positive threshold keeps fewer values while its bound stays below the default's.

The development split was checked first, as the protocol requires: there, too, no threshold certifies 0.95 (best bound 0.849 at `min_confidence=0.0`, on 84 judged values; 0.97 keeps one value, which is not a usable operating point). The shipped default therefore stays at `min_confidence=0.0` — not because it is impressive, but because nothing in the measurement supports moving it.

The mission asked for probabilities that are accurate and a tool that does not emit confident false positives. **The second is now closer to true and the first is still not.** The percentage now orders correctness (1.000 above 0.933, not 0.529 below 0.857), but the two remaining office-hours errors and one grading error sit at 0.90–0.95 with the true values, so the number is not yet a usable filter and 0.955 is not certified at 0.95. The open failures in [known_failures.json](data/processed/field/known_failures.json) are where that would change.

### The development split (for context, not a result)

| Field | Emitted | Judged | Correct | Precision | Recall | Gold rows |
|---|---:|---:|---:|---:|---:|---:|
| `academic_integrity` | 5 | 5 | 5 | 1.000 | 0.720 | 25 |
| `attendance_policy` | 1 | 1 | 1 | 1.000 | 1.000 | 2 |
| `course_code` | 27 | 1 | 1 | 1.000 | 0.250 | 4 |
| `course_title` | 28 | 7 | 7 | 1.000 | 0.778 | 9 |
| `credits` | 9 | 1 | 0 | 0.000 | 0.000 | 3 |
| `disability_services` | 4 | 4 | 4 | 1.000 | 0.857 | 14 |
| `grading` | 70 | 42 | 38 | 0.905 | 0.363 | 179 |
| `instructor` | 15 | 4 | 2 | 0.500 | 0.067 | 30 |
| `materials` | 8 | 5 | 5 | 1.000 | 0.381 | 42 |
| `meetings` | 36 | 0 | 0 | - | 0.000 | 2 |
| `office_hours` | 14 | 14 | 14 | 1.000 | 0.907 | 43 |

Development totals **77/84 = 0.917, lower bound 0.849** — within noise of the previous pass (84/92 = 0.913) on 8 fewer judged values. The coverage loss is real and is the price of the stricter rules: policy fields emit less (materials 26 → 8 emitted, 12 → 5 judged; all 5 correct), because a body must be about the field and must not open another labelled field. Precision was unchanged, so the trade was not free: it bought false positives out of the count by not printing values the span gold used to check.

`credits` reads 0.000 in that table for a gold reason, not an extractor reason: the one judged `credits` span in the dev split is a paragraph about enrolment units, so the correct `4.0` cannot overlap it. `instructor` is the same kind of cell — the dev gold rows for that field are contact lines (`Email:`, `Office: 3078 Derby`), so two correct names score as misses (2/4). The hand check, which asks "is the number right", scored credits 5/5 on 12 dev documents, and meetings 9/11 whole slots, days 11/11, times 9/10, rooms 9/11.

### The held-out measurement: 20 documents from institutions in neither split

The two numbers above cannot answer the question the mission actually asks. Development was tuned on, so 0.915 is a training score. The test split has 66 judged values, and the arithmetic is unforgiving: 66 clean values certify only 0.956, and a 0.98 bound needs 149. So a third gold set was built on documents from the two corpus institutions that belong to **neither** split — 17 UC Berkeley and 10 Georgia Tech documents, of which the 20 under 15,000 cleaned characters were labelled and the 7 longer ones were not (a length filter, not a difficulty filter).

**The number, and the caveat that has to travel with it: 50 of 61 judged values correct — precision 0.820, 95% lower bound 0.719.** The 20 documents emitted 73 values; the gold judges 61 of them (the rest are policy fields the labelled set does not name) and marks the fields each document does *not* state, so a title invented for a page with no title is a false positive here, not silence.

| Field | Emitted | Judged | Correct | Precision | Gold items | Found | Recall | Suppressed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `course_title` | 14 | 14 | 9 | 0.643 | 31 | 18 | 0.581 | 122 |
| `course_code` | 7 | 7 | 6 | 0.857 | 22 | 10 | 0.455 | 7 |
| `credits` | 2 | 2 | 2 | 1.000 | 2 | 2 | 1.000 | 2 |
| `instructor` | 4 | 4 | 4 | 1.000 | 22 | 6 | 0.273 | 4 |
| `meetings` | 7 | 7 | 5 | 0.714 | 12 | 5 | 0.417 | 75 |
| `grading` | 21 | 21 | 20 | 0.952 | 50 | 20 | 0.400 | 18 |
| `office_hours` | 6 | 6 | 4 | 0.667 | 15 | 5 | 0.333 | 3 |
| **All labelled fields** | **73** | **61** | **50** | **0.820** | **154** | **66** | **0.429** | **245** |

The rows sum to the headline. `disability_services` (5 suppressed), `materials` (8) and `attendance_policy` (1) are outside the seven labelled fields, so their losses are in the total but absent from the precision denominator.

**Why it is an upper bound, not a clean estimate.** These are the two schools whose documents were read while writing the very first title/code/credit rules; the split protocol removes them entirely rather than calling them held-out. No rule added after the split was tuned on them and they were not read in this pass, but the earliest rules were shaped on documents from the same institutions. The encoder is not blind to them either: the classifier corpus carries 238 rows from Berkeley and 89 from Georgia Tech, so this measures the *rules* on documents they were not tuned on, not a model that has never read the institution. A number from a genuinely unseen third institution would be expected to come in at or below this one.

**How the labels were kept out of the rules.** The rubric was written first ([docs/heldout-rubric.md](docs/heldout-rubric.md)) and fixes seven fields, what counts as a value, and the handling of absent fields, headings, TBA office hours and top-level-only grading. Every document was then read from `clean_html` text only, in selection order, and re-read against its own text; the extractor was not run on any of them until the file was frozen. The frozen gold is [gold_heldout_fields.json](data/processed/field/gold_heldout_fields.json), SHA-256 `54b0b14f04da…`, recorded in [heldout_selection.json](data/processed/field/heldout_selection.json) **before** scoring. The command refuses to score if the hash differs (`--allow-changed-gold` is for annotating, never for a reported run). Two decisions were fixed in writing during annotation and are disclosed in the rubric: the interval formula had been written as the error-rate bound rather than the precision bound (corrected before scoring, arithmetic only), and three documents print a point scheme out of 1,000, 405 and 400, which are labelled as the points printed rather than dropped as unlabelled. Every label was read by one annotator and no second rater checked them, so the gold itself carries an unmeasured error rate; the 11 wrong values are published individually in the run artifact so a reader can disagree with a specific label rather than with the total.

**No rule changed in the pass that produced this run.** No rule, threshold or cap was edited while this artifact was made: it measures the extractor as it stood then, on documents it was not tuned on. The title rules changed afterwards, measured on development and then scored once on these same documents — that second touch is recorded below, and this artifact is kept frozen as the first touch rather than regenerated. One bug *in the scorer* surfaced and was fixed: `field_gold.compact(None)` returned the string `"none"`, so a gold meeting with no printed room could never match an emitted value that carried a room. It turned exactly one held-out value (a meeting with the right days and times) from wrong to right. It cannot move the recorded split numbers: development judges no meetings at all, and the test split's two judged meetings both matched already.

```bash
python -m pace.field_heldout_eval \
    --output artifacts/field-extract/heldout_eval.json   # the published held-out run
```

That run is a full run (`protocol.truncated: false`); a `--limit` debug run is marked truncated in the artifact, prints the marker, and refuses to overwrite this path. Every emitted value appears once in the artifact's `values` list with the gold entry it matched (or the entries it was compared against, or why none applied), the verdict, the rule that decided it and its evidence, so **50/61 can be recomputed from the file** without re-running the model: the headline, the per-field table and the error list are folds of those rows, not a separate calculation.

**What the 11 errors are.** Five are titles that are a filename, a page artefact, a section heading or a contact line rather than the course's name (`SPAN 1002 Syllabus`, `SPAN 1001 Syllabus`, `Page 1 of 4`, `Collaboration, Not Competition`, `MWF 1-2 pm, O. Hallatschek`). One is a course code read as `LEC 003` (a section label), and one is `grading` `top`, lifted from the phrase "top 25%" in a grade-scale paragraph. Two are class meetings: the Head GSI's office-hours line read as a class time, and a correct `TuTh 2:10-3:30` whose printed room (`1 LeConte Hall`) the extractor dropped, which the exact-slot rule counts wrong. The last two are office hours: the head GSI's availability (`by appointment`, `M/W 10-11am in Bixby North`) — excluded by the rubric because it is staff, not the instructor of record.

That last category is a definitional disagreement, not a fabrication, so here is the sensitivity: if staff availability counted as a correct `office_hours` value, the same run reads 52/61 = **0.852**. The rubric was fixed before the run and is not adjusted after it.

**Recall is the weaker half, and two causes explain most of it.** 66 of 154 labelled items were found (0.43). Five of the 20 documents are Georgia Tech stub pages that print no PDF text — course prefix and number, an instructor name, nothing else — and their labelled items cannot be recovered by any extractor. Policy fields keep one value per document, so a document with four office-hour lines counts as one hit against four gold items. `meetings` gold has 12 slots and only 5 found, mostly because several documents print the lecture line as a single range the reader does not pair, or print rooms it then drops.

**What it threw away.** 245 values were read and suppressed, against 73 emitted. The largest bucket is `course_title` (122): 32 plausible titles that lost the ranking to another line, 20 session/term lines, 19 section headings, 15 contact lines, 15 generic section headings, 12 bare codes, 7 place names, 2 clock times. `meetings` lost 75: 37 exam lines, 30 office-hours lines, 6 day names with no clock or room, 1 weekday outside the group, 1 line with several day groups. `grading` lost 18: 6 duplicates, 4 rows outside the subset that sums to 100, 4 weights whose text is not a category name, 3 documents with only one category, 1 weight with no category beside it. The rest: `course_code` 7, `instructor` 4, `office_hours` 3, `credits` 2, and 14 across the three unlabelled policy fields. Full reasons per field are in the run artifact.

**What this sample can and cannot certify.** It cannot certify 0.98 and is not asked to: 61 judged values bound at most 0.952 even if every one were right, and 149 are needed. Those 61 are the first touch; the second touch below judged 59 and does not change the arithmetic (0.955 would need 66 clean values). What it *can* do is refute the claim. A 0.98 extractor would produce about 1.2 errors in 61 values; this run produced 11, and the probability of seeing 11 or more at that rate is about 3×10⁻⁸. So the honest summary is not "the extractor is 0.98 accurate, unproven"; it is **0.82 on the closest thing to unseen documents available, with a 0.72 floor, which is not 0.98.** The per-field table is where the work is: titles and meetings carry the errors, while credits, instructor and grading are already close to the bar on this sample.

### The second touch: the same 20 documents after the title fix

The five wrong titles in the first run are why the title rules were rewritten. That is exactly the situation the one-touch protocol exists to prevent, so the fix was measured on development first, then scored on the same 20 documents once, after the code was frozen, and written to its own artifact with the contamination recorded in `protocol.note`.

| | First touch (frozen) | Second touch (after the title fix) |
|---|---:|---:|
| Judged values | 61 | 59 |
| Correct | 50 | 50 |
| Wrong | 11 | 9 |
| Precision | 0.820 | 0.847 |
| 95% lower bound | 0.719 | 0.749 |
| Emitted | 73 | 71 |
| Suppressed | 245 | 275 |
| Labelled items found (recall) | 66/154 | 64/154 |

In the second touch, 50 of 59 judged values are correct (precision 0.847, lower bound 0.749); the first touch was 50 of 61 (0.820, 0.719).

**The second number is contaminated by construction and is not an independent estimate.** The title rules were written knowing which held-out title values were wrong — the five strings are named in the first-touch artifact and in this report. A number measured on the same documents that supplied the diagnosis partly remembers the diagnosis. It is a labelled second look, not evidence that the extractor generalises better than 0.820. The first-touch artifact is unchanged and remains the number to quote for "documents no rule was written against"; the 0.847 is what the current code does on documents it has, in effect, been shown the answers for.

What moved, row by row: the five wrong titles (`SPAN 1002 Syllabus`, `SPAN 1001 Syllabus`, `Page 1 of 4`, `Collaboration, Not Competition`, `MWF 1-2 pm, O. Hallatschek`) are now rejections with reasons instead of emitted values, which removes five errors. Three page titles on the Georgia Tech pages (`English Composition II | Syllabus`, `Survey of U.S. History II | Syllabus`, `Intro to Computing | Syllabus`) were correct in the first touch only because the substring rule found the course name inside the page title; they are now rejected as document labels, so they stop counting as matches and those three documents emit no title at all. Four documents gained a correct title once a label could no longer occupy the title slot — three because the document-label line was rejected and the clean title below it won (`Principles of Psychology`, `Physics for Scientists and Engineers`, `Elementary Spanish II, Section OL & SF, 3 credits`), one because the narrowed link filter stopped dropping `Course name: Math 1A LEC 003`. That is why correct stayed at 50 while judged and recall each fell by two.

**The cost, stated plainly.** Three documents whose only title-shaped text was the label line now emit nothing, and three others promote the next candidate into the title slot — schedule text (`Kinematics I`, `Videos: Memento, Christopher Nolan`) or a fragment (`Spanish speakers`). The ledger shows the rejection that let each one through; it does not make the replacement good. Refusing labels is not the same as knowing a page has no title, and this pass did not add a "no title here" decision. The label vocabulary is also a list rather than a parser: `Course name: Math 1A LEC 003` is emitted and counted correct only because the gold string is contained in it, and a line like `Videos: ...` is not on the list at all.

### What the record throws away — the coverage ledger

The extractor used to keep only the winner per field, which made a precision number impossible to audit: an office-hours value could be overwritten by a later, mis-typed section and the overwritten value simply vanished. That is closed. Each policy field now carries `candidates` — every section read for that field, with `status` (`chosen`, `overwritten`, `rejected`, `below min_confidence`) and the reason — and the record carries a top-level `suppressed` list for everything else: a title ranked below the chosen line, a code that looked like a room, a nested sub-weight, a duplicate grading row, a row outside the subset that sums to 100, a credit count read from a requirements clause. `pace.field_ledger.suppressed_values(record)` returns that list; the ledger module defines the shape once, so a rule logs a loss without owning the record's JSON.

The ledger is measured on the development split, the only split allowed to be re-read for it. **217 values emitted, 840 candidates and line-level drops suppressed**, per field. The reason counts under each bullet add up to that field's suppressed total: per-item reasons (`duplicate of 'Homework'`, `component of 'Exams'`) are folded to their rule before counting, so no loss can hide in a one-off label. That check is itself a test — an earlier version of this table listed six title reasons covering 193 of 221 losses, and 28 were invisible until the counts were forced to reconcile.

Nothing is capped any more. An earlier version stopped at 12 rejected codes, 8 rejected titles and 5 title alternatives; measured on 20 room-shaped lines and 20 session lines, it logged 12 and 8 losses. Those caps are gone. The same pass made the pre-filters visible: a grading row whose category cannot be read (eight more on dev), a percentage with a label too long to be one, and every schedule-shaped line `extract_meetings` declines (291 on dev, mostly exam dates and office hours) now appear with their reason. The increase from 380 to 794 is those disclosures, not a change in what the extractor emits: the values that pass emitted were identical as a multiset to the run before it. The title pass moved it from 245 to 299: 43 document labels, 40 labelled attributes and 6 sole candidates below the plausibility floor, offset by the over-broad section rule releasing 21 real titles it had been rejecting (48 → 27). The abstention pass moved the title ledger 299 → 343 (19 abstention notes — 15 "every title-shaped line names something other than the course", 2 "the best line is not a printed title", 2 "a line shaped like a title could not be read cleanly" — 11 reading references and 8 date/week lines refused before they can win a page whose real title sits below them, and 6 reclassifications). The code-evidence pass that followed moved it 343 → 345: three lines now stop at the session/term rule that used to be refused as bare codes, one bare-code refusal fewer.

| Field | Emitted | Suppressed | Gold items | Found | Recall |
|---|---:|---:|---:|---:|---:|
| `meetings` | 36 | 291 | 2 | 0 | 0.000 |
| `course_title` | 28 | 345 | 9 | 7 | 0.778 |
| `course_code` | 27 | 74 | 4 | 1 | 0.250 |
| `grading` | 70 | 72 | 179 | 65 | 0.363 |
| `materials` | 8 | 32 | 42 | 16 | 0.381 |
| `office_hours` | 14 | 9 | 43 | 39 | 0.907 |
| `instructor` | 15 | 6 | 30 | 2 | 0.067 |
| `academic_integrity` | 5 | 8 | 25 | 18 | 0.720 |
| `disability_services` | 4 | 3 | 14 | 12 | 0.857 |
| `attendance_policy` | 1 | 0 | 2 | 2 | 1.000 |
| `credits` | 9 | 0 | 3 | 0 | 0.000 |

Read it as where the precision is purchased, not as a false-positive count — most of the 345 title candidates are headings and document labels that were never names. The real losses are narrower:

- **`meetings` (291 suppressed).** 184 were exam dates (`Final Exams: ...`) and 85 were office-hours lines: schedule-shaped text that is a value of another kind, which is why the field reads so much larger than its 36 emitted meetings. 10 named days with no clock time or room, 7 repeated a slot already emitted, 4 named a weekday outside the group, 1 listed several day groups. The 7 duplicates are the only true losses here; the rest are lines the extractor read and deliberately did not treat as class times, now visible instead of skipped.
- **`course_title` (345 suppressed).** 56 were plausible titles (code-bearing or repeated) that lost the ranking to another line; 51 named the document rather than the course (`Syllabus | Math 140`, `Page 1 of 4`, a file name), 39 were labelled attributes (`Instructor:`, `Email:`, `Course Format:`), 28 contact or meeting lines, 27 section headings, 22 bare codes, 22 place names, 20 session/term lines, 15 documents where every title-shaped line was refused and the field said so, 13 clock times, 12 generic section headings, 11 person lines, 11 reading references, 8 date/week lines, 4 candidates weaker than the document's own title line, 2 sole candidates below the plausibility floor, 2 pages where the best title-shaped line was not a printed title, and 2 where a line shaped like a title could not be read cleanly. A user gets one title and never sees the 56 alternatives, which remains the field where the ledger is largest.
- **`course_code` (74 suppressed).** 28 were other codes in the document that lost on confidence, 25 lines about a place rather than the course, 15 room numbers, 3 office numbers, 3 stop-word departments. The test `course_code` result is 7/7; the suppressed set is where the rest of each dev document's code-shaped tokens went.
- **`grading` (72 suppressed).** 27 rows carried a weight but the text beside it was not a category name, 18 were duplicates of a row already counted, 9 were read with a valid category and weight but excluded from the subset that sums to 100, 8 had a label too long to read as a category, 6 came from documents with only one category, 2 weighted outside 1-100%, and 2 were components of a parent weight. The 9 subset casualties are the ones worth a reader's attention: the sum-to-100 rule can discard a true row when the document prints an incomplete breakdown.
- **`materials` (32 suppressed).** 27 bodies were judged "not about the field", 2 were empty, 1 a placeholder, 1 navigation text, 1 opened another labelled field (`Lectures`). This is the gate that keeps dev `materials` at 8 emitted values; the values it does emit are 5/5 correct, so the precision gain is real, but the user loses material lists on documents where the body is a link farm or a nav block.
- **`office_hours` (14 emitted, 9 suppressed).** 4 bodies opened without naming an availability (three page fragments such as `Careers…` and `How to Improve`, one `Location` body); 1 body named a compressed day group but no clock or way to meet (`MTWTh, after class (Zoom)`, still `body is not about the field`); 1 section headed `Class Times` was rejected on the heading alone (`a class-time heading, not office hours`); 1 section was overwritten by a later one; 1 was a placeholder; 1 navigation. No office-hours value is now silently lost: the overwritten one is in `candidates` with the section that beat it.
- **`instructor` (6 suppressed).** All six were section bodies that were not person names (headers such as `Class Time:` or a policy paragraph). `credits` logs only a lower-ranked credit count, which no dev document has, so its zero is real.

The test split was **not** re-scored to build this ledger. Building it changed no emitted value — the values that pass emitted were identical as a multiset to the run before it — so the recorded test numbers remain the first touch described above. What changed then was what is disclosed: the ledger went from 380 to 740 losses because three caps and three pre-filters stopped hiding candidates, not because extraction got stricter. The title pass moved it to 794 without changing any other field, the abstention pass moved it to 838, and the code-evidence pass moved it to 840 with every other field's emitted count unchanged again.

### What this pass fixed, and how it was checked on development

Each catalogued cause was reproduced on development documents and measured there before the test split was scored. The full status, with the dev evidence per row, is in [EXTRACTION_EVAL.md](artifacts/field-extract/EXTRACTION_EVAL.md#known-failures-found-while-reading-the-test-documents).

- **Office-hours body from the next section.** A body that opens another labelled field (`Class:`, `Course Dates:`, `Lecture Times:`) is rejected, a line that glues two fields together is cut at the second label, and `TBD`/`TBA`/`None.` bodies are rejected before any cue check. Development: 14/14 judged office-hours values correct; test `office_hours` 5/7 (was 0/5).
- **Rooms, offices and session codes as course codes.** A code preceded by `Office:`, `Room:`, `Hall`, `Bldg` or `in`, a dotted room (`WEL 5.132`), a schedule line (`MWF 1351`) and a session code (`SA25`) are all rejected; an explicit `Course Code:`/`Course Number:` line is preferred. Development: documents with `Office: PAR 404` and `Office: WEL 5.239` now return `RHE 306` and `CH301`; test `course_code` 7/7 (was 4/8).
- **Nested and parenthetical weights.** A parenthetical carrying two weights is read as a breakdown of the category beside it; a parenthesised weight is a component when the line already stated a top-level one; a component listed again later is suppressed; and neither end of an `X-Y%` range becomes a row. Development: `WebAssign 15% (Lecture-quizzes 5% + Homework 10%)` yields `WebAssign 15` only and totals 100, and a 90-100% / 80-89% grade scale contributes no rows. Test `grading` 32/33 (was 29/36).
- **A heading as the title.** Generic headings (`General Information`, `Summary Assignment and Grading`, `Teaching Assistants`, `Supplementary Materials`), place names, clock times, code-only lines and session lines are rejected before ranking. Development: 3/3 judged titles correct and no chosen title is a heading; test `course_title` 8/8 (was 8/9).
- **A document label, a heading, or a contact line as the title.** Three more shapes are now rejections with a reason rather than values: a line that names the document (`SPAN 1002 Syllabus`, `Syllabus | Math 140`, `Page 1 of 4`, a file name, anything separated by a pipe or a backslash), a line that labels an attribute (`Instructor: ...`, `Course Format: ...`, `Skills and Experience Flag: ...`) and a meeting or contact slot (`MWF 1-2 pm, O. Hallatschek`); a `X, Not Y` slogan is rejected as a heading. Two of the shapes were previously *over*-rejected: the link filter (meant for `Course Website: Canvas`) matched any capitalised `label: value` line, and the section-heading rule treated any line starting with `General` as a heading. Both are narrowed, and every rejection still lands in the ledger. Development titles went from 3/9 to 5/9 correct with precision unchanged at 1.000; the row-by-row account is below.
- **Every loss is now visible.** The three reject caps (12 codes, 8 titles, 5 title alternatives) are gone, so 20 room-shaped lines produce 20 losses instead of 12 and 20 session lines produce 20 instead of 8. A grading row whose category is too long for the reader is logged (`a weight with no category name beside it`) instead of vanishing; the heuristic logs only label-shaped text, so CSS percentages and grade scales stay out of the ledger. `extract_meetings` records each schedule-shaped line it declines (exam dates, office hours, day-group mismatches, duplicates). Development: the audit's reproduction — a printed 10% row absent from a 90% breakdown — now logs that row and reports `grading_total_consistent: false`.
- **One sum rule.** `grading_sum_consistent` (100 ± 5 points) is now the only consistency test: it gates the confidence bump, the per-value `weights_sum_consistent` flag and the record's `grading_total_consistent`. Before this pass the extractor used 95-105 while the record flag accepted 90-110, and the per-value flag was hard-coded `true`, so a record could certify a breakdown the extractor itself rejected.
- **Confidence from evidence, not the section softmax.** A value's percentage is now computed from its own evidence — a label, the shape the field requires, a day and clock, a body that is about the field — instead of reusing the classifier's section probability. The band table above is the measurement of that change; the section softmax is still what types sections.

**This pass: titles, one row per cause, on the development split.** The held-out run named five title shapes — a file/document label, a page artefact, a section heading and a contact line. Each shape became a rule and was measured on development, where the same work also surfaced two titles lost to rules that were too broad.

| Development document | Gold title | Cause | After |
|---|---|---|---|
| `597fc82e` | `General Chemistry: Macroscopic Investigations and Reaction Principles` | The pre-filter meant for `Course Website: ...` matched any capitalised `label: value` line and dropped the title silently. It now matches link/contact labels only and logs the line (`a labelled link, not a course name`). | emitted, matches |
| `1279607f` | `General Chemistry 103` | The section-heading rule treated any line starting with `General` as a heading. `General`/`Course`/`Class` now only head a heading when a block noun follows (`General Information`, `Course Description`). | emitted, matches |
| `ea18223e` | `INTRODUCTION TO AMERICAN GOVERNMENT` | One printed line carries the code, the title, the instructor and the term; nothing in it is a label or a heading, and the classifier scores it below the 0.25 plausibility floor. | still a miss, now logged (`below the plausibility floor`) |
| `cabc2f31` | `Biology Author: Peter Raven Edition: 12th`, `Title: Biology` | The page title exists only in JavaScript/OpenGraph metadata; the readable lines are a code and a department name. | still a miss |
| `7a0efc5a` | `Differential and Integral Calculus` | Its heading, `M 408C Differential and Integral Calculus Syllabus`, is now a document-label rejection. The clean line elsewhere on the page is not the gold span, so the item was already a miss. | still a miss |

Development titles: **25 emitted, 5 judged, 5 correct, recall 5/9 = 0.556** (was 23 / 3 / 3 / 0.333). No other field's emitted count moved, and overall development precision is 75/82 = 0.915 (was 73/80 = 0.912).

**This pass: the decision the previous one named as missing.** Refusing a document label left the title slot to whatever line came next, and the field had no way to say a page has no title at all. Three documents on the untouched held-out set showed the two symptoms: a schedule line promoted into the title slot, and a page that prints a name but came back with none. The fix is a decision, not a threshold: when no candidate survives, the field emits nothing and **records why**, and the reason distinguishes three states a reader needs to tell apart — every title-shaped line was refused as not a course name, the best line was not a printed title, or a line that could have been the title could not be read cleanly. The note carries the printed line that caused it (never a scraped script line), and the refused labels are still listed above it, so the field's emptiness is auditable in the same record as the silence it replaces.

Two title rules had to be right for that decision to be honest, and both were wrong in opposite directions. A catalogue code can be one letter or initials (`M 408C`, `P.S. 101`), and without that shape the classifier's own score decided those pages at 0.05 — a page that plainly prints its name came back empty. Meanwhile a bare code with a delivery mode attached (`HIS 315k External`) and a registrar row (`Uniques: 49970 and 49975 · TTh in WEL 2.224`) both looked code-bearing, so both were exempt from the plausibility floor. The rule now reads a code as evidence only when it leads the line and a name sits beside it, with a colon counting as the name's introduction.

| Development document | Before this pass | After | Why |
|---|---|---|---|
| `7a0efc5a` | empty, no note (the correct line was read and dropped below the floor) | emitted `M 408C: Differential and Integral Calculus`, matches gold | A one-letter department code is a code. |
| `ea18223e` | empty, logged `below the plausibility floor` | emitted `P.S. 101 INTRODUCTION TO AMERICAN GOVERNMENT Dr. Frank Wayman F'18`, contains the gold name | One printed line carries code, title, instructor and term; the initials code vouches for the line. |
| `aab98f9c` | emitted `January 9-13 (Week 1)` | emitted `Introduction to Computers and Object-Oriented Programming` | A date/week row is refused, and the real title below it wins. |
| `dc439d2e` | emitted `Barbour & Wright, Chapter 1` | empty, `every title-shaped line names something other than the course`, evidence `Syllabus for PSCI 310L` | A reading reference is refused; the line below the label turned out to be a speaker with an affiliation (`Paul Light (Brookings Institution)`) and is refused as a person. |
| `4f58ca74` | empty, `nothing on the page is shaped like a title` | emitted `Introductory Sociology` | A raw page glued its heading to a script block on one line; the printed runs are read separately. |
| `cabc2f31` | empty, abstention quoted a line of Google Tag Manager code | empty, abstention quotes `BIOL 11000` | The evidence line is now a printed line, and the page really does print a code and a department name and no course name. |
| `18ae9f79` | empty, `below the plausibility floor` | empty, `the best line is not a printed title`, evidence `HIS 315k External` | A code with a delivery mode is not a name, so the field says the best line is not a printed title instead of promoting it. |
| `0b119551` | empty, `the best line is not a printed title`, evidence `Criteria (in Word)` | emitted `Interpretation and Argument` | The page prints the name with its term and section attached (`Interpretation and Argument, Fall 2005, Section AA`); the clauses are stripped and the name is what remains. |
| `639f5558` | empty, evidence a date line | emitted `Differential and Integral Calculus` | The page prints a catalogue number in front of the name (`21-120 : ...`); the number is stripped, still counted as the line's evidence, and the name is emitted. |
| `ce8445f2` | empty, evidence the label line | emitted `HIS 315K, US History 1492-1865` | The trailing `, Spring 2019` is a term clause, not part of the name. |

Development titles: **28 emitted, 7 judged, 7 correct, precision 1.000, recall 7/9 = 0.778** — the sequence is 3/9 (before the label rules), 5/9 (after them), 7/9 (now). No other field's emitted count moved, and overall development precision is 77/84 = 0.917 (was 75/82 = 0.915). The two remaining misses are `cabc2f31`'s two gold rows, which are a book citation scraped out of a page that never names the course; the field returns empty with `BIOL 11000` as the reason, and that is the intended answer rather than a recovered one.

**What is still wrong, and not fixed by this pass.** `b6c3fcc2` prints `WRITING NATURE:` on one line and `DISCOURSES OF ECOLOGY` below it, and the field emits only the second line — a partial title that the gold's substring rule accepts. `ea18223e`'s value is the whole packed line, not the name inside it. `8f063a9b` prints `Principles of Chemistry I · Fall 2016`: the name is followed by a term clause the suffix rule cannot see because there is no comma, so the whole line is read as a session line and the page abstains on a title it does print. `664c5638` is a page whose only title-shaped line names a department, and its abstention now quotes that printed line rather than nothing. Each is in the ledger with its reason; none of them is a silent drop, and none was bought with a threshold.

**The held-out set was not re-scored.** The one-touch rule is the point of the split, and the frozen artifacts are unchanged: `heldout_eval.json` sha256 `a6c9e5a3…`, the recorded test run `extract_eval_test.json` sha256 `c36e1489…`, both written before this pass opened. The test numbers quoted above therefore describe the code as it stood when they were written, not the code today; the report keeps both facts rather than replacing the old number with a better-looking one.

### The code-evidence pass: one predicate, and the per-cause table made complete

An audit of the abstention pass found a defect it had introduced and a cause behind it. The defect is reproducible on the shipped extractor: a page of enumerated policy bullets ("1. Read the syllabus…", "2. Come prepared every day") returned `Come prepared every day` as a course title at confidence 0.86 with rule `code_bearing_title`. The list marker was being read as a catalogue number, which gave the line the code bonus, exempted it from the plausibility floor and satisfied the promotion gate. With the pre-pass rules the same page returned nothing.

The cause was that "does this line carry a code?" had four implementations — the title regex, the numeric-prefix regex, a with-name rule and the field's own `_COURSE_CODE` — and they had drifted: one accepted `Set 3`, another `V 1.4`, another `M 408C`, another `21-120`. They are now one module-level pair in [pace/field_extract.py](pace/field_extract.py): `catalogue_code(line)` answers *where* a code sits, and `code_names_a_course(line)` answers whether a name sits beside it. The score bonus, the plausibility floor, the promotion gate and the document-label check all call those two; the bare-code rejection asks `catalogue_code` whether the code is the whole line. A list marker is not a code in this vocabulary, a version (`V 1.4`) is not a code at all, a month is not a department, a room with a decimal (`WEL 2.224`) is not a number, and a hyphenated numeric code needs two or three digits before the hyphen, so `21-120` and `76-101` still count while `2024-2025` and `1.` do not.

Two smaller pieces of the same audit are also closed. The abstention reasons live in one `TITLE_ABSTENTION` mapping and one `abstain(kind, evidence)` helper, so the three exits cannot drift or invent a cause by typo, and the two per-line reasons that are compared elsewhere (`a document label…`, `a session or term line`) are constants too. And the published per-cause table is now complete: [EXTRACTION_EVAL.md](artifacts/field-extract/EXTRACTION_EVAL.md) prints every field×cause row (92 rows) instead of the top two causes per field plus "other reasons", so all 19 title abstentions are visible in the artifact rather than folded into a remainder, with the printed evidence that produced each one.

Development, re-measured against the recorded run: **identical headline** — 28 titles emitted, 7 judged, 7 correct (precision 1.000), recall 7/9 = 0.778, overall 77/84 = 0.917. No title value changed, which is the expected result: the dev split contains no enumerated-list page, so the defect was pinned by a synthetic page and a unit test rather than by a gain on dev. The ledger moved 343 → 345 with two causes shifting by reclassification (bare codes 23 → 22, session/term lines 17 → 20); the emitted multiset is unchanged. The new tests are `test_a_numbered_list_marker_is_not_a_catalogue_code`, `test_one_predicate_answers_whether_a_line_carries_a_course_code`, `test_a_code_without_a_name_abstains_at_the_floor_it_does_not_skip`, `test_every_title_abstention_uses_the_shared_reason_vocabulary` and `test_published_report_lists_every_cause_including_abstentions`.

Nothing else moved: `DEFAULT_MIN_CONFIDENCE` is still `0.0`, the 0.25 plausibility floor is unchanged, the gold files and both frozen held-out artifacts are byte-identical, and the test split was not read or re-scored.

### What it gets wrong

Three of 66 judged values are wrong in the recorded test run. Two of those are gold questions rather than rule bugs, and they stay open in [known_failures.json](data/processed/field/known_failures.json):

- `office_hours` `F 3:00-4:00`: labelled and read off the page, but the gold does not accept it. Changing the extractor to drop a value the document states would trade a correct read for one gold row, so no rule is changed for it.
- `grading` `participation` 10%: printed in the document and emitted; the gold treats the category as out of scope. Satisfying it would mean dropping a printed weight, so it stays open.

The third, a section headed `Class Times` typed as office hours, now has a rule: a heading that names class/lecture/lab/meeting times is rejected as office hours on the heading alone. Development confirms the shape (`a612c6ca` really is typed `office_hours` at 0.607) and the document still emits its real Office Hours section; the test document was not re-read, so the row is fixed at the rule level and the recorded test number is untouched.

Two office-hours failures from the previous pass also get a rule: an office-hours body must name an availability in its opening sentence (a clock time, a day, `by appointment`, office/student/consultation hours, drop-in, or scheduling). A course overview or an instructor/contact block does not, so it is rejected with `body opens without naming when the instructor is available`. The guard against over-rejecting is on development, the only split allowed to decide it: `1c22a5fd` is the same shape — narrative prose after the heading — but its opening says `office hours`, and it is still emitted. Development's 212 emitted values are identical before and after, and `office_hours` stays 14 emitted / 9 suppressed with its reasons reconciling to 9. The two test documents were not re-read.

Older failures, still visible on the section-span stratum: flattened tables can pair a category with the wrong percentage (MIT `8.01SC`, `Final Exam 25%`), a `Course Overview` body is typed as `materials`, and a statement wrapped across printed lines is missed.

### One more thing the split does not prove

It proves the *rules* were not tuned on the test institutions. On the 14 corpus documents it does not make the encoder unseen: the shipped checkpoint was trained on rows from the corpus schools, including MIT, Ohio State and UNC. The 12 value-level documents are the exception — no nine-word phrase from any of them occurs anywhere in the classifier corpus, so for those documents neither the rules nor the encoder had seen the text.

**What 66 judged values can and cannot certify.** The value-level test gold is 66 values: enough to see a cause disappear and to put a lower bound on the recorded run (0.887 at 63/66), not enough to certify 0.98. Even a perfect 66/66 would certify only a 0.956 lower bound, and a 0.98 bound at zero misses needs at least 149 judged values. Development is 75/82 with a 0.846 bound: it can show that a rule fixed a cause or cost nothing, and it cannot certify 0.98 either. No rule in this report was chosen by a headline number, and none should be — a 0.98 claim needs a larger independent gold set, not a tighter threshold on this one. The 20-document held-out set above is that larger set (61 judged values), and it answers the question the other two could not: measured **0.820**, bound **0.719**, well below 0.98 at p ≈ 3×10⁻⁸.

### Artifacts

| | |
|---|---|
| Published table: protocol, both splits, the final number | [artifacts/field-extract/EXTRACTION_EVAL.md](artifacts/field-extract/EXTRACTION_EVAL.md) |
| The split, with its exclusions and the gold institutions | [data/processed/field/extract_split.json](data/processed/field/extract_split.json) |
| Value-level gold (12 documents, every named field) | [data/processed/field/gold_test_fields.json](data/processed/field/gold_test_fields.json) |
| The 12 documents, with source URLs and fetch gates | [data/raw/field-eval/syllabi.jsonl](data/raw/field-eval/syllabi.jsonl), [provenance](data/raw/field-eval/syllabi_provenance.json) |
| Failures, with the state each reached and the dev check | [data/processed/field/known_failures.json](data/processed/field/known_failures.json) |
| Development run | [artifacts/field-extract/extract_eval_dev.json](artifacts/field-extract/extract_eval_dev.json) |
| Test run | [artifacts/field-extract/extract_eval_test.json](artifacts/field-extract/extract_eval_test.json) |
| Previous pass's scored test run (the "before" numbers) | [artifacts/field-extract/extract_eval_test.previous.json](artifacts/field-extract/extract_eval_test.previous.json) |
| First test run of the previous pass | [artifacts/field-extract/extract_eval_test.run1.json](artifacts/field-extract/extract_eval_test.run1.json) |
| Hand-read meetings and credits (dev) | [data/processed/field/hand_gold.json](data/processed/field/hand_gold.json) |
| Hand-read meetings (test) | [data/processed/field/hand_gold_test.json](data/processed/field/hand_gold_test.json) |
| Held-out rubric (written before scoring) | [docs/heldout-rubric.md](docs/heldout-rubric.md) |
| Held-out gold (20 documents, frozen hash) | [data/processed/field/gold_heldout_fields.json](data/processed/field/gold_heldout_fields.json), [selection](data/processed/field/heldout_selection.json) |
| Held-out run, first touch (precision, interval, per field, loss ledger), frozen | [artifacts/field-extract/heldout_eval.json](artifacts/field-extract/heldout_eval.json) |
| Held-out run, second touch after the title fix (contaminated by construction) | [artifacts/field-extract/heldout_eval_second_touch.json](artifacts/field-extract/heldout_eval_second_touch.json) |
| Extractor | [pace/field_extract.py](pace/field_extract.py) |
| Scoring harness | [pace/field_extract_eval.py](pace/field_extract_eval.py), [pace/field_gold.py](pace/field_gold.py), [pace/field_heldout_eval.py](pace/field_heldout_eval.py) |

## Original track: catalog-cell pipeline

## Built

- Downloaded and inventoried the three requested public HF datasets; retained untouched raw files and licensing metadata.
- Trained a sparse real-text baseline, MiniLM, and DeBERTa-v3-xsmall on natural USC catalog cells (8 field classes), with whole subject prefixes held out.
- Trained a grounded course-mention relation classifier (`pace/real_relations.py`) on 3,212 real USC course references across 5 relation types (`prerequisite_of`, `duplicate_credit_of`, `recommended_for`, `corequisite_of`, `related_reference`) using the exact same subject-held-out split.
- Evaluated zero-shot English block transfer and out-of-distribution (Thai script / HTML table) stress behavior on CU-DocBench (`pace/cu_eval.py`), and added OOD script/table abstention guards to `pace/real_infer.py`.
- Exported both the MiniLM field classifier and MiniLM relation classifier to ONNX FP32 and dynamically quantized INT8 (`pace/export_onnx.py`), verifying numerical parity, held-out test accuracy, 3.97x model compression, and CPU latency speedups.

## Data found and disposition

- GU course syllabus: 2,971 natural syllabus rows, 11 text fields. Row token count: mean 1,278, median 1,209, p95 1,945, max 8,621. HF card has no license. **Now used for training the field classifier** (46,338 rows in Task 5) — a deliberate change from "inspected only", recorded in [docs/research.md](docs/research.md#real-text-field-classifier-corpus-additional-sources-and-their-rights) because no permissively licensed syllabus corpus was available. Not cleared for redistribution.
- USC Course Catalog: 5,571 CSV rows, 13 fields; HF card declares Apache-2.0 but contributor-scraped source rights are not independently confirmed. Used field text and course mentions, omitting instructor/section identifiers.
- CU-DocBench: 23 documents, 100 gold pages, 1,065 blocks, four syllabus documents / 12 pages; card says MIT. Used strictly for evaluation (`artifacts/cu-docbench/evaluation.json`).
- Full inventory: [data/inventory.json](data/inventory.json); rights/download provenance: [data/manifest.json](data/manifest.json); details: [docs/research.md](docs/research.md).

## Task 1: Catalog-cell field classification (USC)

- Task: classify a natural CSV cell as one of eight existing columns: title, description, units, prerequisite, restriction, time, days, or class type.
- Exact normalized `TBA` appears in both Days (1,317) and Time (1,335). The source column cannot be recovered from the same literal string, so 2,652 such examples are excluded and inference abstains for normalized `tba`.
- Remaining examples: 33,799; train 23,175 from 144 subjects, validation 5,102 from 30, test 5,522 from 31 unseen subjects.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Serialized size |
|---|---:|---:|---:|---:|
| Char 2–5 gram TF-IDF + logistic regression | 0.99728 | 0.99742 | 0.99728 | 1,963,196 B |
| Fine-tuned MiniLM-L6 (PyTorch FP32) | 0.99982 | 0.99986 | 0.99982 | 90,893,327 B |
| Fine-tuned MiniLM-L6 (ONNX INT8 dynamic) | 0.99982 | 0.99986 | 0.99982 | 22,895,051 B |
| Fine-tuned DeBERTa-v3-xsmall | 1.00000 | 1.00000 | 1.00000 | 282,794,003 B |
- Models, sizes, same split: [comparison.json](artifacts/usc-real/comparison.json). Errors on 5,522 test cells: {'char TF-IDF logistic': 15, 'MiniLM-L6 fine-tuned': 1, 'DeBERTa-v3-xsmall fine-tuned': 0} (ONNX INT8 also 1 error, 100% label agreement with PyTorch FP32).
- Calibration (validation-only temperature fit): MiniLM T=0.7242, test ECE-10=0.000088, >=0.90 coverage=100.0%, precision=99.9819%; DeBERTa T=0.1102, ECE-10=0.000000, >=0.90 coverage=100.0%, precision=100.0%.

## Task 2: Grounded course-mention relation classification (USC)

- Extracted 3,212 real course-code mentions (`[TARGET] SUBJ NUM [/TARGET]`) across Catalogue and Prerequisites fields, split by the exact same subject-group assignment (1,945 train / 615 validation / 652 test).
- 5 relation classes: `prerequisite_of` (2,863), `duplicate_credit_of` (159), `recommended_for` (147), `related_reference` (30), `corequisite_of` (13).

| Model | Overall Acc | Overall Macro-F1 | Overall Errors | Catalogue-only Acc (n=127) | Serialized size |
|---|---:|---:|---:|---:|---:|
| Char 3–5 gram TF-IDF + balanced LR | 0.98926 | 0.80792 | 7 | 0.94488 | 304,452 B |
| Fine-tuned MiniLM-L6 (PyTorch FP32) | 0.99387 | 0.91933 | 4 | 0.96850 | 90,891,663 B |
| Fine-tuned MiniLM-L6 (ONNX INT8 dynamic) | 0.99387 | 0.91933 | 4 | 0.96850 | 22,893,887 B |
- Full metrics: [artifacts/usc-relations/metrics.json](artifacts/usc-relations/metrics.json).

## Task 3: CU-DocBench gold document & OOD evaluation

- Evaluated all 23 CU-DocBench documents (953 non-image gold blocks, including 410 Thai-script blocks and 47 HTML table blocks).
- Zero-shot English syllabus block transfer (`pr_syllabus.pdf`, `cem1_syllabus.pdf`): **14/14 (100.0%)** accuracy.
- Out-of-distribution stress test: before abstention guards, the English catalog classifier produced >=0.80 confidence on 62/73 (84.9%) Thai-script and HTML-table blocks. `RealCatalogEngine` now explicitly abstains (`label='unknown', abstained=True`) on non-Latin script and raw HTML table blocks.
- Full evaluation: [artifacts/cu-docbench/evaluation.json](artifacts/cu-docbench/evaluation.json).

## ONNX export, dynamic INT8 quantization, and latency benchmarks

- Field classifier (`artifacts/usc-miniLM/onnx/`): FP32 ONNX max logit diff vs PyTorch = `5.25e-06`, 100.0% label agreement. Dynamic INT8 reduces model size from **90.89 MB → 22.90 MB (3.97x smaller)** with **100.0% label agreement** on all 5,522 test rows (99.982% accuracy, 1 error).
- Relation classifier (`artifacts/usc-relations/onnx/`): FP32 ONNX max logit diff vs PyTorch = `4.53e-06`, 100.0% label agreement. Dynamic INT8 reduces model size from **90.89 MB → 22.89 MB (3.97x smaller)** with **100.0% label agreement** (4 test errors in both FP32 and INT8).
- Desktop RTX 4070 PyTorch FP32 latency: p50 2.74 ms, p95 3.02 ms (360.2 rows/s).
- CPU single-row latency (including tokenization): PyTorch CPU FP32 p50 20.68 ms → ONNX Runtime FP32 p50 13.88 ms → ONNX Runtime INT8 p50 **9.11 ms (2.27x faster than PyTorch CPU)**.

## Task 4: synthetic-only multitask document understanding

This task is **synthetic-only** and is not real-document evidence. It is reported separately from Tasks 1-3 on purpose.

- Data: 30,000 procedurally generated documents, seed 20261019, 2% length-preserving character-substitution corruption. No images, no real OCR.
- Split: entire synthetic school identities and document template families held out; evaluates only school/template intersections; all ten classes occur in each split. Train 10,279 / validation 1,069 / test 1,208 documents; sliding-window rows 12,098 / 1,207 / 1,551.
- Model: shared MiniLM encoder with 6 transformer layers, document-type head (10 classes) and BIO entity head (19 entity types). 22,732,081 parameters, checkpoint 90,958,791 B; dynamic INT8 58,654,218 B (0.6448x of FP32).

| Text condition | Gold spans | Predicted | Precision | Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|
| Clean (no corruption) | 20,382 | 20,393 | 0.99941 | 0.99995 | 0.99968 |
| Fully corrupted | 20,382 | 20,413 | 0.99833 | 0.99985 | 0.99909 |
| Context-only corrupted | 20,382 | 20,397 | 0.99917 | 0.99990 | 0.99953 |

- Document type (10 classes, corrupted test): shared PyTorch multitask accuracy 1.00000 / macro-F1 1.00000; identical-data TF-IDF + logistic regression accuracy 1.00000 / macro-F1 1.00000. The task is cue-easy, so this is not evidence of real-world generalization.
- Per-template-family corrupted span F1 ranges from 0.9909 to 1.0000. Full report: [artifacts/synthetic-multitask/metrics.json](artifacts/synthetic-multitask/metrics.json).

### Model size vs accuracy (encoder depth study)

Same seed, same generated corpus, same school/template hold-out for every variant; only the number of transformer layers the shared encoder keeps differs.

| Encoder layers | Parameters | Checkpoint | INT8 checkpoint | Clean span F1 | Corrupted span F1 | Transcript span F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 22,732,081 | 90,958,791 B | 58,654,218 B | 0.99968 | 0.99909 | 0.99905 |
| 4 | 19,183,153 | 76,754,848 B | 55,052,946 B | 0.99951 | 0.99890 | 0.99878 |
| 2 | 15,634,225 | 62,550,848 B | 51,451,701 B | 0.99848 | 0.99674 | 0.99742 |

The 2-layer variant is 31.2% smaller than the 6-layer one and gives up 0.24 F1 points of corrupted span F1; the 4-layer variant saves 15.6% of the checkpoint for 0.02 F1 points. Full numbers: [artifacts/synthetic-multitask/depth_comparison.json](artifacts/synthetic-multitask/depth_comparison.json).

### Image-derived transcript run (one document, not a benchmark)

- Document type: `transcript` (p = 0.9929; every other class below 0.001).
- Model entity spans: 220/220 gold matched, precision 1.000, recall 1.000, F1 1.000 across 10 types.
- Parser records: 38 course records over 8 course subject areas (11 sections in the document, the other 3 being assessment sections), 3 assessment records, computed total credits 13.25 vs reported 13.25 (match: True, difference 0.0), 0 records without a title, 0 without a section, 0 unattached performance levels. GPA present in document: False (reported as null, not invented).
- Full report: [artifacts/synthetic-multitask/transcript_report_rowwise.json](artifacts/synthetic-multitask/transcript_report_rowwise.json).

## Reproduction

Run `python -m pace.inspect_datasets`, `python -m pace.real_baseline`, `python -m pace.real_train`, `python -m pace.real_relations`, `python -m pace.cu_eval`, `python -m pace.export_onnx`, `python -m pace.compare_real`, `python -m pace.synthetic_doc_model`, `python -m pace.compare_depths`, `python -m pace.evaluate_transcript data/samples/mock_transcript_lepid_rowwise.txt`, `python -m pace.report`, and `python -m pytest -q`.
