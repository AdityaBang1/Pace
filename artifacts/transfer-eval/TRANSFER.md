# What actually moves the reader on documents it did not see

Five checkpoints, three benchmarks, one question: does more training on the same
corpus buy generalisation, or does it only polish the familiar distribution?
Every number below was produced by `python -m pace.jev_eval` on a fixed split,
with the temperature the checkpoint ships with.

## Benchmarks

| name | file | rows | what it measures |
|---|---|---:|---|
| field corpus test | `data/processed/field_mix_ext/test.jsonl` | 7,571 | 38 fields, IID: catalog + syllabus text from the same families as training |
| syllabus document-shift | `data/processed/transfer/syllabus-docs/test.jsonl` | 534 | structured facts from **new** syllabus documents, 10 schools that all also appear in training (document-held-out, not school-held-out) |
| SIS / Infinite Campus holdout | `data/processed/sis/sis_holdout.jsonl` | 354 | cell values from OneRoster exports composed out of validation-only catalog rows, never trained on |

## Results

| checkpoint | what it is | field corpus macro-F1 | syllabus acc / macro-F1 / ECE | SIS acc / macro-F1 / ECE |
|---|---|---:|---|---|
| `jev-field-v2` | shipped before this pass | **0.9711** | 0.648 / 0.425 / 0.293 | 0.847 / 0.492 / 0.122 |
| `jv3-c2-ctrl` | v2 + 1 epoch, same data, plain mix | n/a | 0.507 / 0.355 / 0.292 | n/a |
| `jv3-c2-domw` | v2 + 1 epoch, syllabus rows at 14% of the batch | 0.9697 | 0.624 / 0.384 / **0.189** | n/a |
| `jv4-soup` | 0.5·v2 + 0.5·domw | 0.9711 | **0.669** / 0.418 / 0.275 | 0.788 / 0.469 / 0.157 |
| `jv5-field` | from the language-only pretrain, 4 epochs, +13 k Infinite Campus rows | 0.9493 | 0.575 / 0.349 / 0.348 | **0.949** / **0.596** / **0.042** |
| `jv7-field` | v3 + 5,624 generated value shapes (rooms, honorifics, day-time ranges, exam+date) | 0.9696 | 0.637 / 0.371 / 0.328 | 0.884 / **0.549** / 0.083 |
| **`jev-field-v3`** | 0.7·v2 + 0.3·jv5, temperature refit | **0.9739** | 0.624 / **0.433** / 0.321 | 0.895 / **0.531** / 0.066 |
| **`jev-field-v4`** | v2 + 1 epoch with half the rows' column header hidden | 0.9734 | **0.839 / 0.659** / **0.133** | **0.915** / 0.476 / **0.065** |
| `jv9-header-2ep` | one more epoch of the same recipe | 0.9322 | 0.858 / 0.663 / 0.120 | 0.924 / **0.586** / 0.058 |
| `jev-field-v5` | 0.5·v2 + 0.5·v4 soup | 0.9702 | 0.817 / 0.614 / 0.150 | **0.921** / 0.512 / 0.061 |

The soup was measured because averaging usually keeps both members' strengths; it
did not (it sits below v4 on document-shift and below v2 on the field corpus), so
v4 is the primary reader and the soup is recorded only as an arm.

`jev-field-v4` is the first checkpoint that changes the document-shift result
rather than trading against it: +21 accuracy and +22 macro-F1 points on the 534
new-syllabus benchmark, at the cost of 5 macro-F1 points on SIS cells (0.531 →
0.476). It comes from hiding the column header on half the training rows, because
until then the model was reading the header and not the value — see
"Header reading" below.

The second epoch (`jv9-header-2ep`) is interesting rather than simply worse: it is
the best checkpoint on SIS cells (0.924 / 0.586 / 0.058) and the best on bare
values (0.9499 / 0.8308), but it gives up 4 macro-F1 points on the published-row
field corpus (0.9322) because more of its training signal is header-free. It was
not promoted — the shipped reader must not lose the in-distribution score — but it
marks the direction of travel: as the header dependency goes to zero, the
header-conditioned headline has to be rebuilt from semantics.

A sixth arm is worth recording because it failed informatively. `jv7-field` was
fine-tuned on 5,624 generated shapes after the probes showed the model calling
`Room 303` a `course_code` at 95% and `Dr. Johnson` a `course_code` at 99%. It fixed every shape probe (9/9 at 99.9%) and lifted SIS
macro-F1 to 0.549 — and it cost 6 macro-F1 points on new syllabus documents
(0.433 → 0.371). Averaging it back into v3 washed the shapes out again (6/9).
The shapes are therefore resolved deterministically in `pace/field_shapes.py`
(15/15 positives, 0 of 10 ambiguous negatives), and the model stays on v3.

## The benchmark that decides it: grouped school holdout

`scripts/build_school_folds.py` groups the 14 real syllabus schools into three
folds (balanced by row count), removes each fold's schools from **training across
every family** together with any composed row carrying their document id, and
returns the three folds as one pooled test of 1,638 rows. Every school is unseen
exactly once; a model is only ever scored on rows from schools it never saw.

| fold | held-out schools | rows | accuracy | macro-F1 |
|---|---|---:|---:|---:|
| 0 | CMU, Penn State, Stanford, UC Berkeley | 544 | 0.403 | 0.341 |
| 1 | MIT, Ohio State, UCLA, UIUC, Michigan | 538 | 0.595 | 0.469 |
| 2 | Georgia Tech, Purdue, UT Austin, UNC, Wisconsin | 556 | 0.477 | 0.412 |
| **pooled** | all 14 | 1,638 | **0.491** | **0.388** (σ 0.052) |

Strict variant, dropping the 149 test rows whose exact text also appears in that
fold's training set (generic values such as `Attendance`, `3`, `Lecture` — 9.1%):
accuracy 0.441, macro-F1 0.322, ECE 0.423.

**This is the number that decides the model, and it changes a conclusion.**
A model trained with those schools removed scores 0.388 macro-F1 on schools it
never saw. `jev-field-v2`, which trained on all 14 schools, scores 0.400 on the
fold that removes the whole syllabus family (`unseen-syllabus`, 1,638 rows). The
two numbers are the same within noise. So the 0.60 accuracy on real syllabi is
**not** school memorisation — knowing the school buys almost nothing. It is the
document-family gap: the encoder was trained on 200 k catalog cells and 1.4 k
syllabus rows, and a syllabus passage is a different kind of text.

On the same fold, `jev-field-v3` scores **0.390** against v2's **0.400**, i.e. the
soup I promoted is *not* better where it matters most. v2 is therefore the primary
reader and v3 is only the district-format specialist; see the correction below.

### Where the school-holdout errors are

Family-level accuracy (coarse groups) is **0.543** pooled, so this is not only a
fine-taxonomy disagreement — the model does not separate these groups on unseen
prose:

| family | rows | accuracy |
|---|---:|---:|
| identity (title, code, description, level…) | 52 | 0.846 |
| when (meetings, term, schedule, exam dates) | 253 | 0.640 |
| grading (policy, grades) | 275 | 0.633 |
| people (instructor, office hours, contact) | 146 | 0.596 |
| **work (assignments, readings, assessment, materials)** | **679** | **0.459** |
| **rules (prerequisites, policies, integrity, accommodations)** | **155** | **0.374** |

41% of the test rows are in `work`, and the upstream extractor puts weekly lecture
topics, reading lists and graded tasks in that one class. That is a taxonomy and
context problem with a measured size, not a tuning problem: see the next section.

## Distillation into the 10.3 M architecture: measured, and not shipped

`python -m pace.jev_distill` trains the 10.3 M architecture on v4's full 42-way
distribution (KL, T=2, alpha 0.5, initialised from the language-only pretrain,
3 epochs, temperature refit on validation).

| benchmark | v4 (15.3 M) | distilled (10.3 M) |
|---|---:|---:|
| field corpus test (7,571) | 0.9983 / 0.9734 | 0.9958 / 0.8435 |
| bare values | 0.9473 / 0.8172 | 0.8846 / 0.6585 |
| syllabus document-shift (534) | 0.8390 / 0.6585 / ECE 0.133 | 0.5880 / 0.3149 / ECE 0.302 |
| SIS cells (354) | 0.9153 / 0.4758 | 0.8475 / **0.6224** / ECE 0.098 |

It loses 13 macro-F1 points on the field corpus and 25 points on new syllabus
documents, and it is worse than the earlier direct fine-tune of the same
architecture (`jev-field`, 0.9155 macro-F1). Only rare-field SIS macro-F1 improves.
Conclusion: at this budget the 15.3 M teacher is the small model; distilling to
10.3 M is not a free win and the checkpoint is kept as evidence, not shipped. A
scratch-initialised arm (no pretrain, 3 epochs) was worse still — 0.9901 / 0.7586 —
so the language pretrain matters more than the teacher's soft labels.

## One active-learning round: labels help, entropy does not pick them better than random

`scripts/active_learning_round.py` ranks the untouched validation split by
predictive entropy, adds the 1,500 most uncertain rows to the training set and
retrains from v4 for one epoch. A random 1,500-row arm at the same budget is the
control.

| arm | field corpus | unseen syllabus documents | SIS cells |
|---|---:|---:|---:|
| v4 (no extra rows) | 0.9983 / 0.9734 | 0.839 / 0.659 / ECE 0.133 | 0.915 / 0.476 |
| + 1.5 k **most uncertain** | 0.9979 / 0.9672 | 0.854 / **0.714** / ECE 0.129 | 0.932 / 0.485 |
| + 1.5 k **random** | 0.9980 / 0.9727 | **0.858** / 0.710 / ECE 0.122 | 0.944 / **0.554** |

More labelled rows do move the document-shift benchmark (+1.5 to +1.9 accuracy,
+5 macro-F1), which is another vote for data over epochs. Uncertainty selection
did **not** beat random at this budget, and it cost 7 macro-F1 points on SIS
cells, so the selection criterion is rejected and the extra-data effect is
recorded as the finding.

## Calibrated abstention: what a percentage buys on unseen data

`scripts/measure_abstention.py` sweeps thresholds on the shipped checkpoint and
reports coverage, precision and the Clopper-Pearson lower bound
(`artifacts/jev-field-v4/abstention.json`). At a **0.90** gate:

| data | rows | accuracy | coverage @0.90 | precision @0.90 |
|---|---:|---:|---:|---:|
| field test, published rows | 7,571 | 0.9983 | 1.000 | 0.998 |
| field test, bare values | 7,571 | 0.9473 | 0.997 | 0.950 |
| unseen syllabus documents | 534 | 0.8390 | 0.863 | **0.902** |
| SIS cells | 354 | 0.9153 | 0.994 | 0.920 |
| held-out publisher: MIT | 17,277 | 0.9300 | 0.974 | 0.943 |
| held-out publisher: USC | 19,811 | 0.9690 | 0.994 | 0.973 |
| held-out publisher: Michigan | 337 | 0.9550 | 0.991 | 0.961 |
| held-out publisher: Gothenburg | 62,170 | 0.2500 | — | no threshold reaches 0.90 |
| held-out publisher: 14-school syllabi | 2,775 | 0.1910 | — | no threshold reaches 0.90 |

That is the practical rule the measurements support: gate at 0.90 and the reader
is right ~90% of the time on new syllabus documents while refusing 14% of rows;
on a publisher whose prose it cannot read at all, every threshold fails, so the
honest behaviour is to abstain on the whole document.

## Leave-one-publisher-out: catalogue cells travel, syllabus prose does not

A harder benchmark than the school holdout: `scripts/build_source_holdout.py`
removes an entire *publisher* from training (its layout, its column names, its
vocabulary) and asks the model to read that publisher's cells. Five folds,
103 k held-out rows, trained from the language-only encoder.

| held-out publisher | rows | accuracy | on labels training did see |
|---|---:|---:|---:|
| MIT course catalog | 17,277 | 0.9865 | 0.9865 |
| USC course catalog | 19,811 | 0.9715 | 0.9894 |
| Michigan course descriptions | 337 | 0.9288 | 0.9288 |
| Gothenburg course syllabi | 62,170 | 0.4357 | 0.6702 |
| 14-school syllabus corpus | 2,775 | 0.3348 | 0.3887 |
| **mean** | 102,370 | **0.7314** | **0.7927** |

Catalogue cells transfer to a publisher never seen; syllabus prose does not. That
is the sharpest statement of the open problem: the model has learnt what a course
*title* or *code* looks like in a catalogue, not yet what a syllabus paragraph
about grading looks like. Macro-F1 in the same run is low (mean 0.2635) because
several held-out labels have no training rows at all — the `seen-label` column is
the fair comparison for those folds.

**The v4 recipe makes this worse, and that is worth knowing.** Re-running the same
five folds with header dropout (`artifacts/source-holdout-hd`):

| | mean accuracy | mean macro-F1 | mean on seen labels |
|---|---:|---:|---:|
| header kept (v2-style) | **0.7314** | **0.2635** | **0.7927** |
| header dropout (v4-style) | 0.6228 | 0.1953 | 0.6580 |

The Gothenburg fold falls from 0.6702 to 0.3850 on seen labels. So the header is
doing two jobs: it is a shortcut *inside* a known corpus, and it is the only
handle the model has when a new publisher names its columns. v4 reads values
better (0.899 → 0.947 accuracy on bare values) but leans on familiar headers when
it meets an unfamiliar publisher. Practical consequence: use v4 on documents whose
column vocabulary is covered by the corpus, and expect it to trail v2 on a
publisher nobody has seen until that publisher's headers are in the training set.

## Header reading: the number that was hiding in plain sight

Every row is written `"<column>: <value>"`, and the column usually names the
field. Scored twice on the same 7,571 held-out rows (`scripts/measure_header_reliance.py`):

| checkpoint | published rows (header in) | value only (`bare_text`) |
|---|---:|---:|
| `jev-field-v2` | 0.9982 / 0.9711 | 0.8991 / 0.7000 |
| `jev-field-v4` | 0.9983 / 0.9734 | **0.9473 / 0.8172** |
| `jv9-header-2ep` | 0.9978 / 0.9322 | 0.9499 / 0.8308 |

So roughly 10 accuracy points and 27 macro-F1 points of the old headline came
from the prefix, which is exactly the signal that disappears when an institution
renames its columns. Hiding the header on half the training rows recovers most of
it (+4.8 accuracy, +11.7 macro-F1 on bare values) and, unexpectedly, also improved
the published-row score. The remaining gap (0.051 accuracy, 0.156 macro-F1) is
the honest size of the semantic task.

## What this says

1. **More epochs on the same corpus do not transfer.** `jv3-c2-ctrl` (same data,
   one more epoch) loses 14 points of accuracy on new syllabus documents against
   v2. The first epoch of a fine-tune from a converged checkpoint moves the model
   away from its optimum faster than it recovers.
2. **Domain weighting helps the shifted benchmark and costs IID.** Putting
   syllabus rows at 14% of each batch (`domw`) lifted syllabus accuracy over the
   control by 12 points and cut shifted ECE from 0.29 to 0.19, while field-corpus
   macro-F1 fell 0.0014.
3. **New data does transfer across formats.** The Infinite Campus rows — 3,296
   labelled cells from 982 composed OneRoster course/class records — took SIS
   accuracy from 0.847 to 0.949 and macro-F1 from 0.492 to 0.596 on rows the model
   had not seen. That is the only large, unambiguous gain in this table.
4. **Averaging keeps both skills.** 70/30 v2/jv5 keeps the field corpus at v2's
   level (0.9739 vs 0.9711), edges the shifted benchmark up (0.433 vs 0.425) and
   keeps most of the SIS gain (0.895 acc, 0.531 macro-F1).
5. **School holdout and family holdout are the same difficulty.** A model that
   never saw a school reaches the same score as a model that saw every school but
   never saw the document family. Effort spent on school coverage is not where the
   remaining points are; effort spent on syllabus-style reading is.
6. **Confidence does not transfer with the weights.** Shifted ECE is 0.29–0.35
   for every checkpoint tried. Temperature scaling fitted on IID validation cannot
   fix it, because the problem is not a global scale: the model is confident on
   text from a document family it has seen four times. This is a gate problem, not
   a calibration-scalar problem.

## Why the shifted benchmark is close to its ceiling

177 of 534 rows are wrong. Reading them:

- 23 are inside the work family (`assignments` vs `course_schedule` vs
  `weekly_readings` vs `assessment`), where the gold and the field definitions
  disagree about the same sentence.
- 28 are weekly schedule lines (`Lecture: derivatives and rates of change`) that
  gold labels `assignments` and the model calls `course_title`; under the field
  definitions they are `course_schedule`.
- 12 are reading citations (`Davidson & Lytle, After the Fact, pp. 24-48`) that
  gold labels `assignments` and the model calls `instructor`; `weekly_readings`
  describes them better than either.
- The rest are gold mistakes visible on inspection — a sentence about calculators
  labelled `exam_schedule`, a paragraph on animal research labelled
  `office_hours`, `Chemistry 1A` labelled `course_code` where the model says
  `program` (it is both).

The validation split shows the same thing: of the 3 `exam_schedule` rows it
holds, the one the model gets wrong is a cell-phone policy paragraph. Until the
gold is repaired, gains on this benchmark are mostly measured noise. The
productive next step is a **section-aware** input (`heading | value`), because a
value like `Lecture: chain rule` is only ambiguous when its heading is stripped —
and the pipeline already segments headings in `pace.field_corpus`.

## Files

| path | what it is |
|---|---|
| `scripts/build_sis_exports.py` | composes OneRoster exports out of real catalog values |
| `data/processed/sis/export/`, `export-holdout/` | the two composed exports (train / holdout) |
| `data/processed/sis/sis_corpus.jsonl`, `sis_holdout.jsonl` | 3,296 training cells, 354 held-out cells |
| `data/processed/transfer/syllabus-docs/train_ic.jsonl` | 240,167 rows: balanced mix + 4× the SIS cells |
| `scripts/soup_checkpoints.py` | weight averaging for identical-architecture checkpoints |
| `scripts/fit_temperature.py` | refits the shipped temperature after averaging |
| `artifacts/jev-field-v3/` | the promoted checkpoint, its card and its metrics |
