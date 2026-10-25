# Jev-shaped decision model, v2: more data, a bigger encoder, and the label noise removed

`artifacts/jev-field-v2` is the second run of the Jev-shaped field reader — the
same idea as [`artifacts/jev-field`](../jev-field/EXPERIMENT.md): a state, a typed
question, and a menu of described options instead of a 42-way softmax. Three
changes were made, and each one moved a measured number:

1. **18,706 more real rows**, from two catalogs the first run never saw:
   Columbia's public Directory of Classes (37 terms) and the Coursera catalog.
2. **A 15.3 M encoder** pretrained on 4.5 M passages, instead of 10.3 M on 3.6 M.
3. **Placeholder rows removed from training.** Bodies whose real heading was lost
   were being labelled by a heading that is not in the text — 63 of `materials`'
   185 training rows were this, 45 of `office_hours`', 8 of `attendance_policy`'s
   16. They stay in validation and test (the reported rows did not move); they
   are no longer training material.

The headline on the unchanged 7,571 held-out rows: **14 errors, accuracy 0.9982,
macro-F1 0.9711, ECE 0.0016** — the same 14 errors and the same accuracy as the
33.4 M bge-small reference trained on the same split, with 2.2× fewer parameters,
and **+5.6 macro-F1 points over the first run** (0.9155). That is a tie rather than
a win, and the unseen-university number in this run is worse than v1's. Both facts
are measured below rather than smoothed over.

## Shape

| | Value |
|---|---|
| Parameters | 15,331,563 (encoder 15,101,120; heads 230,443) |
| Layers × width | 12 × 320, pre-LayerNorm |
| Attention | 4 heads |
| Feed-forward | 1,024, GELU |
| Token embeddings | 128, tied to the MLM head through a 320→128 projection |
| Vocabulary | 16,000 WordPiece, trained on the extended corpus, lowercased |
| Positions | 512 |
| State budget | 127 tokens |
| Option budget | 24 tokens per description |
| Checkpoint | 61 MB fp32; ≈15 MB in int8 |

One encoder pass reads `[CLS] state [SEP] question [SEP]` and mean-pools the state
tokens only; each option is scored from its own description tokens with an
embedding bag and a small head, so adding, renaming or re-describing an option is
free. The score is a scaled cosine plus a per-label bias, and a single temperature
(0.626) is fitted on validation.

## The two new sources, and the cap that makes them safe

| Source | What it is | Cells kept | What it added |
|---|---|---:|---|
| Columbia Directory of Classes (`soid/columbia-catalog-data`) | 37 terms, 2015 Summer – 2027 Summer, one row per section, columns already named for the field | 16,674 | day strings, clock spans, rooms, section types, points, open-to, term |
| Coursera catalog (`azrai99/coursera-course-dataset`, Apache-2.0) | online-course listings: level, schedule, organisation, description | 2,032 | a second layout for level/schedule/program |

Every cell is printed as the source printed it; the only composition is Columbia's
start/end pair, joined as `9:00am - 12:10pm`. Rows are de-duplicated by
`(text, label)` after the normalisation the rest of the corpus uses.

**The cap is the interesting part.** Columbia is 37 terms of one registrar, and
without a limit it is 248,642 rows of a 340,000-row corpus — 73% of everything the
model reads, and 169 k of those rows are section ids. A model trained on that is
mostly trained on one registrar's layout, which is the exact failure the
cross-institution fold measures. Capping each field's new rows at **2,500** keeps
the coverage Columbia is there for (123 day strings, 1,841 clock spans, 1,360
rooms — all under the cap) and drops the bulk repetition:

| | uncapped | capped (shipped) |
|---|---:|---:|
| Columbia rows in the corpus | 248,642 | 16,674 |
| corpus rows | 339,702 | 93,721 |
| training rows | 233,957 | 200,985 |

The uncapped variant was also trained and scored: it reached a higher
in-distribution macro-F1 (0.9786 against 0.9711) and a lower held-out-university
accuracy (0.7293 against 0.7830), which is why the cap exists. Those two numbers
are a working measurement rather than a citable artifact — its checkpoint was
overwritten when the corpus changed, so only the capped rows below can be
re-scored from what is on disk. It still does not restore v1's transfer number,
which is the honest caveat at the end of this document.

The extension is **additive by construction** (`pace.field_corpus_ext`): the base
corpus and its `train`/`validation`/`test` group assignment are copied verbatim,
new groups are assigned to `train`, and a new row is dropped if its text already
carries a different label in the base corpus. The 7,571 test rows are byte-identical
to the rows the first run was scored on, so the columns of the results table are
the same documents.

### What coverage looks like now (training rows per field)

| Field | v1 | v2 | Field | v1 | v2 |
|---|---:|---:|---|---:|---:|
| meeting_days | 20 | **311** | course_description | 19,276 | 22,300 |
| meeting_time | 1,186 | **8,910** | prerequisites | 4,534 | 6,810 |
| class_meeting_room | 802 | **3,031** | program | 1,053 | 1,766 |
| class_type | 18 | **69** | restrictions | 294 | 1,227 |
| credits | 371 | 554 | course_schedule | 275 | 434 |
| term | 158 | 188 | materials | 199 | 123 |
| | | | office_hours | 140 | 91 |

The last two rows go *down*: those are the placeholder-body rows being removed,
not data being lost. The measurement says that helped.

### The day-and-time blind spot is closed

The first run read bare `MWF 9:00-9:50 AM` as `course_code` (0.997) because no
training row started with a day token. Two things changed: Columbia prints
`Days: TR` and `Time: 4:00pm - 5:30pm` for 37 terms, and the mix gained a
`schedule` family that prints a real day, a real clock span and a real room
together (`TR 4:00pm - 5:30pm, 310 Mudd`), labelled by what the line *is* — a
meeting time. Days on their own stay `meeting_days`. Every component is a real
catalog cell; only the combination is new.

| Text | v1 | v2 |
|---|---|---|
| `MWF 9:00-9:50 AM` | course_code (0.997) | **meeting_time (0.9997)** |
| `TR 9:30-10:45 a.m., Journalism 214` | meeting_time | meeting_time (0.9997) |
| `Days: MWF` | meeting_days | meeting_days (0.9982) |

## Training

### Masked-LM pretraining

1.5 M passages per epoch from the extended corpus plus the uncapped corpus, three
epochs (4.5 M passages), max length 160, whole-word masking redrawn every epoch,
batch 128, AdamW at 5e-4 with 5% warmup and cosine decay, bf16 on an RTX 4070.

| Epoch | Masked-token loss | Time |
|---:|---:|---:|
| 1 | 4.02 | 1,816 s |
| 2 | 2.73 | 1,690 s |
| 3 | 2.46 | 1,742 s |

72 M tokens seen, on a corpus that now carries 18.7 k more real documents than the
first run's. The pretraining checkpoint is kept separately
(`artifacts/jev-field-v2-pretrain/`) so a fine-tune never has to repeat it.

### The decision objective

200,985 training rows, 42 described options, option dropout p = 0.15, label
smoothing 0.05, batch 192, lr 8e-4, early stopping on validation macro-F1 (best
epoch 6 of 9).

Two recipe changes, both aimed at the thin classes:

- **class-balanced sampling** (`--sampler-power 0.5`): a row of a field with *n*
  training rows is drawn with weight `n^-0.5`. The epoch stays the same size; only
  the batches change. Reweighting the loss, which the first run tried, does not
  decide which rows share a batch; sampling does.
- **sharper menu descriptions for the thin fields.** `attendance_policy` now names
  excused absences and participation requirements, `office_hours` carries an
  example, `exam_schedule` names midterms and durations. This is the Jev lever:
  the option is text, so sharpening the text sharpens the decision without
  touching the encoder.

### The ablation that explains most of the gain

| Run | Train rows | Test accuracy | Test macro-F1 | Checkpoint |
|---|---:|---:|---:|---|
| v1 (10.3 M, plain CE) | 171,123 | 0.9982 | 0.9155 | `artifacts/jev-field` |
| v2, placeholder rows **kept** | 234,795 | 0.9978 | 0.9485 | `artifacts/jev-field-v2-bodynoise` |
| v2, placeholder rows **removed** (shipped) | 200,985 | **0.9982** | **0.9711** | `artifacts/jev-field-v2` |

The middle row is a real run kept on disk so the difference is measured rather
than asserted: removing 838 rows' worth of misplaced headings, plus the corpus
cap, moved macro-F1 2.3 points beyond what tripling the row count bought.

## Results

### Same 7,571 held-out rows

| Model | Params | Test accuracy | Macro-F1 (38 classes) | Weighted-F1 | ECE-10 | Errors |
|---|---:|---:|---:|---:|---:|---:|
| **jev-small v2 (this)** | **15.3 M** | **0.9982** | **0.9711** | 0.9981 | 0.0016 | **14** |
| bge-small-en-v1.5 (same split) | 33.4 M | **0.9982** | 0.9713 | **0.9981** | **0.0012** | **14** |
| jev-small v1 | 10.3 M | **0.9982** | 0.9155 | 0.9982 | 0.0007 | **14** |

Same errors, same accuracy, macro within 0.0002 — at 46% of the reference's
parameters — and 5.6 macro points above the first run.

### Where the long tail stands

| Field | Test rows | v1 F1 | v2 F1 | bge F1 |
|---|---:|---:|---:|---:|
| materials | 4 | 0.615 | **0.889** | 0.727 |
| attendance_policy | 2 | 0.000 | **0.667** | 0.667 |
| disability_services | 1 | 0.500 | **1.000** | 1.000 |
| office_hours | 13 | 0.818 | 0.917 | 0.917 |
| grading_policy | 16 | 0.941 | 0.941 | 0.941 |
| assignments | 56 | 0.991 | 0.982 | 0.956 |
| prerequisites | 234 | 1.000 | 1.000 | 0.998 |
| course_content | 560 | 0.999 | 0.999 | 0.996 |
| **exam_schedule** | 5 | **0.889** | 0.571 | 0.750 |
| additional_information | 138 | 0.989 | 0.975 | 0.993 |

`attendance_policy` is the class that decides the last two points of macro: it has
**8** clean training rows and one of its two test rows is itself a `section body:`
paragraph whose heading is gone. The model gets the one that starts `Attendance:`
and misses the heading-less one, which is the same one bge misses. `exam_schedule`
went the other way — 5 test rows, one flip, 0.317 of the class's F1.

### A university the model has never seen

Fold 0 holds out MIT, Penn State, Ohio State, Berkeley and UCLA; training uses the
other universities only, and early stopping picks its epoch on a slice of the
training universities.

| Model | Fold-0 accuracy | Fold-0 macro-F1 |
|---|---:|---:|
| jev v1 (10.3 M, 171 k rows, plain CE, 10 epochs) | **0.8322** | **0.6391** |
| bge-small (same fold data) | 0.8300 | 0.6165 |
| jev v2, class-balanced sampler (shipped), 9 epochs | **0.7830** | **0.5659** |
| jev v2, plain CE, 10 epochs | 0.7629 | 0.4805 |

This is the one number that got worse. Corpus dominance is part of it — the
uncapped corpus was 73% one registrar and transferred at 0.7293, which is why the
cap exists. The rest is consistent with capacity: the first run's own report found
that a 149 M encoder transferred no better than a 33 M one, so this axis is about
data diversity, not parameters. The remaining gap is **not** isolated by an
experiment in this run, so it is stated as open rather than explained.

What the fold does show is that the long-tail fix is not free-standing: the same
sampler that lifts `materials`, `attendance_policy` and `office_hours`
in-distribution spends accuracy on schools whose layout it has not seen. It is
not the main cost, though — plain cross-entropy at the same size and corpus is
*worse* on the fold (0.7629 vs 0.7830), so the sampler is not what regressed
transfer; the encoder size and the new data are the remaining suspects, and this
run does not separate them.

## Infinite Campus

Infinite Campus has no single fixed CSV header: its Ad Hoc reports export whichever
columns a district ticks, and its sanctioned interchange format is **OneRoster
1.1** (`courses.csv`, `classes.csv`, `lineItems.csv`, `results.csv`). So
`pace.sis_import` maps both — the OneRoster headers by file, and district exports
by a normalised header-alias table (`Course Number`, `courseMasterNumber`,
`Teacher Name`, `Period`, `Max Enrollment`, …). A column outside the tables is
skipped rather than guessed, placeholders produce no row, and a section shares the
group id of its course so a split cannot leak.

```bash
python -m pace.sis_import --input exports/ --school "Lincoln High" \
  --output data/processed/sis/sis_corpus.jsonl
python -m pace.field_corpus_ext --sis data/processed/sis/sis_corpus.jsonl
```

Honest limit: no real district export is public — the OneRoster fixtures in
`tests/test_sis_import.py` are synthetic but format-accurate — so the adapter is
tested on the format, and the training corpus still contains no K-12 rows. Point
it at a real export and the rows join the corpus on the next rebuild.

## What it still gets wrong

- **`section body:` rows are unlearnable by construction.** 14 test rows carry a
  placeholder heading; the model cannot recover a heading that is not there.
- **A single-source corpus buys in-distribution fit and costs transfer.** Measured
  above: one source can become most of the corpus (73% here) and the fold pays for
  it, which is why new rows are capped per field.
- **Very thin fields remain a corpus problem.** `attendance_policy` has 8 clean
  training rows; `credits` loses a bare `3` read as `instructor`.
- **The menu is not zero-shot for new fields.** Re-describing a trained field
  works (there is a test); inventing a field with no examples does not.

## Reproduction

```bash
# 1. new real sources (Columbia 37 terms, Coursera) and the additive extension
python -m pace.field_corpus_ext
python -m pace.field_mix --context data/processed/field_context_ext/corpus.jsonl \
  --split-dir data/processed/field_context_ext --output data/processed/field_mix_ext \
  --extra-values data/processed/field_uncapped/corpus.jsonl --value-diversity-cap 20000
# 2. vocabulary + masked-LM pretraining (1.5 M passages x 3 epochs, ~87 min on a 4070)
python -m pace.jev_train --stage tokenizer --output artifacts/jev-field-v2 \
  --unlabelled data/processed/field_context_ext/corpus.jsonl data/processed/field_uncapped/corpus.jsonl
python -m pace.jev_train --stage pretrain --output artifacts/jev-field-v2 \
  --unlabelled data/processed/field_context_ext/corpus.jsonl data/processed/field_uncapped/corpus.jsonl \
  --pretrain-rows 1500000 --pretrain-epochs 3 --pretrain-max-length 160 \
  --hidden 320 --ffn 1024 --batch-size 128
# 3. the decision objective, balanced for the long tail
python -m pace.jev_train --stage choice --output artifacts/jev-field-v2 \
  --encoder artifacts/jev-field-v2-pretrain --tokenizer artifacts/jev-field-v2/tokenizer.json \
  --train data/processed/field_mix_ext/train.jsonl \
  --validation data/processed/field_mix_ext/validation.jsonl \
  --test data/processed/field_mix_ext/test.jsonl \
  --epochs 10 --batch-size 192 --lr 8e-4 --patience 3 --option-dropout 0.15 \
  --sampler-power 0.5 --hidden 320 --ffn 1024
# 4. score it, next to the reference on the same rows
python -m pace.jev_eval --checkpoint artifacts/jev-field-v2 \
  --rows data/processed/field_mix_ext/test.jsonl \
  --baseline artifacts/field-classifier-mix-fair \
  --output artifacts/jev-field-v2/test_metrics.json
# 5. a university it has never seen
python -m pace.field_mix --context data/processed/field_context_ext/corpus.jsonl \
  --split-dir data/processed/field_context_ext --output data/processed/field_mix_ext \
  --fold 0 --fold-plan data/processed/field/fold_plan_ext.json \
  --extra-values data/processed/field_uncapped/corpus.jsonl --value-diversity-cap 20000
python -m pace.jev_cross --encoder artifacts/jev-field-v2-pretrain \
  --folds-root data/processed/field_mix_ext --output artifacts/jev-cross-v2 --fold-ids 0 \
  --epochs 9 --patience 2 --sampler-power 0.5 --hidden 320 --ffn 1024 --lr 1e-3
```

## Files

| Path | What it is |
|---|---|
| `pace/field_sources_ext.py` | Columbia and Coursera loaders (column → field, printed heading) |
| `pace/field_corpus_ext.py` | additive corpus extension with the per-field cap |
| `pace/sis_import.py` | Infinite Campus / OneRoster export adapter |
| `pace/field_mix.py` | composition families, including `schedule` and the placeholder rule |
| `pace/jev_train.py` | vocabulary, pretraining, choice fine-tuning, class-balanced sampler |
| `artifacts/jev-field-v2/` | checkpoint, tokenizer, metrics, training logs |
| `artifacts/jev-field-v2-pretrain/` | language-only weights, for a fresh fine-tune |
| `artifacts/jev-field-v2-bodynoise/` | the ablation run with placeholder rows kept |
| `artifacts/jev-cross-v2/fold-0/` | the unseen-university checkpoint and its metrics |
| `tests/test_field_extension.py` | loaders, additive rules, cap, schedule family |
| `tests/test_sis_import.py` | OneRoster and Ad Hoc mapping, grouping, CLI |
