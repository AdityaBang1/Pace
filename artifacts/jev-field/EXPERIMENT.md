# A Jev-shaped decision model, trained backwards onto this corpus

`artifacts/jev-field` is a 10.28 M-parameter model that answers one question:
*given a piece of text and a menu of described academic fields, which description
does this text satisfy?* It is built in the shape of TypeSafe's Jev "system one"
classifier and the open TinyDecide reproduction of it, then trained from scratch
on this project's own corpus. It replaces a 42-way output layer with 42 written
descriptions and one cross-encoder pass.

The reason to bother: a 42-way softmax can only answer the 42 classes it was
trained on, and its scores are a property of the classifier rather than of the
text. Describing the options instead means the same text tower can be pointed at
a different menu, and the model has to earn its score from what the option says.

## Shape (TinyDecide's, so the comparison is at the architecture level)

| | Value |
|---|---|
| Parameters | 10,278,955 (encoder 10,048,512; heads 230,443) |
| Layers × width | 12 × 256, pre-LayerNorm |
| Attention | 4 heads |
| Feed-forward | 768, GELU |
| Token embeddings | 128, tied to the MLM head through a 256→128 projection |
| Vocabulary | 16,000 WordPiece, trained on this corpus, lowercased |
| Positions | 512 |
| State budget | 127 tokens |
| Option budget | 24 tokens per description |
| Checkpoint | 41.2 MB fp32 on disk; ≈10 MB if exported to int8 (parameter count × 1 byte, no int8 artifact is committed) |

One pass encodes `[CLS] state [SEP] question [SEP]`; the state is mean-pooled
over its own tokens only. Options are scored from their description's token
embeddings (an embedding bag plus a small head, no transformer pass), which is
why adding an option is free. The score is a scaled cosine plus a per-label
bias; a single temperature is fitted on validation.

## Pretraining: the language this corpus actually uses

A 10 M encoder cannot arrive with a large model's English baked in, so the first
stage is masked-language-model pretraining on the project's own text — 1.2 M
passages per epoch from `data/processed/field_uncapped/corpus.jsonl`, three
epochs, **37.1 M tokens** seen. Masking is whole-word (every WordPiece piece of a
chosen word, plus the following word 20% of the time), and the mask is redrawn
every epoch, so the same passage is never memorised with the same holes.

| Epoch | Masked-token loss | Time |
|---:|---:|---:|
| 1 | 3.86 | 681 s |
| 2 | 2.66 | 691 s |
| 3 | 2.35 | 681 s |

Batch 128, AdamW, lr 5e-4 with 5% warmup and cosine decay, bf16 autocast on an
RTX 4070. A million-row dataset is stored as one flat int16 buffer plus offsets
(37 MB) instead of per-row objects; the object version was several GB and
roughly 4× slower to iterate.

## Fine-tuning: the decision objective

State is the text (heading and value together when both exist), the question is
fixed ("Which described academic field does this text come from?"), and the menu
is the 42 descriptions in `pace.jev_model.FIELD_DESCRIPTIONS`. Training rows are
`data/processed/field_mix/{train,validation,test}.jsonl` — 171,123 / 7,896 /
7,571 real rows, the same split the bge-small reference was trained on.

One choice beyond plain cross-entropy: **option dropout (p = 0.15)**. Each step
hides a random slice of the menu (never the true answer) and renormalises, so the
model cannot lean on a fixed 42-way shortcut and stays usable when a caller
passes a shorter menu. The loss is a label-smoothed cross-entropy over the
surviving options.

10 epochs, batch 192, lr 8e-4, early stopping on validation macro-F1 (best epoch
7), 1,044 s. The fitted temperature is 0.579.

### Ablation: class-balanced weights

The long tail is thin — `attendance_policy` has 17 training rows,
`disability_services` 43, `materials` 199, against 19,533 for `course_code`. An
inverse-frequency weight (power 0.5) on the same pretrained encoder, re-run from
the language checkpoint:

| Run | Val macro-F1 (best) | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| shipped (plain CE) | **0.9723** | **0.9982** | 0.9155 |
| class-weighted (power 0.5) | 0.9560 | 0.9975 | **0.9439** |

It moves macro-F1 around by ±3 points and costs 5 extra errors; the two runs hit
different rare classes, which is what 1–13 test rows per rare class buys you. The
shipped checkpoint is the one with the better validation score — the rule used
for every other checkpoint in this repository. Both are recorded:
`artifacts/jev-field/choice-balanced.log` and
`artifacts/jev-field/choice.log`.

## Results

### Same split, same rows (7,571 held-out rows)

| Model | Params | Test accuracy | Macro-F1 | Weighted-F1 | ECE-10 | Train time |
|---|---:|---:|---:|---:|---:|---:|
| **jev-small (this)** | **10.3 M** | **0.9982** | 0.9155 | 0.9982 | **0.0007** | 3,098 s (pretrain + fit) |
| bge-small-en-v1.5, trained on this split | 33.4 M | 0.9982 | **0.9713** | 0.9982 | 0.0012 | 496 s |
| all-MiniLM-L6-v2, the project's earlier checkpoint | 22.7 M | 0.9464 | 0.8486 | 0.9457 | 0.0166 | — |

Both models make **14 errors** on the 7,571 rows. (bge's own metrics file reports
macro-F1 0.9691; the 0.9713 above is the same checkpoint rescored through
`pace.jev_eval` so both numbers average over exactly the same labels.) The macro-F1 gap is not a
capacity gap: it is five classes with 1–13 test examples each (`attendance_policy`
2, `disability_services` 1, `materials` 4, `office_hours` 13, `exam_schedule` 5),
where one mistake moves a class F1 by 0.1–0.3. Restricted to the 22 classes with
at least 30 test rows, the ranking reverses:

| Macro-F1 over classes with support ≥ | Jev | bge-small |
|---|---:|---:|
| 30 rows (22 classes) | **0.9988** | 0.9972 |
| 10 rows (28 classes) | 0.9894 | **0.9917** |
| all 38 present classes | 0.9396 | **0.9691** |

### A university the model has never seen (3 disjoint folds, 1,295 rows)

| Fold | Held-out universities | Jev acc | bge acc | Jev macro-F1 | bge macro-F1 | Jev ECE | bge ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | MIT, Penn State, Ohio State, Berkeley, UCLA | **0.8322** | 0.8300 | **0.6391** | 0.6165 | 0.1276 | 0.1490 |
| 1 | Georgia Tech, Stanford, UT Austin, UNC, Wisconsin | **0.8896** | 0.8792 | **0.6863** | 0.6743 | **0.0722** | 0.0989 |
| 2 | CMU, Purdue, UIUC, Michigan | **0.8995** | 0.8913 | **0.7691** | 0.7482 | **0.0660** | 0.0919 |
| mean | | **0.8738** | 0.8668 | **0.6982** | 0.6796 | **0.0886** | 0.1133 |

The 10.3 M model wins every fold on both accuracy and macro-F1, and is better
calibrated on unseen institutions. Each fold is trained on the other universities
only, with early stopping on a slice of the training universities — the same
protocol the bge reference used (`artifacts/field-mix/fold-*-final`).

### What it does on real snippets

Eight unseen lines, run through `JevEngine.classify_many`:

| Text | Label | p |
|---|---|---:|
| Prerequisite(s): MATH 151 or equivalent with a grade of C or better. | prerequisites | 1.000 |
| Credits: 4.00 Contact Hours: 3 lecture, 2 lab | credits | 0.999 |
| Grading: Homework 25%, Midterm 25%, Final 40%, Participation 10% | grading_policy | 0.998 |
| Students with disabilities should contact the Office of Accessibility Services. | disability_services | 0.975 |
| Academic integrity: all work must be your own; plagiarism results in a zero. | academic_integrity | 0.997 |
| Required text: Campbell Biology, 12th edition | materials | 0.998 |
| Instructor: Dr. Maya Chen (mchen@university.edu), office SCI 312 | instructor | 0.999 |
| TR 9:30-10:45 a.m., Journalism 214 | course_code | 0.971 |

## What it gets wrong, measured

- **A menu of genuinely new fields.** Descriptions are what makes the menu
  swappable, but the option head only ever saw the 42 training descriptions.
  Asking for three invented options (`thesis deadline`, `final exam`, `tuition`)
  gets the exam right (0.99) and the other two wrong. Re-describing the trained
  fields works (the option text, not a class id, carries the score — there is a
  test for this); inventing a field without examples does not.
- **A day-and-time string with no heading.** No training row starts with "MWF" or
  "TR"; day values in the corpus are `Days: MWF` and time values are
  `meeting time: 8:25PM - 9:50PM`. Bare `MWF 9:00-9:50 AM` is read as
  `course_code` (0.997); `Class meets TTh 2:00-3:15pm in Room 214` is read
  correctly (0.989), and `9:30-10:45 a.m.` alone is `meeting_time` (0.990).
  This is a coverage gap, not an architecture failure.
- **Placeholder rows.** 8 of the 14 in-distribution errors are `section body:`
  rows — a paragraph cut out of a section, carrying a label the text does not
  contain — the same failure `artifacts/field-mix/EXPERIMENT.md` documents for the
  encoder reference. 7 of the 14 errors are above p = 0.9; the most confident are
  one duplicated `office_hours` line read as `instructor` (0.999).
- **Three fields have no test examples at all** (`drop_add_deadline`,
  `attendance_policy` has 2, `disability_services` 1). The macro numbers above
  inherit that noise; the cross-institution table is the more stable estimate.

## Reproduction

```bash
# 1. vocabulary + masked-LM pretraining (1.2M passages, 3 epochs, ~35 min on a 4070)
python -m pace.jev_train --stage tokenizer --output artifacts/jev-field
python -m pace.jev_train --stage pretrain  --output artifacts/jev-field \
  --pretrain-rows 1200000 --pretrain-epochs 3
# 2. the decision objective (42 described options, option dropout)
python -m pace.jev_train --stage choice --output artifacts/jev-field \
  --train data/processed/field_mix/train.jsonl \
  --validation data/processed/field_mix/validation.jsonl \
  --test data/processed/field_mix/test.jsonl \
  --epochs 16 --batch-size 192 --lr 8e-4 --patience 3 \
  --option-dropout 0.15 --class-weight-power 0.5
# 3. score it, next to the reference on the same rows
python -m pace.jev_eval --checkpoint artifacts/jev-field \
  --rows data/processed/field_mix/test.jsonl \
  --baseline artifacts/field-classifier-mix-fair \
  --output artifacts/jev-field/test_metrics.json
# 4. universities it has never seen
python -m pace.jev_cross --encoder artifacts/jev-field --output artifacts/jev-cross \
  --folds 3 --epochs 10 --patience 3
# 5. serve it like any other field checkpoint
python -m pace.cli classify-field "Prerequisites: CSCI 270 or equivalent" \
  --checkpoint artifacts/jev-field
```

## Files

| Path | What it is |
|---|---|
| `pace/jev_model.py` | encoder, choice model, engine (`classify`, `classify_many`, `score_against`) |
| `pace/jev_train.py` | vocabulary, masked-LM pretraining, choice fine-tuning |
| `pace/jev_eval.py` | accuracy/F1/ECE/confusions, optional same-rows baseline |
| `pace/jev_cross.py` | per-fold unseen-university training and scoring |
| `tests/test_jev.py` | shape, masking, reload, menu-swap, loss and option-dropout properties |
| `artifacts/jev-field/` | checkpoint, tokenizer, metrics, training log |
| `artifacts/jev-field-pretrain/` | language-only weights (`model.pt`) + `pretrained/`, for a fresh fine-tune |
| `artifacts/jev-cross/` | three fold checkpoints + `cross_institution_report.json` |
| `artifacts/field-classifier-mix-fair/` | bge-small retrained on the identical split |
