# Synthetic PyTorch multitask model card

## Summary

Shared `sentence-transformers/all-MiniLM-L6-v2` PyTorch encoder (6 transformer layers kept) with a document-type head and a token BIO entity head over 19 entity types. This checkpoint is trained on procedural synthetic academic text only. It has **no training or validation on real OCR, PDFs, real syllabi, or institution documents** and is not for production or student-record decisions.

## Reproduction / data

- Seed 20261019; 30,000 generated examples; 2% length-preserving character-substitution augmentation. This is not image OCR.
- Split: disjoint school identities and disjoint layout families; only their intersections are evaluated. Train 10,279 / validation 1,069 / test 1,208 documents (42/9/9 schools, 20/10/11 template families); sliding-window rows 12,098 / 1,207 / 1,551. All ten document classes occur in each split. The splitter fails rather than allowing identity leakage when sample size is insufficient.
- CUDA training, 8 epochs, batch size 64, max length 160, entity loss weight 2.5. Checkpoint selected by highest validation exact entity-span F1 (epoch 8 of 8; validation F1 0.99936).
- 22,732,081 parameters; checkpoint is 90,958,791 bytes (86.75 MiB). Dynamic INT8 checkpoint is 58,654,218 bytes (0.645x of FP32) with entity-span F1 0.99826.

## Held-out results

Document type (10 classes), corrupted test:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Shared PyTorch multitask MiniLM | 1.000 | 1.000 |
| TF-IDF + logistic regression, identical examples/split | 1.000 | 1.000 |

This synthetic document task is cue-easy and has no practical evidence of real-world generalization.

Entity exact span (type + character offsets), test set of 1,208 held-out documents:

| Text condition | Gold spans | Predicted | Precision | Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|
| Clean (no corruption) | 20,382 | 20,393 | 0.99941 | 0.99995 | 0.99968 |
| Fully corrupted (every character may be mistyped) | 20,382 | 20,413 | 0.99833 | 0.99985 | 0.99909 |
| Context-only corrupted (entity values intact) | 20,382 | 20,397 | 0.99917 | 0.99990 | 0.99953 |

`corrupted` and `corrupted_context_only` are reported separately on purpose: in the fully corrupted condition a mistyped value tokenizes differently, so exact-character agreement for that value is not expected even for a perfect reader. The labels are generated automatically, not human adjudicated, so these figures are limited to this procedural generator.

Per-template-family corrupted span F1: 1.000 for gradebook, other, report card, requirements, schedule and course selection; 0.9997 syllabus narrative; 0.9994 transcript piped; 0.9987 transcript table; 0.9972 catalog entry; 0.9909 assignment notice (lowest family).

Per-entity-type corrupted F1 (lowest first): STUDENT 0.9916, ASSESSMENT 0.9947, COURSE 0.9966, PERFORMANCE_LEVEL 0.9971, MARK 0.9991, SUBJECT_AREA 0.9991, GRADE_CATEGORY 0.9992, COURSE_CODE 0.9996, CREDIT 0.9998; TEACHER, ROOM, WEIGHT, DATE, TIME, PREREQUISITE, SCHOOL, GRADE_LEVEL, SCORE and GPA are at 1.000.

Full report, per-class support/precision/recall/F1, document confusion matrices, histories and predictions live in [`metrics.json`](metrics.json). The exact split examples are `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

## Model size vs accuracy (depth study)

Same seed, same generated corpus, same school/template hold-out; the only difference is how many transformer layers the shared encoder keeps.

| Encoder layers | Parameters | Checkpoint | INT8 checkpoint | Clean span F1 | Corrupted span F1 | Transcript span F1 | CUDA window p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 22,732,081 | 90,958,791 B | 58,654,218 B | 0.99968 | 0.99909 | 0.99905 | 5.08 ms |
| 4 | 19,183,153 | 76,754,848 B | 55,052,946 B | 0.99951 | 0.99890 | 0.99878 | 3.58 ms |
| 2 | 15,634,225 | 62,550,848 B | 51,451,701 B | 0.99848 | 0.99674 | 0.99742 | 2.23 ms |

The 2-layer model is 31.2% smaller than the 6-layer one and gives up 0.24 F1 points of corrupted span F1; the 4-layer model saves 15.6% of the checkpoint for 0.02 F1 points. The 6-layer checkpoint is the promoted default. Full numbers: [`depth_comparison.json`](depth_comparison.json).

## Image-derived transcript run (single document, not a benchmark)

`data/samples/mock_transcript_lepid_rowwise.txt` is the text read from the supplied transcript image. Run through the promoted checkpoint (`artifacts/synthetic-multitask/transcript_report_rowwise.json`):

- Document type: `transcript` (p = 0.9929; every other class below 0.001).
- Model entity spans on that document: 220 gold / 220 predicted / 220 true positives — precision 1.000, recall 1.000, F1 1.000 across 10 types (COURSE, CREDIT, DATE, GRADE_LEVEL, MARK, PERFORMANCE_LEVEL, SCHOOL, SCORE, STUDENT, SUBJECT_AREA).
- Parser records: 38 course records over 8 course subject areas (11 sections in the document; the other 3 are assessment sections), 3 assessment records (AK HSGQE reading/writing/math scores 337 / 273 / 258), computed total credits 13.25, reported total credits 13.25, credit sum matches reported total (difference 0.0), 0 records without a title or a section, 0 unattached performance levels. GPA is absent from the document and is reported as `null`, not invented.

Entity reference spans and record counts come from the deterministic parser, which is independent of the neural tagger; the document's own reported totals are used as the consistency check. This is one document, not a benchmark.

## Inference output and confidence

`python -m pace.cli analyze FILE --checkpoint artifacts/synthetic-multitask/model.pt --synthetic-pytorch` runs the PyTorch checkpoint through a sliding window and emits document classification, heading/paragraph splits, typed span proposals and separate literal candidates. All outputs require human review and are proposal-only.

Document decisions expose the normalized softmax distribution over all ten document classes; `confidence` is a separate Jev Choice sharpness statistic, `(p_max - 1/n) / (1 - 1/n)`. For each model entity span, `type_probabilities` normalizes the geometric-mean BIO tag evidence over all entity types, conditional on the proposed source boundaries, and `confidence` applies the same Choice formula. These are distribution summaries over neural scores—not probabilities validated against correctness. `confidence_calibrated` and `probabilities_calibrated` are false for every model decision. Literal-pattern fallback candidates are explicitly untyped and use `literal_pattern_match_not_model_confidence`, not a model Choice. Letter-grade candidates are emitted only from an explicit grade/grade-scale context; standalone letters in ordinary prose are not treated as grades.

**Calibration is blocked:** the repository contains no independent human-labeled calibration set for this synthetic PyTorch checkpoint. The existing `artifacts/calibration.json` is validation-only temperature scaling for a different classifier and is not appropriate here. No calibration data or calibrated claims were invented; a separate human-reviewed, representative set and reliability evaluation are required before model probabilities or confidence can be called calibrated.

## Limitations and intended use

- Synthetic generator cues, automatic labels, narrow vocabulary/styles, no image rendering or OCR.
- No real-document or institution-held-out evaluation and no human-verified span gold.
- **Calibration blocker:** no independent human-labeled calibration set exists for this checkpoint; no reliability study, validated abstention threshold, or risk/coverage curve is available. Choice confidence is an uncalibrated distribution-sharpness statistic only.
- The generator is easier than reality: the document-type task is saturated at 1.000 for both the neural model and a TF-IDF baseline, and span F1 is above 0.99 on every template family. These numbers do not transfer to real scanned documents.
- Output must be checked against the source; no academic record is changed.
- PyTorch only. Core ML / ONNX / ExecuTorch / mobile work is out of scope.

Suitable for experimenting and exercising local PyTorch inference—not operational decisions.

## Train

```bash
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask --count 30000 --seed 20261019 --epochs 8 --batch-size 64 --max-length 160 --corruption-rate 0.02
```

Depth variants:

```bash
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask-l4-v8 --encoder-layers 4 --count 30000 --seed 20261019 --epochs 8
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask-l2-v8 --encoder-layers 2 --count 30000 --seed 20261019 --epochs 8
python -m pace.compare_depths
```
