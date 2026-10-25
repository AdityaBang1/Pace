# Experiments

## Headline experiment: real-text academic field classification

The field classifier is the project's main line of work: label a piece of academic text with the **field** it came from, over 42 types (`credits`, `office_hours`, `exam_schedule`, `attendance_policy`, `prerequisites`, …), trained on **163,797 real rows** from eight public datasets. No synthetic documents, no paraphrasing, no generated values.

### The task, and why it is not trivial

A value's field is not a property of the value. `"3.0 units"` is `credits` only because of the column it came from; `"Mon, Wed"` is `meeting_days` only because of the column. A text-only model trained on loose values therefore learns **each institution's layout**, and it shows: hold out a whole dataset and accuracy collapses (MIT 0.63, USC 0.46, UW 0.41, syllabi 0.21, GU 0.11). Hold out whole universities and accuracy is 0.57–0.70 with ECE 0.22–0.33, against 0.989 in-distribution.

The fix is to read the document the way a person does. On 446 real labelled sections, the **heading alone** is 88.1% accurate while the **body alone** is 69.3% accurate, and naive concatenation (75.8%) is *worse* than the heading alone because the body dominates the token budget. A model trained on heading-plus-value pairs and a heading-first cascade both exploit that.

A heavier attempt at the same number — four extra label-preserving views of the real text (including headings borrowed from a *different* source), three encoders from 33 M to 395 M parameters, and longer schedules — ties the reference on top-1 accuracy over the same 1,295 held-out rows (0.8726 vs 0.8695 for the 149 M model) while buying +1.75 macro-F1. The model that buys them is 150 MB, 14× slower per section, and loses one label in four to int8, so the deployable checkpoint stays the 33 M one. Splitting the hard fold by what the model was fed explains the plateau: rows with a real printed heading are at 94% for the reference and 94% for every bigger encoder, rows that are published document facts are at 100%, and rows whose heading is the placeholder `section body` — a paragraph cut out of a section — are at 39–43% for all of them. The aggregate is flat because a quarter of the benchmark is not answerable as posed. The protocol, per-fold table, that split, the quantization measurements and the reason more parameters do not fix transfer are in [artifacts/field-mix/EXPERIMENT.md](../artifacts/field-mix/EXPERIMENT.md).

### A Jev-shaped decision model, trained from scratch on this corpus

The deployable checkpoint above is a fine-tuned MiniLM/bge encoder with a 42-way
head. `artifacts/jev-field` asks the same question a different way: 42 *described*
options, one cross-encoder pass, and a 10.3 M model built in the shape of Jev /
TinyDecide (12 × 256, 4 heads, 16 K WordPiece trained here). It is pretrained on
37 M tokens of this project's own text with whole-word masking, then fine-tuned on
the same split with option dropout.

On the same 7,571 rows it makes the **same 14 errors** as a 33.4 M bge-small
retrained on that split (0.9982 accuracy each), is better on the 22 well-supported
classes (macro-F1 0.9988 vs 0.9972), and is better calibrated (ECE 0.0007 vs
0.0012). On universities it has never seen it wins **every one of the three folds**:
0.8738 accuracy / 0.6982 macro-F1 against 0.8668 / 0.6796, at a third of the
parameters (ECE 0.0886 vs 0.1133). The full recipe, the class-weighted ablation,
the invented-menu experiment that fails, and the coverage gaps are in
[artifacts/jev-field/EXPERIMENT.md](../artifacts/jev-field/EXPERIMENT.md).

```bash
python -m pace.jev_train --stage all --output artifacts/jev-field \
  --pretrain-rows 1200000 --pretrain-epochs 3 --epochs 16 --batch-size 192 \
  --lr 8e-4 --patience 3 --option-dropout 0.15
python -m pace.jev_eval --checkpoint artifacts/jev-field \
  --rows data/processed/field_mix/test.jsonl \
  --baseline artifacts/field-classifier-mix-fair \
  --output artifacts/jev-field/test_metrics.json
python -m pace.jev_cross --encoder artifacts/jev-field --output artifacts/jev-cross \
  --folds 3 --epochs 10 --patience 3
python -m pace.cli classify-field "Prerequisites: CSCI 270 or equivalent" \
  --checkpoint artifacts/jev-field
```

### v2: more data, a bigger encoder, and the label noise removed

`artifacts/jev-field-v2` is the same architecture at **15.3 M** parameters, pretrained on
4.5 M passages (72 M tokens) and fine-tuned on 200,985 rows. Three things changed and
one of them was not data: **838 training rows whose real heading had been lost** —
`section body: <paragraph>` labelled by a heading that is nowhere in the text, 63 of
`materials`' 185 rows and 45 of `office_hours`' — were withheld from training while
staying in validation and test.

On the same 7,571 rows it makes the **same 14 errors** as the 33.4 M bge-small reference
(0.9982 accuracy each) with macro-F1 **0.9711 vs 0.9713** and ECE 0.0016 vs 0.0012, at
46% of the parameters; against v1 that is **+5.6 macro points** (0.9155 → 0.9711), with
`materials` 0.615 → 0.889, `attendance_policy` 0.000 → 0.667, `office_hours` 0.818 →
0.917. Bare `MWF 9:00-9:50 AM`, read as `course_code` by v1, is now `meeting_time` at
0.9997 after Columbia's 37-term directory added real day strings and the corpus gained a
`schedule` family that prints real days, times and rooms together.

The trade-off is measured rather than hidden. Left uncapped, Columbia is 73% of the
corpus's rows, so each new field is capped at 2,500 rows — the coverage survives, the
bulk repetition does not. Even so, the held-out-university fold lands at **0.7830**,
below v1's **0.8322**, and the report says so: plain cross-entropy at the same size is
worse still (0.7629), so the sampler is not the cost, and the run does not separate
encoder size from corpus shift. For an unfamiliar institution the first run's 10.3 M
checkpoint remains the better reader. The Infinite Campus / OneRoster adapter
(`pace.sis_import`) ships with the same work, tested on the format but not yet on a real
district export. See
[artifacts/jev-field-v2/EXPERIMENT.md](../artifacts/jev-field-v2/EXPERIMENT.md).

### Split design

| View | What is held out | What it answers |
|---|---|---|
| `group` | whole subjects / courses / documents | in-distribution accuracy |
| `cross_institution` | whole universities, 3 disjoint folds | does it work on a university it has never seen? |
| `by_source` | one dataset at a time | does it work on a schema it has never seen? |

Every split is group-wise, so no document contributes to both sides. In the cross-institution folds the early-stopping epoch is chosen on a slice of the **training** universities, never on the held-out ones, and the fold plan is written to `data/processed/field/fold_plan.json` and shared by both corpora so the bare and heading-aware models are measured on the same universities.

```bash
python -m pace.field_splits --output data/processed/field --rebuild
python -m pace.field_train \
  --train data/processed/field/group/train.jsonl \
  --validation data/processed/field/group/validation.jsonl \
  --test data/processed/field/group/test.jsonl \
  --output artifacts/field-classifier-minilm --epochs 6 --seeds 20261019 20261020
python -m pace.field_ood --checkpoint artifacts/field-classifier-minilm --output artifacts/field-ood
python -m pace.field_context --output data/processed/field_context
python -m pace.field_context_eval --checkpoint artifacts/field-classifier-minilm \
  --context-checkpoint artifacts/field-classifier-context
python -m pace.field_cross --output artifacts/field-crossinstitution --folds 3 \
  --fold-plan data/processed/field/fold_plan.json
python -m pace.field_pair
python -m pace.field_demo --limit 3
```

Full numbers, the failure decomposition and the deployment recipe are in [REPORT.md](../REPORT.md) and the [model card](../artifacts/field-classifier-minilm/MODEL_CARD.md).

## Primary experiment: real USC course-catalog cells

The current primary run uses natural field cell values from USC/USC-Course-Catalog, not synthetic fragments. Raw files, licensing caveats, row fields and length statistics are documented in `data/inventory.json`, `data/manifest.json`, and `docs/research.md`.

### Task and ambiguity treatment

The model identifies which source field a cell came from: `course_title`, `course_description`, `credit_units`, `prerequisites`, `restrictions`, `meeting_time`, `meeting_days`, or `class_type`. Exact normalized `TBA` appears in both Days (1,317) and Time (1,335), i.e. identical input text has conflicting labels. These 2,652 rows are excluded from training/evaluation because text alone cannot reveal which source column they came from. The deployed helper abstains on this exact normalized ambiguity. All other natural values are preserved as complete cells.

Split assignment is by course-subject prefix: 144 train subjects, 30 validation subjects, 31 held-out test subjects; 23,175 / 5,102 / 5,522 rows. Run the baseline first; it writes the fixed split assignment. Then pass that same file to each encoder run. This is one university/term CSV classification task, not scanned document OCR or multi-institution syllabus reading.

```bash
python -m pace.inspect_datasets
python -m pace.real_baseline --input data/raw/usc_course_data.csv --output artifacts/usc-real
python -m pace.real_train --input data/raw/usc_course_data.csv --epochs 4 --batch-size 64 --output artifacts/usc-miniLM --split-assignment artifacts/usc-real/split_assignment.json
python -m pace.real_train --input data/raw/usc_course_data.csv --epochs 4 --batch-size 64 --output artifacts/usc-deberta-xsmall --base-model microsoft/deberta-v3-xsmall --split-assignment artifacts/usc-real/split_assignment.json
python -m pace.compare_real
python -m pace.real_benchmark --runs 500
python -m pace.real_infer 'Prerequisite: BIO 101'
python -m pytest -q
```

### Corrected measured comparison

| Model | Accuracy | Macro-F1 | Weighted-F1 | Artifact size |
|---|---:|---:|---:|---:|
| Char 2–5 TF-IDF logistic | 0.997284 | 0.997415 | 0.997284 | 1,963,196 bytes |
| Fine-tuned MiniLM-L6 | 0.999819 | 0.999861 | 0.999819 | 90,893,327 bytes |
| Fine-tuned DeBERTa-v3-xsmall | 1.000000 | 1.000000 | 1.000000 | 282,794,003 bytes |

All compare on the same 5,522 unseen-subject cells. Residual test errors: sparse 15, MiniLM 1, DeBERTa 0. MiniLM is the practical neural candidate; DeBERTa's slight gain is at ~3.1x size and ~3x epoch time. Sparse remains ~46x smaller and very competitive. This is not yet evidence on actual syllabus documents, OCR noise, or other institutions. Full metrics and exact split in `artifacts/usc-real/`, `artifacts/usc-miniLM/`, and `artifacts/usc-deberta-xsmall/`.

Real validation calibration: MiniLM temperature 0.7242, test ECE-10 0.000088, >=0.90 coverage 100% / precision 99.982%; DeBERTa temperature 0.1102, test ECE 0, coverage and precision 100%. These measurements are specific to this structured distribution and single split, not guarantees under domain shift. `TBA` abstains deterministically.

### Inference timing

`artifacts/usc-miniLM/benchmark.json` records 120 input rows, batch one on RTX 4070, including tokenizer + PyTorch forward, excluding loading/OCR/app overhead: p50 2.739 ms, p95 3.024 ms, ~360 rows/s, input token p50 8, p95 34, CUDA peak allocated ~103 MB. This is desktop GPU data, not mobile performance. No runtime export or quantization has been completed.

### Failure analysis

An early benchmark showed repeated Days/Time errors. Inspecting original rows established every shared confusion involved literal `TBA` in both source columns. The corrected experiment removes underdetermined examples and inference abstains on the literal. The remaining MiniLM error is one course-description text that looks like a course title; see `test_error_examples` in the metrics. Do not generate synthetic fixes before determining whether the error is clinically important in realistic OCR/document context.

## Secondary experiment: synthetic-only multitask document understanding

Deliberately kept separate from the real-data numbers above. It exercises document-type classification (10 classes) plus token-level entity extraction (19 types) over 30,000 procedurally generated academic documents, with whole school identities and whole template families held out. **It is not real-document evidence.**

| Encoder layers | Parameters | Checkpoint | INT8 | Clean span F1 | Corrupted span F1 | Transcript span F1 | CUDA window p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 (default) | 22,732,081 | 90,958,791 B | 58,654,218 B | 0.99968 | 0.99909 | 0.99905 | 5.08 ms |
| 4 | 19,183,153 | 76,754,848 B | 55,052,946 B | 0.99951 | 0.99890 | 0.99878 | 3.58 ms |
| 2 | 15,634,225 | 62,550,848 B | 51,451,701 B | 0.99848 | 0.99674 | 0.99742 | 2.23 ms |

All three variants use the same corpus, seed and hold-out; only the number of transformer layers the shared encoder keeps differs. The 2-layer model is 31.2% smaller for 0.24 F1 points; the 4-layer model saves 15.6% of bytes for 0.02 points. Document type is 1.000 for the multitask model *and* for an identical-data TF-IDF baseline, which is the clearest sign that this generator is cue-easy.

Image-derived transcript run: document type `transcript` (p = 0.9929), 220/220 model entity spans matched, 38 course records across 8 course subject areas plus 3 assessment sections, computed credits 13.25 matching the reported 13.25, GPA absent and reported as null. One document, not a benchmark.

```bash
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask --count 30000 --seed 20261019 --epochs 8 --batch-size 64 --max-length 160 --corruption-rate 0.02
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask-l4-v8 --encoder-layers 4 --count 30000 --seed 20261019 --epochs 8
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask-l2-v8 --encoder-layers 2 --count 30000 --seed 20261019 --epochs 8
python -m pace.compare_depths
python -m pace.evaluate_transcript data/samples/mock_transcript_lepid_rowwise.txt --checkpoint artifacts/synthetic-multitask
```

Details: [model card](../artifacts/synthetic-multitask/MODEL_CARD.md) and [REPORT.md](../REPORT.md).

## Previous synthetic smoke test

A small deterministic, manually templated dataset and sparse baseline remain under `pace.data`, `pace.train`, `pace.baselines`. They were implemented before the course correction. Both sparse baseline and encoder scored 100% on its held-out paraphrases; this exposed that it is too easy and is not product evidence. Keep it separate from the real-data numbers above.

## Deployment research

Maintain normal PyTorch during model development. After quality is established, compare ExecuTorch (`torch.export` → target-specific `.pte`) and ONNX Runtime Mobile (ONNX, iOS/Android runtimes), including quantization and actual phone metrics. See official links and support details in `docs/research.md`. No Core ML/ExecuTorch/ONNX artifact has been produced or device-tested.
