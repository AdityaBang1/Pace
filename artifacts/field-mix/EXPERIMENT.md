# A heavier field-classifier run: what was trained, and what it did not fix

The reference checkpoint (`artifacts/field-classifier-minilm`, MiniLM-L6) is
trained on real catalog and syllabus rows and scores 0.99 in-distribution but
0.57–0.70 on universities it has never seen. This experiment asked whether a
much larger training effort — a stronger 12-layer encoder, four extra
label-preserving views of the same real text, and a longer schedule — improves
the number that matters: accuracy on **real syllabi from universities the model
has never seen**.

On top-1 accuracy it does not: the best heavy model is +0.31 points over the
reference on 1,295 held-out sections, which is a tie. It does buy **+1.75 points
of macro-F1** — fewer rare fields dropped — and the encoder that buys them is
**not deployable on a phone without quantization-aware training** (int8 changes
one label in four, and it is 14× slower than the small model). This document
records the protocol and the measurement so the small gain and the deployment
wall are both on the record.

## What was trained

| | |
|---|---|
| Encoder | `BAAI/bge-small-en-v1.5` (12 layers, 384 hidden, 33.4 M params) |
| Reference encoder | `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 22.7 M params) |
| Corpus | 133k–135k training rows per fold, built from 75,015 real heading+value rows and 1.4 M real catalog values |
| Schedule | 14 epochs max, patience 4 on validation macro-F1, bf16, batch 96, max length 160 |
| Early stop | fold 0 epoch 9, fold 1 epoch 4, fold 2 epoch 5 |
| Hardware | 1× RTX 4070, ~60–120 s per epoch, 5.9–6.4 GB peak |
| Total training | ~30 min for the three transfer folds, ~1.4 h of GPU time including the pilots and an abandoned `deberta-v3-base` attempt |

The encoder is 1.5× deeper than the reference and the corpus is 1.8× larger,
so this is a genuinely heavier run, not a re-seed.

## The four extra views

`pace.field_mix` composes training rows that keep every word real but break the
pairing a model can memorise. Each row is a real value from a real document; the
label comes from that document. Only the presentation is composed.

| Family | Rows | What it teaches |
|---|---:|---|
| `natural` | 59,080 | the row exactly as printed |
| `cross` | 13,133 | a **different source's real heading** above the same labelled value — the direct attack on "learned each institution's layout" |
| `short` | 10,209 | heading plus only the first 120 characters of the value, so a long body cannot outvote its heading |
| `value_diversity` | 45,830 | values the 6,000-per-type cap excluded, under their real source heading |
| `sentence` | 351 | one real sentence from a real syllabus section under that section's heading |
| `heading` | 137 | the heading alone |
| `bare` | 4,547 | the value alone, kept at 10% because it is genuinely ambiguous |

Validation and test are **never** composed: they are the real rows, unchanged.

## Protocol

Identical to the recorded cross-institution evaluation, so the comparison is
paired:

- the universities in `data/processed/field/fold_plan.json` are dealt into
  three disjoint folds; a fold's schools contribute nothing to that fold's
  training, validation or heading pool;
- each fold trains its own model and is tested only on its held-out schools;
- 447 + 480 + 368 = **1,295 real syllabus sections** from 14 universities;
- the early-stopping epoch is chosen on a slice of the *training* universities,
  never on the held-out ones;
- the reference checkpoint is scored on exactly the same rows with the same
  vocabulary (no rows skipped by either side).

`scripts/eval_field_transfer.py` reproduces any single comparison.

## Result: a tie, slightly below the reference on accuracy

| Fold | Held-out schools | Rows | Reference acc | Mix acc | Reference macro-F1 | Mix macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| 0 | Berkeley, Ohio State, MIT, Penn State, UCLA | 447 | **0.8300** | 0.8300 | **0.6483** | 0.6165 |
| 1 | Georgia Tech, UT Austin, Wisconsin–Madison, Stanford, UNC | 480 | **0.8917** | 0.8792 | **0.6952** | 0.6743 |
| 2 | Purdue, Illinois, Carnegie Mellon, Michigan | 368 | 0.8886 | **0.8913** | 0.6783 | **0.7482** |
| **pooled** | 14 universities | **1,295** | **0.8695** | 0.8657 | 0.6742 | **0.6753** |

- accuracy: **−0.38 points** (0.8657 vs 0.8695), inside the ±1.6-point binomial
  noise on 1,295 rows;
- macro-F1: **+0.11 points**;
- fold 2 is the only fold where the mix model wins, and it wins on both metrics.

In-distribution, the production model (`artifacts/field-classifier-mix`, trained
on every source, temperature 0.426) scores **0.9983 accuracy / 0.9709 macro-F1
on 7,571 held-out-group rows**. The validation split saturates at 0.999
accuracy in every run, which is the same ceiling the reference reports and the
reason early stopping is nearly arbitrary here.

### An apparent win that is not one

Run the heading-first cascade on the 446 labelled sections of 83 real syllabi
and the new context model looks spectacular:

| Cascade | fallback | accuracy | macro-F1 |
|---|---:|---:|---:|
| reference context model, threshold 0.95 (its best point is 0.8946 at 0.90) | 20.6% | 0.8812 | 0.7365 |
| new context model, threshold 0.95 | 20.6% | **0.9417** | **0.8517** |
| new context model, always use it | 100% | **0.9955** | **0.9516** |

That is memorisation, not transfer. Those 83 syllabi are in the training
corpus, so the model has read the very sections it is being scored on; the
reference model has read them too but is too small to memorise them, which is
the only reason its number looks worse. On the 1,295 sections from the 14
universities that are actually held out, the same model ties the reference
(0.8657 vs 0.8695). **The fold result is the honest number; the 446-section
result must not be quoted as model quality.** It is, however, a useful warning:
a benchmark drawn from the training corpus can make any model look fixed.

## Where the error actually lives (fold 0)

Splitting the 447 fold-0 test rows by the heading they were fed clears up the
whole picture. Three of the corpus's headings are placeholders, not real
headings: `document fact` (a published fact about the document), `section body`
(a paragraph lifted from a section), and an empty heading. Everything else is a
real heading the institution printed.

| Group | Rows | Reference acc | bge-small acc | ModernBERT acc |
|---|---:|---:|---:|---:|
| `document fact` | 163 | **1.0000** | **1.0000** | **1.0000** |
| real heading | 175 | 0.9429 | 0.9314 | 0.9371 |
| placeholder (`section body` / none) | 109 | 0.3945 | 0.4220 | 0.4312 |
| **all** | **447** | 0.8300 | 0.8322 | 0.8367 |

**Three quarters of the benchmark is essentially solved and one quarter is not
answerable as posed.** On a real heading the model is at 94.3% before any of
this work and 93.7% after it — the aggregate is flat because 109 rows (24%)
ask the model to name the field of a paragraph that was cut out of a section,
carrying no heading and often spanning several fields. That group is at 39–43%
for every encoder, including the 149 M one, and it is where all the apparent
encoder differences come from.

This is the single most useful result in the document: the production metric
should not average those rows with real sections, and a reader that has no
heading for a block should say so instead of assigning one of 42 labels with
97% confidence. The extraction path inherits the same limit — a paragraph
without a heading is not a field, it is context for one.

The remaining fold-0 errors are diagnostic, not random:

- 76 of 447 rows wrong. **Median confidence on a wrong row: 97.7%**; 50 of the
  76 are wrong above 90%. The model is not uncertain, it is confidently wrong —
  the failure the reference model card already documents on unseen universities.
- Top-3 accuracy on the same rows is **0.8859** and top-5 **0.9060**. The right
  field is usually in the shortlist; the ranking of the top two is what fails.
- The errors concentrate on labels whose *gold itself is ambiguous* in the
  source corpus. `Attendance: 75` and `Attendance: 50` are enrolled counts, but
  they are labelled `attendance_policy`; `Credit Hours: 3 Department of
  Political Science 2140 Derby Hall…` is a section body that happens to open
  with a credits label and is labelled `credits` while the body is about an
  address. No amount of training recovers a label the text does not carry.
- Only 1,491 of a fold's ~133,000 training rows are real labelled syllabus
  sections, and the fold vocabulary has 41 field types. **Twenty of them have
  fewer than 600 training rows and fifteen fewer than 200** (`drop_add_deadline`
  has 2, `course_modality` 4, `class_type` 19, `meeting_days` 20). That is the
  binding constraint: the model cannot learn a field's semantics from a handful
  of examples, however deep the encoder.

What would move the number, in order of expected effect:

1. **More real labelled syllabi from more universities.** The corpus has 83
   syllabi; the benchmark is 14 universities. Doubling the labelled syllabus
   coverage is worth more than any encoder or schedule change measured here.
2. **A soft target instead of one label per section.** Several bodies genuinely
   belong to two fields; training on the top-2 (multi-label) would stop forcing
   the model to pick arbitrarily between them.
3. **An abstention head trained on cross-institution folds.** The model's
   percentages are calibrated in-distribution (ECE 0.002) and badly
   overconfident out of it (ECE 0.15 on the fold-0 rows). A gate that says
   "this heading is unfamiliar" would convert confident-wrong labels into
   declines, which is what a reader can act on.

None of these is a longer run of the same data.

## Bigger and newer encoders, measured (fold 0)

The same corpus and split were also trained with two larger, newer encoders.
Fold 0 is the hardest fold and the one every encoder has been scored on.

| Encoder | Params | Fold-0 acc | Pooled acc (1,295) | Pooled macro-F1 |
|---|---:|---:|---:|---:|
| MiniLM-L6 (reference) | 22.7 M | 0.8300 | 0.8695 | 0.6742 |
| bge-small-en-v1.5 | 33.4 M | 0.8322 | 0.8657 | 0.6753 |
| ModernBERT-base | 149.0 M | **0.8389** | **0.8726** | **0.6917** |
| ModernBERT-large | 395.5 M | 0.8255 (1 epoch) | — | — |

ModernBERT-base was run through all three folds. Each fold's checkpoint is its
epoch-1 model — the same point in the schedule for all three, and the epoch the
early stop selected in folds 0 and 1 anyway (fold 2's run was stopped after epoch
1 so the three checkpoints remain comparable). Per fold:

| Fold | rows | reference acc | ModernBERT acc | reference macro-F1 | ModernBERT macro-F1 |
|---|---:|---:|---:|---:|---:|
| 0 | 447 | 0.8300 | **0.8389** | **0.6483** | 0.6277 |
| 1 | 480 | 0.8917 | **0.8938** | 0.6952 | **0.7293** |
| 2 | 368 | **0.8886** | 0.8859 | 0.6783 | **0.7204** |
| **pooled** | **1,295** | 0.8695 | **0.8726** | 0.6742 | **0.6917** |

What the table says, precisely:

- **Capacity buys in-distribution fit, not top-1 transfer.** One epoch of
  ModernBERT-base already reaches 0.993 validation macro-F1 against bge-small's
  0.980 *best*, and 0.9994 validation accuracy. Pooled over the three held-out
  folds that becomes **+0.31 points of accuracy** — a tie — and **+1.75 points of
  macro-F1**, which is the one direction that repeats (folds 1 and 2 improve the
  rare labels; fold 0 does not). The model is better at not dropping rare fields,
  not better at reading an unseen university.
- **ModernBERT-large is not a result.** Its learning-rate schedule spans 3 epochs
  and it was stopped after the first (30.7 minutes per epoch on this GPU, 11.2 GB
  peak at batch 16), so 0.8255/0.6552 is an undertrained checkpoint, not a
  verdict. At that cost a fair 3-epoch run is 1.5 h per fold; the trend from
  33 M to 149 M gives no reason to expect it to pay.
- Training exposed a real bug that would have silently handicapped every new
  encoder: layer-wise learning-rate decay only recognised BERT's
  `encoder.layer.N` naming, so ModernBERT's `layers.N` blocks all received one
  flat rate. `pace.field_train._layer_id` now handles both, with a test.

## On a phone: only the small encoder survives quantization

`scripts/export_field_onnx.py` exports a checkpoint to ONNX, quantizes it to
dynamic int8 and measures single-section CPU latency on 4 threads. Same held-out
rows, same tokenizer, same max length:

| | bge-small | ModernBERT-base |
|---|---:|---:|
| Parameters | 33.4 M | 149.0 M |
| int8 file | **33.9 MB** | 150.1 MB |
| int8 label agreement with fp32 | **100.0%** | **75.5%** |
| CPU latency p50 per section | **3.6 ms** | 50.4 ms |
| CPU latency p95 | **4.8 ms** | 104.7 ms |

This is the part that decides the deployment question. A 20-section syllabus at
3.6 ms is ~70 ms of model time on a desktop CPU, so roughly a quarter to half a
second on a phone core — comfortably interactive. ModernBERT-base is 14× slower
and, worse, **dynamic int8 changes one label in four**: its layer-norm-heavy
blocks do not quantize as cleanly as BERT-style blocks. A 150 MB model that
loses 24.5% of its predictions after the compression any phone build requires is
not a model you can ship without a per-layer quantization-aware training recipe
that has not been written here.

ModernBERT-base's pooled macro-F1 (+1.75 points) is real and worth keeping in
the record, but it does not survive the compression a phone build needs: 24.5%
of its labels change after dynamic int8, and 150 MB / 50 ms per section is 14×
the small model's cost. Recovering that would require quantization-aware
training of the encoder, which has not been done here.

Recommendation from the measurements: **ship the 33 M bge-small encoder**, keep
it int8 (100% agreement, 3.6 ms), and reject further parameter growth until
(a) quantization-aware training makes a bigger encoder survive int8, and
(b) there is labelled data that makes transfer the thing capacity can fix.

## What was added anyway

- `pace/field_mix.py` — the corpus builder above, with tests
  (`tests/test_field_mix.py`) that hold its guarantees: composed rows never
  enter validation or test, a held-out school contributes nothing to training,
  a cross row never borrows its own source's heading, and the per-label cap is
  respected.
- `scripts/eval_field_transfer.py` — scores any checkpoints on identical rows.
- `pace.field_train --save-each-epoch` — a multi-hour run now leaves the best
  model on disk after every improving epoch instead of only at the end.
- `pace extract-fields` — one command from arbitrary UTF-8 text to the record
  the extractor keeps (course, meetings, grading, policies), each value with its
  evidence span.
- `scripts/export_field_onnx.py` — ONNX fp32 + int8 export with label agreement
  and CPU latency, so a checkpoint's deployability is measured, not assumed.

## Reproduction

```bash
python -m pace.field_mix --fold 0 --extra-values data/processed/field_uncapped/corpus.jsonl
python -m pace.field_train \
  --train data/processed/field_mix/fold-0/train.jsonl \
  --validation data/processed/field_mix/fold-0/validation.jsonl \
  --test data/processed/field_mix/fold-0/test.jsonl \
  --output artifacts/field-mix/fold-0-final \
  --base-model BAAI/bge-small-en-v1.5 --epochs 14 --batch-size 96 \
  --max-length 160 --patience 4 --save-each-epoch
python scripts/eval_field_transfer.py \
  --rows data/processed/field_mix/fold-0/test.jsonl \
  --checkpoint artifacts/field-crossinstitution-context/fold-0 \
  --checkpoint artifacts/field-mix/fold-0-final \
  --name reference --name mix
```

Run folds 1 and 2 by changing the fold number throughout; the pooled row above
is the sum over the three test files.

## Rights

Same caveat as the reference checkpoint: six of the eight corpus sources
declare no reuse licence, so these checkpoints and the mixed corpus are local
research artifacts and must not be published or redistributed without
per-source clearance.
