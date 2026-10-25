# jev-field-v4 — the reader that has to read values, not column headers

`artifacts/jev-field-v4` is the fourth run of the Jev-shaped field reader: 15,331,563
parameters, 16 K WordPiece vocabulary, a 127-token state window and a menu of 42
*described* fields. Architecturally it is identical to
[`jev-field-v2`](../jev-field-v2/MODEL_CARD.md); the difference is what it was
trained to read.

## Why this checkpoint exists

Every row in the corpus is written `"<source column>: <value>"`, and the column
usually names the field — `"Class Name: Literature and Popular Culture"`,
`"credits: 4.0"`. A model trained on that text alone can reach 0.998 accuracy by
reading the prefix and never looking at the value. Measured on the same 7,571
held-out rows, `jev-field-v2` scores:

| input | accuracy | macro-F1 |
|---|---:|---:|
| row as published (header included) | **0.9982** | **0.9711** |
| value only (`bare_text`) | 0.8991 | 0.7000 |

A 10-point accuracy and 27-point macro-F1 gap between two views of the *same*
rows is the signature of header reading, and it is exactly what fails on a new
institution: the columns are renamed, the values are not.

## What changed

`scripts/build_header_dropout.py` writes a training split in which half of the
rows that carry a header lose it (`header_probability` 0.5). Validation and test
are untouched, so the published-row number stays comparable with earlier runs.
Training is one epoch from `jev-field-v2` (lr 6e-4, batch 192, option dropout
0.15, sampler power 0.5, 316 s on an RTX 4070).

## Results

| benchmark | v2 | **v4** |
|---|---:|---:|
| field corpus, published rows (7,571) | 0.9982 / 0.9711 | **0.9983 / 0.9734** |
| field corpus, **bare values** | 0.8991 / 0.7000 | **0.9473 / 0.8172** |
| bare-value ECE (10-bin) | — | 0.0348 |
| syllabus **document-shift** (534 new syllabus docs) | 0.6480 / 0.4250 / ECE 0.293 | **0.8390 / 0.6585 / ECE 0.1325** |
| SIS / Infinite Campus holdout (354 cells) | 0.8470 / 0.4920 / 0.122 | **0.9153** / 0.4758 / 0.0651 |

Numbers are accuracy / macro-F1 / ECE where three are shown. The regression gate
(`scripts/regression_gate.py --incumbent artifacts/jev-field-v2 --candidate
artifacts/jev-field-v4`) accepts it: field macro-F1 +0.0022, unseen-syllabus
macro-F1 +0.1958, SIS macro-F1 −0.0158 (inside the −0.02 tolerance).

The document-shift result is the point. On 534 syllabus rows from documents it
never saw, v4 is **+21 accuracy points and +22 macro-F1 points** better than the
checkpoint it replaces, with the calibration error cut from 0.293 to 0.133. That
is what learning the value rather than the column buys.

## What it still does not do

- It is not a new institution's *schema*. A field nobody described stays
  unpredictable; the menu has to name it.
- Rare-field macro-F1 on SIS cells is 0.4758, slightly below v3's 0.5307 — for
  district-format cells, keep `artifacts/jev-field-v3`.
- Bare-value confidence is far from calibrated on prose (ECE 0.0348 on values,
  0.13 on shifted syllabus documents). Gate new-institution output, do not trust
  its confidence.
- Leave-one-source-out training is still the honest generalisation test; see
  [`../../artifacts/source-holdout/source_holdout_report.json`](../source-holdout/source_holdout_report.json).

## Artifacts

| file | what it is |
|---|---|
| `model.pt` | fp32 weights (61.4 MB) |
| `tokenizer.json` | the 16 K WordPiece vocabulary shared with v2 |
| `metrics.json` | the training run, including the test block above |
| `header_reliance.json` | published-rows vs bare-value scoring, per source |
| `quantization.json` | dynamic int8: 36.6 MB, 1.98 ms/row, identical decisions |
| `choice.log` | the training log |

## Reproduce

```bash
python scripts/build_header_dropout.py            # data/processed/field_mix_dropout
python -m pace.jev_train --stage choice --output artifacts/jev-field-v4 \
  --encoder artifacts/jev-field-v2 --tokenizer artifacts/jev-field-v2/tokenizer.json \
  --train data/processed/field_mix_dropout/train.jsonl \
  --validation data/processed/field_mix/validation.jsonl \
  --test data/processed/field_mix/test.jsonl \
  --epochs 1 --batch-size 192 --lr 6e-4 --option-dropout 0.15 --sampler-power 0.5 \
  --hidden 320 --layers 12 --heads 4 --ffn 1024
python scripts/measure_header_reliance.py --checkpoint artifacts/jev-field-v4
python scripts/regression_gate.py --incumbent artifacts/jev-field-v2 \
  --candidate artifacts/jev-field-v4
```
