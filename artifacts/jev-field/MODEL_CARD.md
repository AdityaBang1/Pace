# jev-small — model card

A 10.28 M-parameter decision model that reads a piece of academic text and picks
the described field it satisfies, out of 42 candidates. Built in the shape of
TypeSafe's Jev and the open TinyDecide reproduction; trained from scratch on this
project's corpus. Full protocol, tables and failure analysis:
[EXPERIMENT.md](EXPERIMENT.md).

## Summary

| | |
|---|---|
| Task | choose one described field for a line or section of a course document |
| Input | state text ≤ 127 tokens plus a fixed question |
| Options | 42 written descriptions (`pace.jev_model.FIELD_DESCRIPTIONS`), swappable |
| Parameters | 10,278,955 |
| Checkpoint | 41.2 MB fp32 (`model.pt` + `tokenizer.json` + `config.json`) |
| Device | CPU or CUDA today (RTX 4070 measured); no mobile export is committed |
| Trained | RTX 4070, 3,098 s total (2,054 s masked-LM pretraining + 1,044 s choice) |

## Intended use

- Labelling a heading/value pair, a table cell, or a short section from a syllabus,
  catalog page or course site as one of the 42 fields, with a calibrated
  probability.
- Feeding the field extractor (`pace.field_extract`) or the CLI
  (`pace.cli classify-field --checkpoint artifacts/jev-field`).
- Scoring text against a *re-phrased* version of those 42 descriptions, through
  `JevEngine.score_against`.

Not intended for: free-form document summarisation, fields outside the 42
descriptions, or deciding anything where a wrong label is costly without a
confidence threshold.

## Results

Same 7,571-row split, alongside the reference retrained on it:

| Model | Params | Accuracy | Macro-F1 | Weighted-F1 | ECE-10 | Errors |
|---|---:|---:|---:|---:|---:|---:|
| **jev-small** | **10.3 M** | **0.9982** | 0.9155 | 0.9982 | **0.0007** | 14 |
| bge-small-en-v1.5 | 33.4 M | 0.9982 | 0.9713 | 0.9982 | 0.0012 | 14 |
| all-MiniLM-L6-v2 (older checkpoint) | 22.7 M | 0.9464 | 0.8486 | 0.9457 | 0.0166 | — |

Universities the model never saw (3 folds, 1,295 rows): **0.8738 accuracy /
0.6982 macro-F1** against bge-small's 0.8668 / 0.6796, better on every fold, with
lower ECE (0.0886 vs 0.1133).

## Confidence

The engine ships a fitted temperature (0.579) and returns the full distribution.
ECE-10 is 0.0007 in-distribution and 0.066–0.128 on unseen universities; the
worst fold is the one holding out MIT/Berkeley/UCLA.

Measured on the 7,571 held-out rows: at p ≥ 0.9 the model accepts 99.80% of rows
at 99.907% precision. It is not error-free at high confidence — 7 of its 14
errors sit above 0.9, and the two most confident (0.999) are the same duplicated
`office_hours` line read as `instructor`. Of the 14 errors, **8 are
`section body` placeholder rows** — a paragraph cut out of a section, with a
label the text simply does not carry — the same failure this repository already
documents for the encoder reference. The other six are: one `course_description`
read as `course_title`, one `assignments` read as `grading_policy`, one
`additional_information` and one `attendance_policy` read as
`disability_services`, and the duplicated `office_hours` pair.

## Limitations

- **New fields are not zero-shot.** The score comes from the option's
  description, so re-wording a trained field works; inventing an unseen field
  without examples does not (measured: 1 of 3 invented options correct).
- **Coverage, not reasoning, is the usual failure.** Bare `MWF 9:00-9:50 AM` has
  no training analogue (the corpus writes `Days: MWF` and
  `meeting time: 8:25PM - 9:50PM`), so it is read as `course_code`.
- **Long tail.** `attendance_policy`, `disability_services`, `materials`,
  `office_hours` and `exam_schedule` have 1–43 training rows and 1–13 test rows;
  their per-class numbers move by 0.1–0.3 on single mistakes.
- **Truncation.** 127 state tokens; a long course description is cut, and the
  heading has to be part of the text for the model to see it.
- **Language.** English, lowercased WordPiece trained on this corpus; other
  languages degrade to subword pieces.

## Provenance and ethics

Training rows are public course catalogues and syllabi already vendored under
`data/` (licensing notes in `data/inventory.json` and `docs/research.md`).
Nothing is generated text. The model has no memorisation guard beyond its size
and is not audited for fairness across institutions; it is a labeler, not a
policy decision-maker.
