# Model card — jev-small v2 (academic field reader)

## What it is

A 15.3 M-parameter encoder that answers one question: *given a piece of academic
text and a menu of described fields, which description does this text satisfy?*
It returns the chosen field, a probability, and the ranked alternatives. It does
not generate facts and it does not write to a student record.

- Checkpoint: `artifacts/jev-field-v2/` (`model.pt` 61 MB fp32, `tokenizer.json`,
  `config.json`, `metrics.json`)
- Question: *Which academic field does this text belong to?*
- Options: 42 field descriptions in `pace.jev_model.FIELD_DESCRIPTIONS` — the
  option is text, so a field can be renamed or re-described without retraining
  the encoder.
- Served by `pace.jev_model.JevEngine` and
  `python -m pace.cli classify-field TEXT --checkpoint artifacts/jev-field-v2`.

## Intended use

Labelling one section, cell or line of a real academic document — a syllabus, a
course catalog page, a schedule — with the field it belongs to, so a reader can
decide what to extract, display or ignore. Offline, on a phone, one text at a
time or a batch.

Out of scope: OCR (a separate pipeline produces the text), date/room/time
*parsing* (classification only), grades or eligibility decisions, and any use
where the percentage is treated as a guarantee.

## Training data

| Stage | Data | Size |
|---|---|---|
| Vocabulary | WordPiece, 16,000 tokens, trained on the extended corpus | 16 K |
| Masked-LM pretraining | the extended corpus + the uncapped corpus | 1.5 M passages × 3 epochs = 4.5 M passages, 72 M tokens |
| Decision fine-tune | `data/processed/field_mix_ext/train.jsonl` | 200,985 rows, 42 fields |

The labelled corpus is real institutional text, not generated: USC course
catalog cells, Gothenburg syllabus fields, UW-Madison catalog and meetings, MIT
catalog, UMich course pages, a syllabus entity corpus, UF business syllabi,
INSEAD business syllabi, CU-DocBench page blocks, plus — new in v2 — Columbia's
public Directory of Classes (37 terms) and the Coursera catalog. New sources
contribute at most 2,500 rows per field, so no single registrar becomes most of
the corpus.

Rows whose real heading was lost (`section body: …`) are withheld from training
and kept in validation/test. The composition families (`natural`, `cross`,
`heading`, `short`, `bare`, `sentence`, `schedule`, `value_diversity`) are
described in `pace/field_mix.py`; every word in them comes from a real document,
only the heading/value pairing is new, and nothing composed enters validation or
test.

## Measured behaviour

7,571 held-out rows, identical to the rows the first run and the bge reference
were scored on:

| Metric | jev-small v2 | bge-small-en-v1.5 | jev-small v1 |
|---|---:|---:|---:|
| Accuracy | **0.9982** | **0.9982** | **0.9982** |
| Macro-F1 (38 present fields) | 0.9711 | **0.9713** | 0.9155 |
| Weighted-F1 | 0.9981 | 0.9981 | **0.9982** |
| ECE-10 | 0.0016 | **0.0012** | 0.0007 |
| Errors | **14** | **14** | **14** |
| Parameters | **15.3 M** | 33.4 M | 10.3 M |

A university the model has never seen (fold 0: MIT, Penn State, Ohio State,
Berkeley, UCLA; 447 rows):

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| jev v1 | **0.8322** | **0.6391** |
| bge-small (same fold data) | 0.8300 | 0.6165 |
| jev v2 (this checkpoint), 9 epochs | 0.7830 | 0.5659 |
| jev v2, plain CE, 10 epochs | 0.7629 | 0.4805 |

**The v2 model is the better in-distribution reader and the worse transfer
reader.** The gap is about data diversity rather than the sampler: plain
cross-entropy at the same size is worse on the fold, and the first run's report
already found that a 149 M encoder transferred no better than a 33 M one.

Sanity checks on text the first version failed:

| Text | Label | p |
|---|---|---:|
| `MWF 9:00-9:50 AM` | meeting_time | 0.9997 |
| `TR 9:30-10:45 a.m., Journalism 214` | meeting_time | 0.9997 |
| `Days: MWF` | meeting_days | 0.9982 |
| `Attendance: … four absences … 1% of the final grade` | attendance_policy | 0.9788 |
| `Students with disabilities should contact … Accessibility Services.` | disability_services | 0.9931 |
| `Prerequisite(s): MATH 151 or equivalent …` | prerequisites | 0.9984 |

## Known limitations

- **Heading-less paragraphs stay hard.** If the heading was dropped upstream the
  model cannot always recover it; 14 test rows are `section body:` paragraphs.
- **Transfer is the weak axis.** For a deployment aimed at unfamiliar institutions,
  the first run's 10.3 M checkpoint is still the better choice (0.8322 vs 0.7830
  on fold 0) even though it is 5.6 macro points behind in-distribution.
- **Very thin fields.** `attendance_policy` has 8 clean training rows and one of
  its two test rows is heading-less; `credits` loses a bare `3` read as
  `instructor`.
- **New fields are not zero-shot.** The menu is swappable for *re-described*
  trained fields (there is a test for that); a field with no examples at all is
  not learned by description alone.
- **No K-12 data yet.** `pace.sis_import` reads Infinite Campus / OneRoster
  exports, but no real district export is public, so the training corpus contains
  no K-12 rows. The adapter is tested on the format only.

## Rights and provenance

Every source is public and none of the derived corpus or checkpoints may be
published or shipped until per-source clearance exists: several sources declare
no reuse licence at all, the Columbia catalog repository declares none either,
and the Coursera catalog is Apache-2.0. Public downloadability is not permission.
Strip `instructor` and `contact_information` before any release. This is a local
research artifact.

## Reproduce

See [`EXPERIMENT.md`](EXPERIMENT.md#reproduction). The pretraining checkpoint is
kept in `artifacts/jev-field-v2-pretrain/` so a fine-tune can start from the
language model without rerunning the 87-minute pretraining.
