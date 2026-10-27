# Pace model evaluation report

Pace uses a hybrid document-understanding pipeline. Compact machine-learning models help classify academic text where semantics matter, while deterministic code handles document structure, transcript row reconstruction, validation, calculations, and database writes.

This report summarizes the evaluation findings that shaped the current product. It intentionally focuses on model behavior and system design rather than acting as a source-by-source experiment notebook.

## Scope

The model's job is narrow: classify pieces of academic text into typed fields that the rest of the application can reason about. It does not generate academic facts, calculate GPA, choose courses, or write directly to the student's record.

Transcript parsing is even more constrained. Pace primarily treats a transcript as a layout-reconstruction problem: find the printed course rows, preserve their term context, join wrapped titles, attach grades and credits to the correct course, and distinguish completed work from work still in progress.

## Evaluation approach

A random held-out split is useful for catching regressions but is not enough to measure whether the model will transfer to a new school. Pace therefore uses harder evaluation protocols where practical:

- hold out institutions or source families from training,
- compare familiar-source and unseen-source performance separately,
- measure accuracy and macro-F1 rather than accuracy alone,
- inspect confidence calibration on shifted data,
- test heading-present and heading-removed inputs,
- simulate OCR and spacing corruption,
- test long-document and layout edge cases,
- run parity checks between research code and the mobile implementation.

## Main findings

### Familiar data can make a model look better than it is

Early classifiers produced very strong results when the held-out examples resembled the training distribution. The same models could perform much worse on documents from schools or document families that were entirely absent from training.

That became one of the most important evaluation lessons in Pace. Product decisions are based on transfer tests and real document behavior rather than one headline score.

### Shortcut features mattered

Some early training examples included headings next to values in a way that made the target class unusually easy to infer. Measuring the same models with the heading removed exposed how much of the score came from that shortcut.

Training with header dropout improved robustness when headings were absent or renamed. The current evaluation process therefore tries to match the information that will actually be available to the model at inference time.

### Document structure carries meaning

Academic values are often ambiguous in isolation. A number may represent credits, a score, a room, or something else depending on the surrounding row or heading. Short text can have the same problem.

The surrounding structure therefore matters as much as the text itself. Pace preserves headings, rows, columns, terms, and evidence spans where possible instead of flattening every document into independent strings.

### Transcript parsing works better deterministically

PDFs frequently return text in an order that differs from the page a person sees. Course titles may be split across runs, grades and credits may be returned separately, and two visual columns may interleave.

Pace's transcript parser reconstructs that structure with deterministic rules. It groups printed rows, joins wrapped titles, carries term context forward, separates current from completed work, and checks totals where the document provides them.

This proved more reliable than trying to make a classifier infer the entire table.

### The best design is hybrid

The resulting pipeline is intentionally divided by responsibility:

1. **Document reading** — use the PDF text layer when available and local OCR when it is not.
2. **Layout reconstruction** — preserve or rebuild rows, sections, and reading order.
3. **Semantic classification** — use a compact local model for fields that benefit from learned semantics.
4. **Validation** — deterministic checks reject or flag contradictory and incomplete results.
5. **Review** — imported values remain inspectable before they are committed.
6. **Academic logic** — GPA, credits, requirements, schedules, Pathways, and Pace status are normal deterministic application logic.

## Robustness findings

OCR-style corruption reduces model quality, especially when spaces or line structure disappear. Long documents also expose failure modes that are not visible in short snippets. These cases are treated separately from ordinary classification accuracy.

Confidence can also become over-optimistic under distribution shift. A high score on familiar data does not guarantee the same reliability on an unfamiliar school. Pace therefore keeps review and provenance as part of the product instead of treating the classifier's confidence as a guarantee.

## Mobile deployment

The compact classifier is exported to ONNX and runs through `onnxruntime-react-native`. The goal is useful semantic classification without sending academic documents to a remote inference service.

The application keeps the ML surface deliberately small. The model returns typed decisions, not generated prose. The deterministic parts of the pipeline remain independently testable.

## App-side verification

The repository includes separate checks for model/mobile parity, transcript parsing, application import logic, SQLite commit behavior, and academic-domain calculations.

```bash
npm run typecheck
npm run test:engine
npm run test:app
npm run test:transcript
npm run test:db
npm run test:domain
```

## Limitations

Pace should not be interpreted as a universal document parser. Transcript and academic-document formats vary significantly between schools. A layout that has not been tested should be treated as new input, not assumed support.

The current system is designed around that limitation: preserve source evidence, use deterministic checks where possible, surface uncertainty, and make imported information reviewable.

For the product architecture, see [`docs/APP.md`](docs/APP.md). For the shorter research summary, see [`docs/research.md`](docs/research.md).