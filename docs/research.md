# Pace model research and evaluation

Pace uses machine learning only where semantic classification is useful. Document structure, transcript row reconstruction, validation, GPA and credit arithmetic, academic planning, and database writes are handled by deterministic code.

The model is not treated as a source of academic facts. Its outputs are interpreted alongside the source text and the app keeps evidence for imported values so they can be reviewed before they are committed.

## Evaluation strategy

The main goal of evaluation is transfer, not memorizing familiar document formats.

Where possible, experiments hold out entire institutions, source families, or document styles from training. The evaluation tracks accuracy, macro-F1, and calibration separately and also tests the effect of OCR and layout corruption. Results on familiar data are not treated as evidence that the same model will generalize to a new school.

The app also has deterministic tests for transcript parsing, document-to-mobile parity, database commit and undo behavior, and academic-domain calculations.

## What the experiments showed

### Familiar-source results can overstate generalization

Early classifiers performed extremely well when the evaluation data looked like the data seen during training. Performance dropped substantially when the same approach was tested against documents from institutions or document families that had been withheld.

That changed the design of Pace. Cross-source and cross-institution tests became more important than a single random train/test split, and low-confidence or ambiguous classifications are treated as review candidates instead of trusted automatically.

### Headers can become shortcut features

Some early examples paired a field value with a heading that effectively revealed the correct class. A model could therefore score very highly while relying too heavily on the heading.

Training with header dropout made the classifier more robust when headings are missing, renamed, or formatted differently. The broader lesson is that evaluation needs to test the information that will actually be available at inference time.

### Structure matters more than loose values

Academic text is often ambiguous without context. A value such as a number, room, time, or short phrase may represent different fields depending on where it appears in the document.

For structured academic documents, the surrounding heading, row, column, and section are often more useful than the isolated value. Pace therefore preserves document structure where possible instead of flattening the entire document into unrelated snippets.

### Transcript parsing is primarily a layout problem

A transcript can look like a clean table while its PDF text layer returns course names, grades, credits, and wrapped lines as separate text runs or in a different reading order.

Pace handles this with deterministic reconstruction. It groups text by the printed layout, rebuilds course rows, joins wrapped titles, attaches terms, distinguishes completed and in-progress work, and checks values such as credits. Machine learning is not asked to infer the entire transcript table.

### A hybrid pipeline works better

The current design combines:

1. local document reading and OCR when necessary,
2. deterministic layout reconstruction,
3. compact semantic classification where it adds value,
4. deterministic validation and academic calculations,
5. user review before imported values are written to the record.

This was more reliable than trying to make one model solve document type, layout, extraction, validation, and academic logic at once.

## Local deployment

The compact classifier is exported to ONNX and runs in the app through ONNX Runtime React Native. PDF text extraction, OCR, transcript parsing, validation, and SQLite storage also run locally.

The model produces typed decisions rather than generated prose. GPA, credits, requirements, schedules, Pathways, and Pace status are calculated by normal application code.

## Robustness work

Evaluation has included:

- held-out institution and source-family tests,
- heading-present versus heading-removed inputs,
- OCR-like spacing and text corruption,
- long-document behavior,
- confidence calibration,
- parity checks between the research implementation and the mobile implementation.

The main remaining limitation is document diversity. Academic documents vary significantly between schools, so the app should not claim universal transcript or syllabus support. New layouts should be tested independently before being treated as supported.

## Reproducibility

The repository keeps the relevant training and evaluation code under `scripts/`, `pace/`, and `experiments/`, with model metadata under `artifacts/` where applicable.

Useful app-side checks include:

```bash
npm run typecheck
npm run test:engine
npm run test:app
npm run test:transcript
npm run test:db
npm run test:domain
```

The product architecture is documented separately in [`APP.md`](APP.md). This file intentionally focuses on the findings that influenced Pace's design rather than serving as a source-by-source research notebook.