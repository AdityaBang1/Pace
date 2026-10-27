# Pace

Pace is an iOS academic planning app that turns information students already have in their transcripts into a structured academic record they can actually use.

Instead of starting with manual data entry, Pace can import a transcript, reconstruct its academic structure, let the student review what was found, and then use that information throughout the app for GPA and credit tracking, course organization, future schedules, Pathways, requirements, and planning comparisons.

The app is local-first. Transcript processing, model inference, academic calculations, and storage are handled on-device. Pace does not require an account or backend server to use the core app.

## What Pace does

### Transcript import

Pace can import a PDF transcript and turn the document into structured academic data.

The import pipeline uses the PDF text layer when it is available and native document-reading support when it is not. Transcript rows are reconstructed from the document layout, including course titles, marks, credits, terms, and in-progress coursework.

Before anything is committed to the academic record, the user gets a review step. Imported values keep source information so they can be traced back to the document they came from.

### Academic record

Imported courses become part of Pace's normal academic data model rather than remaining a one-time document summary.

Pace tracks things such as:

- academic years and terms
- completed, current, and planned courses
- earned and in-progress credits
- reported and calculated GPA values
- course levels and expected grades
- standardized test scores

The Academics tab lets the user move between years, schedules, and terms while keeping the imported record connected to the source document.

### GPA and credits

Pace includes a configurable GPA system instead of assuming every school uses the same weighting rules.

The user can define grade bands and course weighting behavior, and Pace keeps transcript-reported GPA separate from GPA values it calculates itself.

Credit totals are tracked across completed, current, and planned coursework.

### Schedules

A school year can have more than one schedule. This lets a student keep their actual or main schedule while also creating alternatives without changing the original academic record.

Schedules can contain planned courses, expected grades, credits, and course levels that are used for projections.

### Pathways

A Pathway is a multi-year plan made by selecting a schedule for each academic year.

Students can keep multiple Pathways, make one active, duplicate them, and compare alternatives. Pace can compare Pathways using projected GPA, credits, requirements, and year-by-year course differences.

### Requirements and targets

Pace can track customizable credit requirements and user-defined targets.

The active academic record and Pathway can then be evaluated together so Pace can show whether the current plan reaches those goals or where something is still missing.

This is deterministic planning logic, not a prediction of student success.

## How transcript understanding works

Academic documents are difficult because the order of text inside a PDF does not always match the way the page looks to a person. A course name, grade, and credit may be stored as separate text runs or returned in an unexpected reading order.

Pace uses a hybrid approach:

1. The document is read locally.
2. Text and layout information are collected from the page.
3. Deterministic parsers reconstruct transcript rows and academic structure.
4. Compact local models are used for classification tasks where semantic interpretation is useful.
5. Validation logic checks and reconciles the result.
6. The user reviews the import before it is written to SQLite.

The deployed model artifacts are exported to ONNX and run through `onnxruntime-react-native` on the device.

Machine learning is intentionally not responsible for every part of the pipeline. Layout reconstruction, row assembly, GPA calculations, credit totals, schedule logic, Pathway comparison, and other rules are handled with deterministic code where that is more reliable.

## Privacy and local processing

The core Pace app does not require a server or account.

- transcript parsing happens locally
- ONNX inference happens locally
- academic data is stored in SQLite on the device
- GPA and planning calculations happen locally

This keeps the document-processing path independent of a remote AI service.

## Tech stack

- TypeScript
- React Native 0.88
- Expo SDK 58
- Expo Router
- `@expo/ui` SwiftUI components
- Expo SQLite
- ONNX Runtime for React Native
- native iOS PDF and Vision integration
- Python and PyTorch for model development and evaluation

## Project structure

```text
app/                 Expo Router screens
src/database/        SQLite schema and repositories
src/domain/          GPA, record, Pathway, requirement, and Pace logic
src/engine/          document and transcript parsing logic
src/ui/              shared native UI and theme code
modules/             native iOS document-reading integration
assets/models/       mobile model assets
scripts/             model export, evaluation, and development utilities
tests/mobile/        TypeScript and SQLite test suites
artifacts/            research and evaluation outputs
```

The runtime application and the research/evaluation material live in the same repository, but the research artifacts are not required to use the app.

## Running the app

Pace uses native modules, so use an Expo development build rather than Expo Go.

```bash
npm install
npm run prebuild
npm run ios
```

You can also start the development server directly:

```bash
npm start
```

## Tests

```bash
npm run typecheck
npm run test:engine
npm run test:app
npm run test:transcript
npm run test:db
npm run test:domain
```

The test suites cover the mobile document engine, transcript logic, application behavior, SQLite commits, and academic domain logic.

## Current scope

Pace is still under active development. Transcript formats vary substantially between schools, so support is based on the document structures that have been implemented and tested rather than a claim that every transcript will parse correctly.

The current architecture is designed so additional formats can be added without changing the student's academic data model or planning features.

## Why Pace

Most students already have the information needed to begin academic planning. It is usually sitting in a transcript.

Pace is built around a simple idea: that information should be the starting point, not something the student has to type in again before they can begin planning.