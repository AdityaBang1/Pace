# The Pace app

Pace is an iOS application built on **Expo SDK 58** (React Native 0.88, React 19.3). It turns a photograph of an
academic document into structured academic data, entirely on the phone, with the
reviewer approving every value before anything is saved.

This document describes what is implemented, how the model is wired in, what has
been *proven* on this machine, and what has **not** been run yet (device tests).

---

## 1. The shape of the product

```
photograph ──► Apple Vision OCR ──► PaceEngine ──► review ──► SQLite
                    (text+boxes)      (local ONNX)    (user)     (transaction)
```

Externally there are three interactions: **Scan → Review → Import**. Everything
else — tokenisation, the document gate, section classification, deterministic
value readers, evidence spans, confidence, transactions — sits underneath that
line and is never exposed as model internals.

---

## 2. Architecture

```
app/                         Expo Router screens (UI only)
  _layout.tsx                providers: gesture root, safe area, SQLite, import draft
  index.tsx                  Home: scan CTA, recent imports, courses
  scan.tsx                   camera / photo / text file + real progress stages
  review.tsx                 the signature screen: proposals, edits, source, import
  course/[id].tsx            imported course, grading bar, meetings, policies, undo
  academics.tsx              all courses
  settings.tsx               privacy, reader details, delete-all, diagnostics link
  debug.tsx                  development diagnostics (__DEV__ only)

src/engine/                  the reader (TypeScript, shared with the Node harness)
  PaceEngine.ts              the one application-facing entry point
  ir.ts                      AcademicRecord construction, stable ids, gate
  rules/                     deterministic readers (grading, meetings, credits, …)
  tokenizer.ts, sectionModel.ts, documentType.ts, matcher.ts
  native/                    phone adapters: model files, ONNX Runtime, manifest

src/features/
  scan/pipeline.ts           OCR → engine, stage callbacks, image persistence
  scan/geometry.ts           OCR line boxes → evidence spans
  import/draft.ts            AcademicRecord → reviewable proposals
  import/commit.ts           proposals → database rows
  import/ImportContext.tsx   the draft, the edit operations, the transaction

src/database/                schema + repositories (expo-sqlite)
src/ui/                      theme, primitives, formatters
modules/pace-vision/         Swift Expo module: Vision OCR + PDFKit, with line geometry
assets/models/               the frozen reader bundle and manifest
tests/mobile/                Node parity harness + app logic checks
```

The UI imports **only** `AcademicRecord` (from `src/engine/types.ts`) and the
draft types. It never imports a tokenizer, a logit, or a checkpoint name.

---

## 3. The frozen reader

`assets/models/manifest.model` is the single source of truth:

| part | artifact | notes |
|---|---|---|
| section reader | `section-model.onnx` | MiniLM, 22.7 M params, 192 tokens, fp32 |
| context reader | `context-model.onnx` | heading+value fallback, loaded **lazily** |
| document gate | `doc-type.model` + `doc-type.bin` | char TF-IDF + logistic, flattened |
| course matcher | `matcher.model` | 14 coefficients, threshold 0.95 |
| headings / policy cues | `rules.model` | 254 heading patterns |
| tokenizer | `tokenizer/vocab.model` | WordPiece, lower-cased, accents stripped |

**Why fp32 and not int8.** Both variants were exported and measured on the same
probes. `section-model-int8.onnx` agrees with fp32 on only 87.5% of rows (max
logit difference 3.63) while the fp32 export agrees on 100% (max 4e-06). A reader
that changes its answer for one section in eight is not a reader, so the bundle
ships fp32: 2 × 90 MB of graphs, plus 6 MB of tables. The int8 artifacts remain on
disk for the day a quantization recipe keeps parity.

**Why the second graph loads lazily.** The context model only runs for headings
the primary model is unsure about (~7% of sections on real syllabi). Opening both
graphs at launch doubled start-up work for a path most documents never take.
`PaceEngine.create` now builds the second session on the first unfamiliar heading.

`engine.understand(text, { assumeDocumentType })` lets the reviewer re-read the
same text as a different type. It re-opens the gate, records in `warnings` that a
person made that call, and never runs on a document type the reader cannot
extract from (the UI only offers `syllabus` and `course_catalog`).

---

## 4. OCR boundary (photographs and PDFs)

`modules/pace-vision` is a local Expo module (Swift, `expo-module.config.json`,
autolinked from `modules/`). It returns a document of pages:

```ts
{ kind: 'pdf' | 'image', pageCount, textPages, ocrPages, truncated, durationMs, text,
  pages: [{ page, method: 'text_layer' | 'ocr', text, imageWidth, imageHeight,
            durationMs,
            blocks: [{ id, text, page, x, y, width, height, confidence }] }] }
```

**Each page is read by the method that is exact for it.** `recognizeDocument`
notices a PDF (by extension or `%PDF-` header) and opens it with PDFKit, then takes
one of two paths per page:

* **The page's own text layer**, when it has a usable one — at least 24 characters,
  at least two lines unless it is a short page, and no replacement characters or
  `(cid:…)` escapes. Those are the author's bytes: a course code, a mark and a
  weight cannot be misread. Line boxes come from PDFKit selection bounds, so a
  value still names its page and line. This is the path a syllabus exported from a
  word processor or a course catalog takes, and it is the path the research
  pipeline's runs were measured on (`pypdf` layout mode in `pace/ingest.py`).
* **Recognition of rendered pixels**, when the layer is missing, empty, or has no
  line structure — a scan, a page photographed into a PDF, a file whose text is
  drawn word by word with no newlines. The page is drawn at a 2000-pixel long side
  (capped at 2600) and read by the same Vision request a photograph gets.

Every page reports which method produced it, and the pipeline warns on the record
when a PDF page had no text layer, so a reading of pixels is never presented as
the document. There is no page cap hidden in the UI: documents longer than 25
pages are read up to the cap and the record carries a warning saying so.

**Why there is no third-party PDF package here.** No Expo-maintained module
converts PDF text or pages, and the community options each solve half the problem:
`react-native-pdf-thumbnail` renders page JPEGs (which then still need OCR, and
carry no text), while `expo-pdf-text-extract` returns text with no geometry, so
provenance would lose its highlight in a PDF. pdf.js needs a worker and a canvas
that Hermes does not provide. PDFKit is already on the device, so the existing
Swift module gained ~80 lines instead of a second native dependency.

**Why not `expo-ocr-kit` (0.1.4).** It is a real module, not a stub: one
`recognizeText(uri)` returning `{ text, blocks: [{ text, boundingBox }] }` from
Vision on iOS and ML Kit on Android, MIT. It was evaluated on 2026-10-25 and not
adopted, for three reasons:

1. **It cannot read a PDF.** It takes an image URI only, so PDFKit stays in this
   repository either way; adopting it would mean two OCR paths (the package for
   photographs, Vision for rendered pages) with different ordering rules.
2. **Its iOS path rotates ordinary phone photos twice.**
   `CGImageSourceCreateThumbnailAtIndex` is called with
   `kCGImageSourceCreateThumbnailWithTransform: true`, which bakes EXIF
   orientation into the thumbnail, and the same EXIF orientation is then passed to
   `VNImageRequestHandler(cgImage:orientation:)`, which applies it again. A
   portrait camera JPEG (orientation 6) would reach Vision sideways. This module
   reads through `UIImage`, whose `.cgImage` is unrotated, so orientation is
   applied exactly once.
3. **It drops what provenance uses.** No page index, no per-line confidence, and
   no `truncated` flag for long documents; the reader would have to reconstruct
   all three above the boundary.

What it does have is Android coverage (ML Kit). If Android becomes a target, the
cheapest route is to use it for `Platform.OS === 'android'` images only, where its
EXIF handling lives in Kotlin and PDF import stays iOS-only — not to replace this
module on iOS.

**Multi-page text and offsets.** The reader sees one string. `joinPages()` joins
pages with a blank line and records each page's start offset; `lineGeometry()`
rebuilds absolute line offsets and pairs them with Vision's boxes. A value on page
3 therefore resolves to page 3, and the source sheet shows `Page 3`. A PDF's page
image is not retained: its source sheet quotes the line and names the page rather
than drawing a highlight over a page the app cannot show.

* Coordinates are **normalised 0…1 with a top-left origin**. Vision returns
  bottom-left, so the module flips `y` once (`1 - maxY`) rather than leaving that
  trap to every consumer.
* One block per recognised line, sorted into reading order (descending `midY`,
  then `minX`), joined with `\n` — the same text the engine receives.
* `VNRecognizeTextRequest` uses `.accurate` with language correction, the device's
  languages intersected with the ones Vision ships for the current revision
  (English always present, capped at four), and a small academic vocabulary as
  `customWords` so "prerequisite" and "credit" are not corrected into other
  words. Digits matter more than speed here.
* A page whose long side is under 2000 pixels is lifted to 2000 before recognition
  (at most 4×, high interpolation). A screenshot or a cropped capture is smaller
  than a camera photograph, and the measured ladder read `86` for `98` at a
  768-pixel long side but every cell correctly at 1266.
* The module is required **optionally** (`requireOptionalNativeModule`), so a web
  or Expo Go run produces a readable error at scan time instead of a crash at
  import time.

`src/features/scan/geometry.ts` re-derives line offsets from that joined text and
maps each evidence span to a box. If the block count and line count disagree, it
returns nothing — a missing highlight is better than one pointing at the wrong
line.

---

## 5. Review, import, provenance

`buildDraft()` turns an `AcademicRecord` into `Proposal<T>` values (value, band,
evidence id, source line, offsets, included/edited flags). `draftToCommitInput()`
converts only the included proposals into rows. Both are pure functions and are
covered by `tests/mobile/app_logic.ts`.

The import is **one SQLite transaction**: document, import record, course (or a
link to an existing one), grading categories, meetings, policies and every
evidence row. "Undo import" deletes the course and its rows, or — when the import
was linked to an existing course — only the rows whose evidence belongs to that
document, then marks the import `undone`.

Identity is stable and content-derived: `document.id` is a hash of the text, and
evidence/entity/relation ids are derived from the document id plus the span. Re-
reading the same page produces the same ids, which is what lets accept/reject
state, edits and cross-document links survive a re-scan.

Provenance in the source sheet: the document's own line (quoted, with the value
marked), the page index, Vision's confidence, and — when geometry exists — the
photograph with the line highlighted. The overlay is projected onto the drawn
image rect, so it lands correctly for portrait and landscape pages.

---

## 5b. Native controls

Where a native control is genuinely better than a styled `Pressable`, the app
uses **`@expo/ui` 57** (SwiftUI): the document-type menu in Review and the
storage toggle in Settings. Both are written the way the package requires —
components from `@expo/ui/swift-ui`, modifiers from `@expo/ui/swift-ui/modifiers`
(`pickerStyle('menu')`, `tag(...)`), inside a `Host` whose React Native `style`
carries the size. `matchContents` is deliberately not used there: both sit inside
scroll views, and the docs warn against matching size on the same axis as a scroll
container. The `Host` heights are explicit for that reason.

The package is iOS/Apple-only, so each control has a platform twin
(`NativeToggle.tsx`, `NativeDocumentTypePicker.tsx`) that renders the same
behaviour with React Native primitives. Metro picks the right file, which is why
the Android bundle builds even though `@expo/ui/swift-ui` is never imported there.
Expo UI is stable as of SDK 56 and does not require a special build, but this app
needs a development build anyway for Vision and ONNX Runtime.

## 5c. Motion

Every animation in the app had to pass two gates before it was written: how often
the user meets it, and what it is for. Most of the app did not pass.

| surface | frequency | decision |
|---|---|---|
| Press feedback (buttons, chips, rows, the scan card) | tens/hour | **built** — 120 ms, `scale: 0.97`, feedback on press-in, label and background scale together |
| Home list arriving on focus | tens/hour | kept as a 160 ms fade, not a slide; removed entirely under reduced motion |
| Scan progress / review card appearing | occasional | 200 ms fade with a strong ease-out |
| The ✓ when a scan stage finishes | occasional | 160 ms fade — no scale-from-zero |
| Screen to screen | — | **not built**: the native stack's own push. All `animation` overrides were deleted, including the ones that duplicated the platform's modal slide |
| Tab switches, scrolling, toggles | 100+/day | nothing, by design |

Text links ("Settings", "See all", the close affordance) are the one pressable that
is dimmed rather than scaled: a 3% scale on a word reads as a wobble, not as
touch. They keep their opacity feedback and nothing else.

Two things make the press feedback cheap: there is no gesture and no shared value
(the press is a two-state style change), so React re-renders once per press and the
transition itself runs on Reanimated's UI thread; and it animates only `transform`,
which never re-runs Yoga. [src/ui/press.tsx](src/ui/press.tsx) is the one
implementation, used by the shared primitives and the two largest targets on Home.
Reduced motion pins the scale at 1 and spends the same 120 ms on an opacity dip
instead, so the press is still acknowledged.

Haptics follow meaning rather than component: a selection tick on chips (the one
segmented-control-shaped control), a light impact on a **primary** button only, a
medium impact when an undo or "delete all" commits, and the success/error pattern
when an import lands or fails. The back button, source rows, quiet buttons and text
links deliberately have none — a tick on every navigation is how an app gets its
haptics turned off.

Two setup faults were fixed on the way. `react-native-reanimated` was imported by
three screens but only present as a transitive dependency of `expo-router`; it and
`react-native-worklets` are now declared through `npx expo install`. And the project
carried a `babel.config.js` that added `react-native-worklets/plugin` manually on top
of `babel-preset-expo`, which already adds it — the duplicate transform is exactly
the misconfiguration that makes worklets fail at runtime. The file is gone and the
preset supplies it. `CADisableMinimumFrameDurationOnPhone` is set, so ProMotion
iPhones are not capped at 60 fps.

**Not judged:** feel. Press timing, the haptic at the moment an import lands, and
whether the fade on Home is invisible enough cannot be assessed from a bundle. They
need a release build on a physical iPhone — a dev build's JS thread is slow enough
to hide the problems these choices exist to avoid.

## 6. What is proven, and how

Run from the repository root:

```bash
npm run test:engine       # Python ↔ TypeScript parity: 135 checks
npm run test:app          # draft, gate, commit, ids, matching, page offsets: 49 checks
npm run test:db           # schema, commit, undo, links, delete-all on real SQLite
npm run typecheck         # tsc --noEmit over app, engine, module, tests
npx expo-doctor           # 20/20 checks
PYTHONPATH=. python -m pytest -q    # the model repository's own suite
```

**SDK 58.** The app runs on Expo **58.0.19** with React Native **0.88** and React
**19.3.0**; `@expo/ui` and `expo-router` are on their 58 lines. The upgrade was
mechanical because the app never used the APIs that changed: no legacy
`expo-file-system` imports, no `expo-router/react-navigation` imports, no
`initialRouteName`, no removed React Native APIs, and no `File.write` call until
the model-marker work below (it is awaited there, which is the SDK 58 signature).
Router's new default web output (`server`) is irrelevant to a native build, but
`web.output` is left unset deliberately rather than pinned to `static`.

The upgrade also needed a clean dependency tree: `expo install --fix` rewrote
`package.json` for 58, but `expo-router@58` has an optional peer on
`expo-symbols@^58`, and npm refused to reconcile that against the SDK 57 tree
still on disk. Removing `node_modules` and the lockfile, adding `expo-symbols`,
`expo-font` and `@types/react@19.3` explicitly, then installing fresh produced a
tree where `expo-doctor` passes all **20/20** checks.

The parity harness (`tests/mobile/run.ts`) loads the same artifacts through
`onnxruntime-node`, runs the same fixtures, and compares document type, sections,
course fields, grading, meetings, policies, confidence bands, evidence text and
offsets, and warnings against the Python pipeline's frozen IR. Tokeniser parity is
checked on codes, percentages, punctuation, Unicode, abbreviations and
OCR-corrupted strings before the graph is compared, because a tokeniser
disagreement would otherwise look like a model disagreement.

`npx expo export` bundles the app for **both iOS and Android** (Hermes bytecode
~4.5 MB) with all nine model artifacts (183 MB of assets, two fp32 graphs among
them), which is how the Metro asset wiring for `.onnx`, `.bin` and `.model` files
— and the platform split between the SwiftUI controls and their React Native
twins — is verified without a device.

### Reader behaviour on documents the model has never seen

`scripts/engine_probe.ts` runs the shipped reader over any text file and prints
what the review screen would show. Three documents written for this check (not
fixtures, not in any training split):

* **An unseen syllabus** (Riverside Preparatory, dotted grade table): classified
  `syllabus` 0.767, gate open, 5 sections, course title/code/instructor read,
  grading `Participation 15% · Close Reading Essays 30% · Comparative Paper 25% ·
  Final Portfolio 30%`, total 100, meeting `MO WE 13:15–14:30 · 214`, office
  hours found, 9/9 evidence rows grounded, 294 ms end to end on desktop.
* **A restaurant menu** and **a bank statement**: `other` 0.97 and 0.89, gate
  blocked, zero course/grading/meeting/policy rows, import refused.
* **The same syllabus with OCR-style corruption** (`l`→`1`, `I`→`l`, `l5%`):
  classified `other` 0.59 and gated. With the reviewer override the reader
  recovers 3 of 4 categories, flags the 85% total, keeps every value grounded,
  and records in `warnings` that a person set the type. This is the degradation
  path the UI is designed around — not a claim of OCR robustness.

That last case is the honest picture of OCR tolerance: Apple Vision on a real page
is far better than hand-typed corruption, but a page whose small text confuses
characters can fall under the document gate, and the recovery is the reviewer's
explicit "Read as Syllabus".

A dot-leader table (`Participation ............ 15%`) previously produced **zero**
categories because the rule demanded a colon or dash between label and percentage.
Both implementations now accept dot leaders; `tests/test_grading_dot_leaders.py`
and the dot-leader case in `tests/mobile/app_logic.ts` pin it, including the trap
that "lose 10% per day" must not become a 10% category.

### Not yet verified

These need a Mac with Xcode and a real iPhone; **none of them has been run**:

* `npx expo prebuild --platform ios` and a development-build install;
* the Swift module compiling and returning boxes on device;
* PDFKit rendering a real multi-page PDF, and the page offsets holding on device;
* the text-layer rule firing on a real digital syllabus (it was checked against
  the two fixture PDFs' extracted text on the developer machine, not by PDFKit
  on a device), and its per-line selection bounds lining up with the text;
* `onnxruntime-react-native` loading a 90 MB fp32 graph from app storage;
* Vision OCR quality on a photographed page;
* on-device latency and memory for the full pipeline (25 pages of OCR is the slow case);
* `@expo/ui` rendering the native picker and toggle inside a scroll view;
* the acceptance walkthrough in the product spec (scan → review → import → course).

The honest status is: the engine, the port and the logic are measured; the native
shell is written but unexecuted.

### 6b. Hardening pass

Five failure modes were found by reading the paths a real user takes, and fixed:

| failure | what happened before | what happens now |
|---|---|---|
| Double-tapped **Import** | two transactions with the same course id; the second failed on the primary key *after* the first had saved | `commit()` is idempotent behind a ref (state lands a render too late) and returns the first import's course; the review button has its own ref guard |
| Double-tapped **Take photo** / **Choose file** | `busy` was state, so two scans (and two camera sheets) could start before the button disabled | the scan screen holds an `inFlight` ref and the whole task runs under it |
| App killed during the 90 MB model copy | `ensureModelFiles` skipped any file that *existed*, so a truncated graph was loaded on every later launch | the copy compares byte lengths with the bundled asset, rewrites a wrong-length file, and writes a `.ready.model` marker last; startup skips only when the marker and every length still agree |
| "Keep source" turned off | the picker's cache URI was written into `documents.source_uri`, pointing at a file iOS may purge | no URI is recorded at all; the quoted line stays, and the review screen simply has no thumbnail to promise |
| Source sheet after import | the course screen passed `width: 0, height: 0` to the highlight, which then drew nothing — even for a kept photograph; a PDF was also passed to `<Image>` | the sheet measures the file with `Image.getSize` when the row has no dimensions, and a `.pdf` URI is never given to `<Image>` |

A second pass audited the data layer with a new harness, `tests/mobile/db_commit.ts`,
which runs the real `schema.ts`, `repo.ts` and `commit.ts` against `node:sqlite`
(`expo-sqlite` is only type-imported, so nothing native is needed). It found a bug
that no amount of reading had caught:

* **Every import failed with `FOREIGN KEY constraint failed`.** `commitImport`
  inserted the course with `import_id = <this import>` *before* the `imports` row,
  and SQLite checks that reference immediately on a connection where the schema's
  `PRAGMA foreign_keys = ON` has taken effect. The order is now documents → import →
  course, and the harness asserts the import lands, the course exists, and undo
  leaves the document behind. `commitImport` also refuses a document whose gate was
  closed, so a future caller cannot bypass review.

Other defects fixed in the same pass, each with a check where one was possible:

| failure | what happened before | what happens now |
|---|---|---|
| Undo after a *linked* import | cleanup keyed on `evidence_id` missed every value the reviewer had typed (no evidence row), and the course kept rows the app said were undone | rows are deleted by the ids `commit.ts` derives from the import id, with the evidence query kept as a backstop, then the course's grading total is recomputed |
| "View source" on a linked import's rows | the course loaded evidence only for its own document, so a schedule's rows pointed at nothing | evidence is loaded for the course's document **and** every evidence id its rows reference |
| "Undo import" after undoing a link | the newest import row was `undone`, so the button disappeared and the creating import could never be removed | the course's own `import_id` is the target, with the old newest-committed query as a fallback |
| "Delete all data" | rows went, photographs and PDFs stayed on disk — a privacy claim the storage did not honour | files are collected before the rows go, then deleted, then `pages/` and `files/` are removed |
| Recent imports after undo | the joined course row is gone, so history said "Unnamed document" | the course name recorded in `imports.summary` is used as the fallback |
| Closing Review while "Importing…" | the awaited commit still navigated, dropping the user into a course they thought they had cancelled | header controls are inert while a commit is in flight |
| Two quick "Read as" picks | whichever `understand` resolved last won, so the picker could disagree with the draft | a sequence counter keeps only the newest reading; failures set `status: 'failed'` with a message instead of rejecting silently |
| Course title with two evidence spans | the first span by field was quoted, which could be a different line than the value came from | the entity's own `evidenceId` is preferred |
| A read error on Home, Academics or a course | rendered as "Nothing imported yet" / "No courses yet" / "This course is gone" | the previous content stays and a `Notice` with retry appears |

`expo-sqlite` is type-only in that path, so `npm run test:db` is a plain Node run and
needs no simulator: 28 checks over commit, cascades, undo, links, history names and
delete-all.

The photograph path also stopped decoding at full resolution. A 12–48 MP capture
is now read through a `CGImageSource` thumbnail capped at 2400 px, with EXIF
orientation applied exactly once by `VNImageRequestHandler`
(`CreateThumbnailWithTransform` stays off). That is both the memory bound and the
correctness point that ruled out `expo-ocr-kit` above. The page size the module
reports is the *oriented* size: Vision normalises its boxes against the upright
page, so a portrait capture (landscape pixels plus EXIF rotation) now yields
`imageWidth < imageHeight` and a highlight frame that matches the page the user
sees, instead of a landscape frame with the box thrown off.

---

## 7. Known limitations

| limitation | effect | workaround in the app |
|---|---|---|
| Doc-type gate abstains on some schedules (0.70 `other` on the schedule fixture) | a schedule page is read as non-academic and extraction is suppressed | reviewer can re-read as a type; schedule extraction itself is not implemented |
| The gate is sensitive to character corruption (an OCR-style syllabus at 0.59 `other`) | a hard-to-read page can be gated | review offers "Read as Syllabus"; a partial read is then shown with warnings |
| Dot leaders are read, but a category whose label is itself corrupted loses its row | a breakdown can total less than 100 | total is shown and flagged; nothing is normalised |
| The page cap is 25 | a PDF longer than that is read only up to the cap | the record warns that the document was truncated; nothing claims full coverage |
| A PDF has no retained page image | its source sheet quotes the line and names the page, without a highlight | photograph the page if the highlight matters |
| A PDF page with no text layer is recognised from pixels | its characters are a reading, not the author's bytes | the record warns which pages were read that way; review shows every value |
| fp32 graphs (185 MB bundle) | large download, slower cold start | second graph is lazy; int8 was rejected on measured parity |
| Course-title extraction can include trailing units ("MATH 1A. Calculus I. 4 units") | a title can need a small edit | title is editable in review, always shown as a proposal |
| Confidence is a rule confidence, or an uncalibrated model score for sections/pages | not a probability you can threshold on | values are shown as bands, everything requires confirmation |
| No handwriting, no rotated-page recovery, no multi-column layout repair | photographed prose only | — |

---

## 8. Privacy, as implemented

* OCR is Apple's on-device Vision framework.
* The reader is bundled in the app; inference runs through ONNX Runtime locally.
* The database is SQLite in the app's own container.
* There is no account, no analytics SDK, no network call in the product path.
* Imported page images and PDFs are copied into the app's private document
  directory; Settings → Storage turns that copy off, and Settings → Data deletes
  everything.
* PDF pages are rendered locally with PDFKit; no PDF network or script feature is
  used.

Settings states exactly these facts and nothing more.

---

## 9. Swapping the model

1. Export new artifacts into `assets/models/` (`npm run export:models` regenerates
   the current ones from the frozen checkpoints).
2. Update `assets/models/manifest.model` (ids, sizes, hashes, labels, temperature).
3. Run `npm run test:engine` and `npx tsx tests/mobile/app_logic.ts`; regenerate
   `fixtures/expected/*.json` only when a semantic change is intended.
4. Nothing in `app/`, `src/database/` or `src/ui/` changes.

---

## 10. AI-assisted development disclosure

The implementation in this repository was written with AI coding assistance
(Freebuff/Codebuff). The architecture it follows was decided against measurements
already in the repository:

* the frozen reader was chosen from the transfer/audit evidence, not novelty;
* the port was accepted only after 135 Python↔TypeScript parity checks;
* the gate behaviour on the schedule fixture was recorded as a limitation rather
  than papered over;
* int8 was rejected because its parity measurement was worse.

Anyone maintaining this should be able to explain, without the assistant: why the
reader is fp32, why the context graph is lazy, what a `Proposal` carries, where the
transaction boundary is, and what the parity harness compares. If any of those
answers is missing, the code should be read before it is extended.
