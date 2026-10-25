# PaceDecide

A reproducible, **local-first academic document understanding** research prototype. It classifies document/section/relation snippets and surfaces source-grounded candidates; it does not generate academic facts or write to a student record. Every model output is a proposal with cited source text and user confirmation required.

## The app

The model now ships inside **Pace**, an Expo SDK 58 iOS application in this repository. Photograph a syllabus or import a PDF — a PDF page is read from its own text layer when it has one, and Apple Vision reads the pages that do not (PDFKit renders them locally) — and the frozen reader runs on the phone through ONNX Runtime: document gate, section classification, deterministic value readers, evidence spans. An academic transcript is read by its own deterministic reader ([src/engine/transcript.ts](src/engine/transcript.ts)) instead: one course per printed row, grouped by the term banner above it, with the printed mark and credit, and the syllabus readers do not run. Every value is reviewed and edited before one SQLite transaction writes it, and every imported value can be traced back to the line — and, for a photograph, the highlighted region of the page — it came from. The review and settings screens use native SwiftUI controls through `@expo/ui`. There is no server, no account, and no upload.

```bash
npm install
npm start          # Expo dev server (development build: the OCR module is native)
npm run test:engine            # Python ↔ TypeScript parity (135 checks)
npm run test:app               # draft, gate, commit and page-offset logic (49 checks)
npm run test:db                # schema, commit, undo, links on real SQLite (28 checks)
npm run typecheck
```

Architecture, the frozen-artifact table, the OCR contract, the import transaction, what the parity harness proves, and the full list of checks that **still require a device** are in [docs/APP.md](docs/APP.md). The short version: the engine and its port are measured; the native shell is written and bundles, but has not yet been run on an iPhone.

## Status and honest limitations

This is a research prototype, not a production-ready academic document parser. The headline work is the **real-text academic field classifier**: a 22.7M-parameter MiniLM that labels a piece of academic text with one of 42 field types (`credits`, `office_hours`, `exam_schedule`, `attendance_policy`, …) and reports a calibrated percentage, trained on **163,797 real rows** from eight public datasets — course catalogs, class schedules and syllabi, no synthetic documents.

The number that matters is not the in-distribution 0.989 accuracy, it is the cross-institution one: hold out whole universities and accuracy falls to **0.57–0.70** with ECE 0.21–0.33, because a model trained on loose values learns each institution’s *layout* rather than the field. Reading the document's **heading** instead is what transfers: on real labelled sections the heading alone is 88.1% accurate versus 69.3% for the body, and on the 1,326 rows a bare-value model and a heading-aware model share, the heading-aware model is **+28.2 accuracy points and +32.1 macro-F1 points better on universities it has never seen** (0.873 / 0.685 versus 0.590 / 0.364). That cascade — classify the heading, fall back to heading+value when the heading is unfamiliar — is the deployment recipe the measurements support. See [REPORT.md](REPORT.md) and the [model card](artifacts/field-classifier-minilm/MODEL_CARD.md).

The current best reader is the **Jev-shaped decision model**, `artifacts/jev-field-v2`: 15.3 M parameters, trained from scratch on this project's corpus, that makes the **same 14 errors in 7,571 held-out rows** as a 33.4 M bge-small trained on the same split (0.9982 accuracy each, macro-F1 0.9711 vs 0.9713, ECE 0.0016 vs 0.0012) at 46% of the parameters — and it runs on a phone. It is also **+5.6 macro-F1 points over the first run** (0.9155), mostly from removing training rows whose real heading had been lost.

The one number that did not improve is the harder one: on a university it has never seen (fold 0), v2 scores **0.7830** against the first run's 0.8322. The report rules out the sampler as the cause (plain cross-entropy is worse still, 0.7629) and states the rest as open rather than explained. Everything — the data expansion (Columbia's 37-term directory of classes, the Coursera catalog) and its per-field cap, the day-and-time fix, and what still confuses it — is in [artifacts/jev-field-v2/EXPERIMENT.md](artifacts/jev-field-v2/EXPERIMENT.md).

**Then the grouped school-holdout benchmark was built, and it settled a question.** `scripts/build_school_folds.py` groups the 14 real syllabus schools into three folds, removes each fold's schools from training across every family, and pools the three test sets into 1,638 rows in which every school is unseen exactly once. A model trained with those schools removed scores **0.491 accuracy / 0.388 macro-F1** — statistically the same as the 0.400 v2 reaches on the fold that removes the entire syllabus family. **Knowing the school buys almost nothing.** The gap is document-family transfer: the encoder is trained on 200 k catalog cells and only ~1.4 k syllabus rows. On that benchmark v3 scores 0.390 against v2's 0.400, so v3 is only the district specialist. A harder benchmark is now on disk as well: `scripts/build_source_holdout.py` removes one entire *publisher* at a time, and the result separates two different abilities — catalogue cells transfer to a publisher never seen (MIT 0.958, USC 0.966, Michigan 0.976 accuracy) while syllabus prose does not (Gothenburg 0.689, the 14-school syllabus corpus 0.464, on labels that training did see). The gap is document form, not vocabulary.

**Then a measurement error surfaced, and fixing it produced the current best reader.** Every training row is written `"<column>: <value>"`, and the column usually names the field — so a model can score 0.998 while reading the prefix and not the value. Scored twice on the same 7,571 rows, `jev-field-v2` is **0.9982 / 0.9711 with the header and 0.8991 / 0.7000 with only the value**: a tenth of its accuracy and a quarter of its macro-F1 was header reading, which is exactly what vanishes when an institution renames its columns. `scripts/build_header_dropout.py` hides the header on half the training rows, and one epoch from v2 gives [`artifacts/jev-field-v4`](artifacts/jev-field-v4/MODEL_CARD.md): **0.9983 / 0.9734 on published rows, 0.9473 / 0.8172 on bare values**, and on 534 syllabus documents from documents it never saw it moves from **0.648 / 0.425 / ECE 0.293 to 0.839 / 0.659 / ECE 0.133** — +21 accuracy and +22 macro-F1 points, with calibration error cut in half. The regression gate accepts the swap (field macro-F1 +0.002, unseen-syllabus macro-F1 +0.196, SIS macro-F1 −0.016); v3 stays the district-format specialist.

What the district pass did add is format coverage. `artifacts/jev-field-v3` — a 70/30 average of v2 and a run fine-tuned on 13 k **Infinite Campus / OneRoster** cells built by `scripts/build_sis_exports.py` out of real catalog values and read back through `pace.sis_import` — scores **0.895 accuracy / 0.531 macro-F1 / ECE 0.066** on a 354-row Infinite Campus holdout against v2's 0.847 / 0.492 / 0.122. Ship it for SIS-format cells, not as a general replacement. A control arm is worth knowing too: one more epoch on the same corpus *loses* 14 accuracy points on new documents, so the gains come from new data, never from longer training. Shifted confidence stays over-optimistic (ECE 0.32–0.42 on unseen prose) for every checkpoint tried, so new-institution output must be gated, not trusted. Full table, failed arms, family-level error breakdown and the taxonomy finding are in [artifacts/transfer-eval/TRANSFER.md](artifacts/transfer-eval/TRANSFER.md).

The earlier first task — catalog-cell field classification from USC’s 5,571 Spring 2024 rows — is still the cleanest, narrowest result, and the synthetic multitask experiment is a pipeline smoke test only, not product evidence. CU-DocBench has 23 documents / 100 selected pages and gold page-block structure, useful for later evaluation, but only four syllabi and original file rights need review. Full inventory and rights notes are in [data inventory](data/inventory.json) and [research](docs/research.md).

Real USC metrics are now available, but measure classification of existing catalog CSV cells—not PDF/image OCR or syllabus-document understanding. The USC card declares Apache-2.0; because it is a contributor-scraped catalog, verify the uploader's authority before commercial redistribution. The six newer corpus sources declare **no reuse licence at all** (one declares `other`): public downloadability is not permission, so the checkpoints and derived corpus are local research artifacts only — do not publish, ship or commercialise them without per-source clearance, and strip `instructor` and `contact_information` before any release. This Windows machine has an RTX 4070 and CUDA PyTorch. Training remains normal PyTorch; the deployed reader is an ONNX export that runs on the phone (see [docs/APP.md](docs/APP.md)). Phone latency and the iOS build itself are still unmeasured — the desktop numbers are the only ones claimed.

A smaller synthetic benchmark from the earlier pass is retained as a pipeline smoke test only, not as product evidence. The initial real-data errors exposed an underdetermined `TBA` value shared by both Days and Time columns; the corrected pipeline excludes it from classifier training and abstains on it at inference. See the [real experiment report](REPORT.md).

A small, task-separated, reproducible evaluation is available with `python -m pace.evaluate_datasets`. It verifies the local CU-DocBench file against its pinned SHA-256 before scoring bundled VLM block/layout predictions against gold labels. CC-BY-4.0 synthetic PerQA is used only for an offline structured date/time/event-field smoke test from a small test sample whose exact bytes are SHA-256 pinned ([sample](data/samples/perqa_test_sample.json)); it is **not** natural-language syllabus extraction or scanner calibration. The task metrics remain separate, no full PerQA corpus is downloaded, and neither dataset changes schedule inference confidence. Real syllabus schedule calibration remains blocked pending independent human-labeled, licensed, task-matched data; see the [evaluation protocol and provenance](docs/research.md#small-multi-dataset-evaluation-tasks-are-deliberately-separate).

The plan's remaining measurements are in [artifacts/transfer-eval/AUDIT.md](artifacts/transfer-eval/AUDIT.md): the **context-length sweep is answered** (127 vs 256 vs 384 vs 512 on unseen schools — no monotone gain, so the small window stays), **OCR robustness** costs −4.2 accuracy points when word spaces are lost and −6.0 combined, **long documents** expose two real defects (a 397 k-character syllabus yields *zero* sections when line structure is lost, and heading/body agreement on segmented sections is only 22%), and **deployment** is measured at 14.7 ms/item fp32 on desktop CPU with INT8 blocked by a broken ONNX graph for this encoder (traced attention reshapes bake the example's batch and sequence length).

## First real task

A shared PyTorch MiniLM encoder (22,716,296 total model parameters including classification head) predicts one of eight catalog cell labels: title, description, credits, prerequisites, restrictions, meeting time, meeting days, and class type. It is trained on natural cell text as provided; no arbitrary document splitting. Held-out test groups are entire unseen course-subject prefixes. Identical text values assigned multiple source labels are excluded (currently `TBA` in Days and Time) and are explicit inference abstentions. A sparse character n-gram logistic regression baseline and DeBERTa-v3-xsmall comparison use the same split. Results and limitations are in [REPORT.md](REPORT.md) and `artifacts/usc-real/comparison.json`.

The real USC task is not full document OCR, entity extraction, or syllabus understanding. A separate **synthetic-only PyTorch multitask experiment** (30,000 generated documents, held-out school identities and template families) now exercises document-type classification plus 19 entity span types. On its synthetic held-out set (1,208 documents, 20,382 gold spans), exact span F1 is 0.99909 on fully corrupted text and 0.99968 on clean text; document classification is 1.000 for both PyTorch and an identical-data TF-IDF baseline, reflecting easy generator cues. This is not real-document evidence. See the [synthetic model card](artifacts/synthetic-multitask/MODEL_CARD.md) and [metrics](artifacts/synthetic-multitask/metrics.json). Optional local inference: `python -m pace.cli analyze FILE --checkpoint artifacts/synthetic-multitask/model.pt --synthetic-pytorch`. It emits uncalibrated scores/proposals only. No Core ML/mobile work is in scope here.

### How small is the model, really?

Measured, not asserted. Three variants of the same encoder were trained on the same corpus, seed and hold-out, differing only in how many transformer layers they keep ([depth comparison](artifacts/synthetic-multitask/depth_comparison.json)):

| Encoder layers | Parameters | Checkpoint | Corrupted span F1 | Transcript span F1 |
|---:|---:|---:|---:|---:|
| 6 (default) | 22,732,081 | 90,958,791 B | 0.99909 | 0.99905 |
| 4 | 19,183,153 | 76,754,848 B | 0.99890 | 0.99878 |
| 2 | 15,634,225 | 62,550,848 B | 0.99674 | 0.99742 |

The 2-layer variant is ~31% smaller and costs ~0.24 F1 points; the 4-layer variant saves 15.6% of the checkpoint for ~0.02 F1 points. The default 6-layer checkpoint is 86.75 MiB in FP32 and 55.95 MiB dynamically quantized to INT8. In absolute terms this is a *small* transformer — MiniLM-class, not a frontier model — but "small" is not the same as "cheap to run": the whole document still has to be read through a sliding window, and the honest limit is that every number above is synthetic-only.

### Image-derived transcript run

`python -m pace.evaluate_transcript data/samples/mock_transcript_lepid_rowwise.txt --checkpoint artifacts/synthetic-multitask --output artifacts/synthetic-multitask/transcript_report_rowwise.json` runs the promoted checkpoint on the transcript text read from the supplied image. Result: document type `transcript` (p = 0.9929), 220/220 model entity spans matched (precision 1.000, recall 1.000, F1 1.000), 38 course records across 8 course subject areas (11 sections in the document, the other 3 being assessment sections), 3 assessment records (AK HSGQE reading/writing/math = 337 / 273 / 258), computed total credits 13.25 matching the reported 13.25 exactly, 0 records missing a title or section, 0 unattached performance levels. GPA is absent from the document and is reported as `null` rather than guessed. Full report: [transcript_report_rowwise.json](artifacts/synthetic-multitask/transcript_report_rowwise.json). This is one document, not a benchmark.

### Term-block transcript (numeric marks, wrapped titles)

A second real layout is transcribed as text in [transcript_term_marks.txt](data/samples/transcript_term_marks.txt): a `2025-2026 Grade 09   Term 2` banner over `Course / Mark / Weight / Credit` rows, percentage marks with no date column, four course titles that wrap onto the next printed line, and `Credit: 7.500   GPA: 4.13333` printed under the last row. `python -m pace.cli parse-transcript data/samples/transcript_term_marks.txt --summary` reads all eight rows: the local course code is split off with its own span (`220725y` / `Accel Adv Alg:Concept&Connect`), a wrapped title is joined into one value while each printed line keeps its own evidence span, weight and credit are read per row from the header's named columns, computed credits `7.5` match the printed `7.500` (`credit_sum_matches_reported_total: true`), GPA `4.13333` comes from the label, and the term is attached to every row and never mistaken for the school name. No model runs on this path — rows are recognised by shape and every value is a literal slice of the source.

### Course-block grade report (one row per course, one row per task)

A third real layout is transcribed as text in [grade_report.txt](data/samples/grade_report.txt): an attendance table (`Course / Absent / Tardy`) above a `Grade Report:` block whose header is `Course / Task / S1`, one row per course carrying its section (`10145y-101`), title, instructor in brackets, period and grade, and one row per graded task printed under it. There is no date, no credit and no weight anywhere on the page, so neither reader above can anchor a row. `python -m pace.cli parse-transcript data/samples/grade_report.txt --summary` reads all seven courses and all 25 task rows: the grade column is reported by the name the page prints (`grade_columns: ["S1"]`), each task keeps the course it was printed under, the last course's header wraps onto the next printed line (`[de la Rosa,` / `Jacquelyn Gabriellj 9 WEEK 97`) and is completed from it rather than becoming a course of its own, the one task whose label the scan lost keeps its mark with `title: null` instead of a guessed label, the six attendance rows are read rather than dropped, and no credit is invented: `credits_printed: false` with `computed_total_credits: null` says the column is absent, not that it is zero.

`python scripts/build_grade_report_page.py` draws that fixture as [grade_report.pdf](data/samples/grade_report.pdf) and [grade_report.png](data/samples/grade_report.png), so the same page can be read from pixels: `python -m pace.cli parse-transcript data/samples/grade_report.png` returns the same seven codes, sections and grades, with the two cells the scan misread (`68` for `89`, and a period label it dropped) repaired from their own crops or left null - `verified_fields: ["course_code", "mark"]`, because this layout prints no credit or weight cell to verify.

### Reading a PDF or a screenshot, not only pasted text

`pace parse-transcript` now takes a `.pdf` or an image directly. A PDF page is read from its own text layer in layout mode, which keeps the printed columns; a page whose text layer holds less than 24 characters (a scan, a stamp, a photo) is rendered at 2.5x and read by OCR, and so is any image. `--ocr auto` is that rule, `--ocr always` reads every page from pixels, and `--ocr never` refuses rather than returning something empty.

Every page says how it was read, and every row inherits it:

```json
"source": {"kind": "image", "page_count": 1, "methods": {"ocr": 1},
           "exact_source_text": false,
           "pages": [{"page": 1, "method": "ocr", "exact_source_text": false,
                      "detail": {"backend": "rapidocr", "boxes": 50, "upscale": 1.263}}],
           "note": "Values from an OCR page are a reading of rendered pixels, not the author's characters: check them against the page image before use."}
```

A text layer is the author's characters, so its rows carry `evidence_exact: true` and their spans can be sliced back out of the page. An OCR page is a *reading*, so its rows carry `evidence_method: "ocr"` and `evidence_exact: false`: the values are still deterministic slices of the extracted text and the parser is unchanged, but that text is not the document. This is the difference the JSON keeps, and it is why the same eight rows can be reported as exact from a PDF and as requiring review from its PNG.

OCR is optional and small: `pip install -e '.[ocr]'` installs [RapidOCR](https://github.com/RapidAI/RapidOCR) (ONNX detector + recogniser, 13.7 MB of models measured here, no service and no GPU), or put the `tesseract` executable on PATH. With neither installed, an image raises a message naming both fixes instead of returning an empty page; a PDF that has a text layer keeps working. Images smaller than a 2000-pixel long side are lifted to one before recognition (at the 2x render the recogniser read `86` for `98`; at 1.26x every cell read correctly).

**No number is kept because the page pass said so.** Every code, mark, weight and credit parsed from an OCR page is re-read from its own crop — the box that produced the characters, padded and doubled, recognised twice (as-is and contrast-stretched), all crops on a page in one batched call. The two readings are then checked against what the page already knows:

- a mark is a whole number in `0..100`, a weight and a credit are positive and at most 10, a local code is three to six digits then a letter;
- a cell whose printed width fits more digits than were read has lost characters (the 600-pixel rung read a two-digit mark as `4`), so it is refused;
- row credits must add up to the printed total, or no credit on the page is trusted — the page does not say which row is wrong.

The outcome per cell is `confirmed`, `repaired` (the focused read disagreed, was stable across both renderings, scored above the recogniser's 0.5 acceptance floor, and passes the rules) or `unread`. An unread cell is emitted as `null` with `reason`, the page reading, every focused reading and its evidence — never a guessed digit. Repairs are listed too: `"repaired": {"mark": {"page_reading": "06", "focused_reading": "90", ...}}`. Course codes are confirmed or marked `unconfirmed`, never rewritten from a crop: `b`/`h` in a two-letter suffix is weak evidence against a code that already matches the printed pattern. Titles are not re-read — a title is long, wraps, and a wrong one is not a wrong number — and cell checking needs a backend that reports box geometry: RapidOCR does (about 0.1 s of warm verification for 32 cells on the control page, one batched call per page; the ladder records the per-page figures under `on_device.verify_ms` in [photo_ladder.json](artifacts/ingest/photo_ladder.json)), while the Tesseract CLI path reports `verification.status: "unavailable"` instead of pretending.

Every cell is one record. `cell_checks` carries the emitted `value` beside the `status`, and both are written by the same function from the same decision, so a `confirmed` status above a `null` cannot be produced. That matters when the page's arithmetic refuses the credits *after* the focused pass: each refused credit goes through that same write path and becomes an `unread` cell — status, null and reason together — rather than a null bolted under `confirmed`. The tallies are then re-derived from the settled records instead of patched: `cells_checked` **is** `confirmed + repaired + unconfirmed + unread`, at page and document level, so the page tally and the document tally cannot disagree, and `unread` can never exceed the cells that were checked. The 768-pixel sensor rung, where the printed total refuses seven credits, reports `28 = 20 + 1 + 0 + 7` on its page and on the document, with seven `unread` credit cells and one repaired mark.

The four checked fields are listed in `verification.verified_fields`. Everything else the record emits — titles, term text, the footer GPA and printed total credits, assessment values — is marked `unverified` **on the record that carries it**, with the reason (`record["unverified"]["title"]`), and counted in `verification.unverified_values`. A document-level list of field names is not enough: someone who takes one title out of the JSON has no way to know it was never checked, so the mark travels with the value. The compact `--summary` view does the same for the two numbers it flattens: `reported_total_credits` and `gpa` are listed under `summary.unverified` on either path. A text file or a PDF text layer gets the same marks as a photograph — `verification.status` stays `not_needed`, because no crop stage ran on the author's characters, and `verified_fields` is empty to say so; `not_needed` means no check happened, not that every value passed one.

How small can it get? [scripts/measure_ingest_ladder.py](scripts/measure_ingest_ladder.py) downscales the control page rung by rung, runs the real path and compares every value to the fixture; the result is [resolution_ladder.json](artifacts/ingest/resolution_ladder.json).

| Page width | Rows | Wrong values | Unread cells | Repairs |
|---:|---:|---:|---:|---:|
| 1224, 1100, 1024, 900 | 8 | 0 | 0 | 0 |
| 800 | 7 | 0 | 7 | 3 |
| 700 | 6 | 0 | 6 | 0 |
| 600 | 6 | 0 | 8 | 1 |
| 512 | 8 | 0 | 0 | 1 |
| 448 and below | 0 | 0 | 0 | 0 |
| **256x202 capture** | **8** | **0** | **0** | **2** |

Zero wrong values at every rung: when the page is too small the answer degrades to `null` with a reason, and a row that could not be read at all is named in `source.warnings` instead of vanishing. The contiguous exact range is 1224 → 900 pixels wide; 512 also happened to read exactly, but it sits below a lossy band, so it is not a guarantee. Below 448 the recogniser no longer produces a row the parser can read, and the record says so. The user's actual [256x202 capture](data/samples/transcript_term_marks_256x202.png) does read all eight rows exactly, because its text is proportionally larger than the downscaled control's: the page pass misread `06` and `86`, and the focused re-reads repaired them to `90` and `98`.

Verified end to end: `python scripts/build_transcript_pdf.py` renders the fixture to `data/samples/transcript_term_marks.pdf` and `.png`, and `parse-transcript` on either returns the same eight rows as the text fixture — same codes, titles, marks, weights and credits, `Credit: 7.500` closed by the row sum and GPA `4.13333` from the label — with `text_layer` and `ocr` provenance respectively. `python -m pace.cli ingest FILE --text` shows the extracted text and how each page was read. Every command that takes a document (`parse-transcript`, `ingest`, `extract`, `scan-schedule`, `analyze`, `classify-field --file`) shares this reader, and `-` reads stdin, so `pbpaste | pace extract -` works too.

### What a hand-held camera adds

A clean render is not what a phone delivers. [scripts/measure_photo_ladder.py](scripts/measure_photo_ladder.py) places the control page on a plain background and applies the degradations a hand-held capture applies — a few degrees of rotation, keystone perspective (corners moved inside the frame, never cropping the print), an illumination gradient, a dark exposure, defocus, JPEG compression, a smaller sensor — alone and in combination, then reads every rung through the real OCR path and compares it to the eight fixture rows. Results: [photo_ladder.json](artifacts/ingest/photo_ladder.json).

| Condition | Rows | Repaired | Refused | Missing | Wrong |
|---|---:|---:|---:|---:|---:|
| clean control | 8 | 0 | 0 | 0 | 0 |
| rotate 2° / 4° / 6° | 8 / 8 / 8 | 1 / 1 / 0 | 0 | 0 | 0 |
| keystone 5% / 9% | 8 / 8 | 1 / 0 | 0 | 0 | 0 |
| gradient 0.55, exposure 0.45, exposure 0.30 | 8 / 8 / 8 | 0 | 0 | 0 | 0 |
| blur 0.8 / 1.5 px | 8 / 8 | 0 | 0 | 0 | 0 |
| motion blur 5 px | 4 | 0 | 4 | 16 | 0 |
| JPEG q45 / q30 | 5 / 7 | 0 | 2 / 7 | 12 / 4 | 0 |
| sensor 768 px | 7 | 1 | 7 | 4 | 0 |
| phone combo, 1024 px | 4 | 1 | 4 | 16 | 0 |
| phone combo, 768 px | 5 | 2 | 5 | 12 | 0 |
| phone combo, 512 px | 0 | 0 | 0 | 32 | 0 |
| the user's 256×202 capture | 8 | 2 | 0 | 0 | 0 |

**Zero wrong values at every rung.** Where the reader cannot recover a cell it says so: the value is `null` with the page reading, the focused readings and a reason, or the whole row is missing (counted above). The guarantee covers `course_code`, `mark`, `weight` and `credit` — and it counts the user's capture as one of its measurements, not only the eighteen numbered rungs (`guarantee.measurements: 19`, `includes_user_capture: true`), because a guarantee that sums a subset is a claim about a subset. Every value outside those four fields is listed under `guarantee.unverified` with its count and reason, per rung and in total (`checked: 0`), so no title or GPA in the artifact can be read as checked. Each rung also publishes its per-page tallies under `pages`, so `cells_checked = confirmed + repaired + unconfirmed + unread` can be read page by page from the artifact, not only from the document total.

Rotation was the one condition that cost every row, and it was a gap in the reader rather than in the recogniser: at 2° the detector merged the header with the first course row and the table parser saw no columns. [pace/deskew.py](pace/deskew.py) now estimates the page's tilt from a projection profile of its ink — rotating a 500-pixel copy of the mask through coarse and fine angles, keeping the angle whose row sums are sharpest — and turns the page back before recognition. The same angle is replayed when a verification crop is taken, so cell boxes still land on the pixels the recogniser read. It costs about 37 ms on the control page and refuses to move a page whose aligned profile is not clearly sharper than its unrotated one (`MIN_GAIN`), which keeps a straight page straight. Covered by [tests/test_deskew.py](tests/test_deskew.py) and an end-to-end 4° photo test in [tests/test_ingest.py](tests/test_ingest.py).

A phone also records its turn in EXIF rather than baking it into the pixels. [pace/orientation.py](pace/orientation.py) resolves all eight orientation tags — including the four mirrored ones — in one place, and both the OCR read and the verification-crop replay call it, so the page the recogniser saw and the pixels a cell crop is taken from cannot disagree. A side-tagged JPEG that previously read as zero rows now returns the same eight rows and 32 checked cells, every one confirmed, as the upright file; [tests/test_orientation.py](tests/test_orientation.py) pins every tag on both paths. iPhone's default HEIC/HEIF cannot be decoded by the shipped Pillow/RapidOCR stack, so it is named and refused with the export fix instead of falling through to a silent zero-row text parse. AVIF is read when Pillow has its decoder, and any other ISO-BMFF brand is named rather than parsed as text.

Perspective correction and binarisation were measured and are **not** shipped: keystone 5% and 9% already read all eight rows exactly without a warp, and global/adaptive thresholding destroyed the table structure on every degraded input (0–5 rows) including the user's clean capture (1–5 instead of 8). Unsharp masking helped the blurred, downscaled pages but cost rows on JPEG-only captures (5 → 1) and on the real 256×202 capture (8 → 6), so it is not applied. What is left is motion smear, blur + JPEG + a small sensor at once: characters are still read, but cells drop out of the detected table, and the row is either partial (refused credits) or absent. A 5-pixel smear recovers 4 of 8 rows; the combined capture is beyond the reader at 512 px; at 768–1024 px it recovers 4–5 of 8 rows with no wrong values.

**On-device cost of the shipped OCR path**: the three ONNX models are **13,709,164 bytes** on disk (detector 2.43 MB, recogniser 10.69 MB, orientation classifier 0.59 MB; the ONNX runtime and Python are additional). Reading a page cold takes about **2.6 s** on this CPU including model loading; warm, the whole path — deskew estimation, upscaling, detection, recognition and parsing — takes **1.4–1.5 s** for the clean control and the 768-pixel phone capture, of which the focused cell re-reads and parsing are about **80–120 ms**; deskew alone is **37 ms**. Those are this machine's CPU numbers, not phone measurements; phone latency for the ONNX path has not been measured.

## Repository layout

The phone app and the research behind it are one repository, and everything needed to build either is in it — including the frozen model bundle the app ships with. `assets/models/` is 233 MB and is committed as ordinary files rather than through Git LFS, so a plain `git clone` is all it takes: no LFS install, no separate download step.

| path | contents |
|---|---|
| `app/`, `src/` | the Expo Router screens, the reader and its native adapters |
| `modules/pace-vision/` | the Swift module: Apple Vision OCR, PDFKit page rendering, line geometry |
| `assets/models/` | the shipped bundle: `section-model.onnx` and `context-model.onnx` (fp32, 86 MB each), their int8 exports, `doc-type.model` + `doc-type.bin`, `matcher.model`, `rules.model`, `manifest.model`, `tokenizer/vocab.model` |
| `pace/`, `scripts/` | the Python research code and the runs that produce the numbers in this file |
| `tests/`, `tests/mobile/` | pytest suites, the Node parity harness and the app-logic and database checks |
| `artifacts/` | model cards, metrics and experiment notes, committed as evidence |
| `data/samples/` | the small document samples the tests and the demo read |
| `docs/` | the app's design notes and the research plan |

Two kinds of file are deliberately **not** in the repository, because they are either regenerable or too large for Git:

- `data/raw/`, `data/processed/`, `data/synthetic/` — downloaded corpora and generated splits, rebuilt by the scripts in the Reproduce section below. Only `data/samples/` ships.
- `artifacts/**/*.pt` and other checkpoint blobs — the 13 GB of trained weights. The model cards, metrics and logs next to them are committed, and `npm run export:models` rebuilds `assets/models/` from a checkpoint. Tests that need a missing checkpoint skip themselves rather than fail.

### Building on a Mac

```bash
git clone https://github.com/AdityaBang1/Pace.git && cd Pace
npm install
npm run test:engine && npm run test:app && npm run test:db   # no simulator needed
npm run typecheck
npx expo prebuild --platform ios     # generates ios/ and runs pod install
npx expo run:ios --device            # dev build on a connected iPhone
```

The camera and PDF reading live in the Swift module, so the app needs a development build — Expo Go cannot load it. The 233 MB bundle is copied into the app's document directory on first launch, which is why the first launch takes noticeably longer than later ones.

## Reproduce (Python 3.10+)

Install the core and training/test dependencies into an isolated environment. For an NVIDIA GPU, first install a CUDA-enabled PyTorch wheel appropriate to your driver from [PyTorch's install selector](https://pytorch.org/get-started/locally/); `pyproject.toml` does not force a 2.9 GB CUDA wheel onto CPU users.

```bash
python -m pip install -e '.[train-transformer,test]'
# Optional, CUDA wheel if needed; choose the current wheel for your platform/driver:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install pyarrow
python -m pytest -q
python -m pace.inspect_datasets
python -m pace.real_baseline --input data/raw/usc_course_data.csv
python -m pace.real_train --input data/raw/usc_course_data.csv --epochs 4 --batch-size 64 --output artifacts/usc-miniLM --split-assignment artifacts/usc-real/split_assignment.json
python -m pace.real_train --input data/raw/usc_course_data.csv --epochs 4 --batch-size 64 --output artifacts/usc-deberta-xsmall --base-model microsoft/deberta-v3-xsmall --split-assignment artifacts/usc-real/split_assignment.json
python -m pace.compare_real
python -m pace.real_infer "Prerequisite: BIO 101"
python -m pace.real_benchmark --runs 500
# Synthetic-only PyTorch multi-task experiment (not real-document evidence):
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask --count 30000 --seed 20261019 --epochs 8 --batch-size 64 --max-length 160 --corruption-rate 0.02
python -m pace.cli analyze document.txt --checkpoint artifacts/synthetic-multitask/model.pt --synthetic-pytorch
# Encoder-depth study (same corpus/seed/hold-out, only the layer count changes):
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask-l4-v8 --encoder-layers 4 --count 30000 --seed 20261019 --epochs 8
python -m pace.synthetic_doc_model --output artifacts/synthetic-multitask-l2-v8 --encoder-layers 2 --count 30000 --seed 20261019 --epochs 8
python -m pace.compare_depths
# Image-derived transcript run:
python -m pace.evaluate_transcript data/samples/mock_transcript_lepid_rowwise.txt --checkpoint artifacts/synthetic-multitask
# Term-block transcript with numeric marks, weights and wrapped titles:
python -m pace.cli parse-transcript data/samples/transcript_term_marks.txt --summary
# The same eight rows from a PDF and from a screenshot, with per-page provenance:
python scripts/build_transcript_pdf.py
python -m pace.cli parse-transcript data/samples/transcript_term_marks.pdf --summary
python -m pace.cli parse-transcript data/samples/transcript_term_marks.png --summary
python -m pace.cli ingest data/samples/transcript_term_marks.png --text
python -m pace.cli parse-transcript data/samples/transcript_term_marks_256x202.png --summary
# Where the table stops reading exactly (writes artifacts/ingest/resolution_ladder.json):
python scripts/measure_ingest_ladder.py
# Camera-like degradations and their cost (writes artifacts/ingest/photo_ladder.json):
python scripts/measure_photo_ladder.py
# Real-text academic field classifier (163,797 real rows, 8 datasets, 42 field types;
# capped at 6,000 rows per field type -- the cap is deliberate, see REPORT.md):
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
# Generalisation to universities the model has never seen (same held-out plan for both):
python -m pace.field_cross --output artifacts/field-crossinstitution --folds 3 \
  --fold-plan data/processed/field/fold_plan.json
python -m pace.field_pair
# A heavier transfer run: bigger encoders, cross-source headings, longer schedule.
# Top-1 ties the reference (0.8726 vs 0.8695 over 1,295 rows, 149M encoder) while
# macro-F1 gains +1.75; the 149M model loses one label in four to int8 and is 14x
# slower, so the 33M checkpoint stays the deployable one. See
# artifacts/field-mix/EXPERIMENT.md for the protocol, per-fold table and
# quantization numbers.
python -m pace.field_mix --fold 0 --extra-values data/processed/field_uncapped/corpus.jsonl
python -m pace.field_train \
  --train data/processed/field_mix/fold-0/train.jsonl \
  --validation data/processed/field_mix/fold-0/validation.jsonl \
  --test data/processed/field_mix/fold-0/test.jsonl \
  --output artifacts/field-mix/fold-0-final --base-model BAAI/bge-small-en-v1.5 \
  --epochs 14 --batch-size 96 --max-length 160 --patience 4 --save-each-epoch
python scripts/eval_field_transfer.py --rows data/processed/field_mix/fold-0/test.jsonl \
  --checkpoint artifacts/field-crossinstitution-context/fold-0 \
  --checkpoint artifacts/field-mix/fold-0-final --name reference --name mix
# A Jev-shaped decision model, pretrained on this corpus and fine-tuned on
# 42 described options. v2 is 15.3M parameters, trained on 18.7k more real rows
# (Columbia's directory of classes, the Coursera catalog), and matches the 33M
# bge reference on the same 7,571 rows: 14 errors each, macro-F1 0.9711 vs
# 0.9713, at 46% of the parameters. See artifacts/jev-field-v2/EXPERIMENT.md;
# the earlier 10.3M run, which transfers better to unseen universities (0.8322
# vs 0.7830 on fold 0), is artifacts/jev-field/EXPERIMENT.md.
python -m pace.field_corpus_ext           # + Columbia (37 terms) and Coursera
python -m pace.field_mix --context data/processed/field_context_ext/corpus.jsonl \
  --split-dir data/processed/field_context_ext --output data/processed/field_mix_ext \
  --extra-values data/processed/field_uncapped/corpus.jsonl --value-diversity-cap 20000
python -m pace.jev_train --stage tokenizer --output artifacts/jev-field-v2 \
  --unlabelled data/processed/field_context_ext/corpus.jsonl data/processed/field_uncapped/corpus.jsonl
python -m pace.jev_train --stage pretrain --output artifacts/jev-field-v2 \
  --unlabelled data/processed/field_context_ext/corpus.jsonl data/processed/field_uncapped/corpus.jsonl \
  --pretrain-rows 1500000 --pretrain-epochs 3 --pretrain-max-length 160 \
  --hidden 320 --ffn 1024 --batch-size 128
python -m pace.jev_train --stage choice --output artifacts/jev-field-v2 \
  --encoder artifacts/jev-field-v2-pretrain --tokenizer artifacts/jev-field-v2/tokenizer.json \
  --train data/processed/field_mix_ext/train.jsonl \
  --validation data/processed/field_mix_ext/validation.jsonl \
  --test data/processed/field_mix_ext/test.jsonl \
  --epochs 10 --batch-size 192 --lr 8e-4 --patience 3 --option-dropout 0.15 \
  --sampler-power 0.5 --hidden 320 --ffn 1024
python -m pace.jev_eval --checkpoint artifacts/jev-field-v2 \
  --rows data/processed/field_mix_ext/test.jsonl \
  --baseline artifacts/field-classifier-mix-fair \
  --output artifacts/jev-field-v2/test_metrics.json
python -m pace.jev_cross --encoder artifacts/jev-field-v2-pretrain \
  --folds-root data/processed/field_mix_ext --output artifacts/jev-cross-v2 --folds 4 \
  --epochs 9 --patience 2 --sampler-power 0.5 --hidden 320 --ffn 1024
python -m pace.cli classify-field "Prerequisites: CSCI 270 or equivalent" \
  --checkpoint artifacts/jev-field-v2
# Infinite Campus / OneRoster exports become labelled rows (district Course Master,
# classes.csv, lineItems.csv, results.csv). See artifacts/jev-field-v2/EXPERIMENT.md.
python -m pace.sis_import --input exports/ --school "Lincoln High" \
  --output data/processed/sis/sis_corpus.jsonl
python -m pace.field_corpus_ext --sis data/processed/sis/sis_corpus.jsonl
# Inference: one loose value, or a section read the way a person reads it
python -m pace.cli classify-field "Office Hours: Mon 2-4pm" \
  --checkpoint artifacts/field-classifier-minilm
python -m pace.cli classify-section --heading "Office Hours" --value "Wednesdays 3-4 PM" \
  --checkpoint artifacts/field-classifier-minilm \
  --context-checkpoint artifacts/field-classifier-context
# Is the percentage true? Reliability, ECE, certified precision at a threshold
python -m pace.field_calibration --checkpoint artifacts/field-classifier-minilm \
  --data data/processed/field/group/test.jsonl --output artifacts/field-calibration/capped_test.json
python -m pace.field_calibration --folds-root artifacts/field-crossinstitution \
  --corpus data/processed/field/corpus.jsonl --fold-plan data/processed/field/fold_plan.json \
  --output artifacts/field-calibration/unseen_universities.json
# Same rows, several checkpoints
python -m pace.field_compare --data data/processed/field_uncapped/group/test.jsonl \
  --checkpoint capped=artifacts/field-classifier-minilm \
  --checkpoint uncapped=artifacts/field-classifier-uncapped
# Transcript cells are a table, not prose: audit what the model does with them
python -m pace.field_transcript_audit --context-checkpoint artifacts/field-classifier-context
# The extractor: raw syllabus text -> structured record with evidence and confidence
python -m pace.field_extract path/to/syllabus.txt      # or '-' to read stdin
python -m pace.cli extract-fields path/to/syllabus.txt # the same, as a pace subcommand
python -m pace.field_extract data/samples/mock_syllabus.txt --output artifacts/record.json
# The application record: one schema, evidence on every value, no document-type claim
python -m pace.cli understand path/to/syllabus.txt   # schema pace.academic-ir/0.2
python scripts/audit_model_layer.py                  # re-measure what the model layer does today
# How accurate is it? Dev is the tuning loop; test is scored once, at the end
python -m pace.field_extract_eval --write-split
python -m pace.field_extract_eval --split dev
python -m pace.field_extract_eval --split test
python -m pace.field_extract_eval --protocol-report   # writes EXTRACTION_EVAL.md
# How well does it hold on documents from institutions in neither split?
python -m pace.field_heldout_eval --output artifacts/field-extract/heldout_eval.json
# Rebuild the value-level gold corpus from the two institutions' public listings
python scripts/fetch_eval_syllabi.py --per-school 6
```

### The extractor's output

`python -m pace.field_extract` is the one command to run: it takes raw syllabus or course-catalog text (file or stdin), cleans the HTML, types the sections with the existing field checkpoints, and returns

```json
{
  "course":   {"course_title": {"value": "...", "confidence": 0.99, "evidence": {"text": "...", "start": 120, "end": 149}}, "course_code": {...}, "credits": {...}, "instructor": {...}},
  "grading":  [{"category": "Homework", "weight": 20, "confidence": 0.99, "weights_sum": 100}],
  "meetings": [{"days": ["MO", "WE", "FR"], "start_time": "09:10", "end_time": "10:00", "room": "Pimentel Hall", "alternatives": false, "confidence": 1.0, "evidence": {...}}],
  "policies": {"office_hours": {"value": "...", "confidence": 0.99, "evidence": {...},
                            "candidates": [{"value": "...", "confidence": 0.99, "section": "...", "status": "chosen", "reason": "..."},
                                           {"value": "...", "confidence": 0.7, "section": "...", "status": "overwritten", "reason": "a later section supplied the value"}]},
              "attendance_policy": {...}, "academic_integrity": {...}, "disability_services": {...}, "materials": {...}},
  "suppressed": [{"field": "course_title", "value": "Grading", "reason": "a section heading", "evidence": "..."}],
  "grading_total_percent": 100,
  "grading_total_consistent": true,
  "sections_typed": 11,
  "min_confidence": 0.0
}
```

A full worked example — a short syllabus, the one command, the verbatim record it prints, and a walkthrough of every field and every suppressed entry — is in [docs/example.md](docs/example.md).

Every value carries the line it came from (`evidence.text`) and its offsets, so a wrong answer can be argued with. Titles are grounded: the string returned must occur verbatim in the document. Confidence is per value, never per record: a missing field is absent, not low-confidence.

The record does not hide what it discarded. Every policy field carries `candidates` — each section read for it, with `status` (`chosen`, `overwritten`, `rejected`, `below min_confidence`) and the reason — and a top-level `suppressed` list records the title that lost the ranking, the code that looked like a room, the nested sub-weight, the duplicate row and the row outside the sum-to-100 subset (`pace.field_ledger.suppressed_values(record)` returns it, and that module is the only place the loss shape is defined). Nothing is capped: every rejected candidate is listed, and a line that reads like a schedule or a grading row but produces no value is listed with the reason it produced none. The coverage ledger is measured on the development split: **214 emitted against 794 suppressed**, per field and per reason in [REPORT.md](REPORT.md#what-the-record-throws-away--the-coverage-ledger) — so the precision below can be read next to what it cost. One rule decides whether a breakdown adds up (`100 ± 5`), used by the per-value flag and the record's `grading_total_consistent`. The test split was not re-scored to build it.

Three gold sets measure it. The published section-span gold covers 14 untouched corpus documents (MIT, Ohio State, UNC) and scores **4/7 = precision 0.571, lower bound 0.225** on the values it can still judge (21 emitted, down from 37 as the stricter rules print less); that sample is too small to carry an estimate. A stricter value-level gold was read field by field off **12 documents from two institutions that are not in the classifier corpus at all** (Yale, Florida), marking every named field the document states *and* those it does not: **63/66 = precision 0.955, 95% lower bound 0.887**, against 51/68 = 0.750 / 0.649 on the same sample before that run. The third set, and the only one built to answer the accuracy question rather than to support a tuning loop, is **20 documents from the two corpus institutions that are in neither split** (UC Berkeley, Georgia Tech), labelled under a rubric written before any extractor output existed on them and frozen with a recorded hash. It scores **50/61 = precision 0.820, 95% lower bound 0.719** — 11 wrong values, 73 emitted, 245 read and suppressed. After a title fix that was written knowing which five titles were wrong there, the same 20 documents were scored once more: **50/59 = 0.847, bound 0.749**. That second touch is contaminated by construction and is not an independent estimate; the frozen 0.820 remains the number to quote. [Both are in the report](REPORT.md#the-second-touch-the-same-20-documents-after-the-title-fix). That is well below 0.98, and the sample is large enough to say so rather than merely fail to prove otherwise (a 0.98 extractor would produce ~1.2 errors in 61 values; 11 were observed, p ≈ 3×10⁻⁸). It is also an upper bound rather than a clean estimate: those two institutions' documents were read while the very first rules were written, and the encoder trained on rows from both. Per-field breakdown, error list and the loss ledger: [REPORT.md](REPORT.md#the-held-out-measurement-20-documents-from-institutions-in-neither-split).

The confidence now means more than it did: each percentage comes from the value's own evidence (a label, the shape the field requires, a day and clock, a body about the field) instead of the section-type softmax, and precision rises with it — 1.000 in the 0.70–0.90 band, 0.933 in 0.90–0.99, where the last pass had 0.857 at the bottom and 0.529 at the top. It is still not monotone and 0.955 is still not certified at 0.95; the wrong values are named in the report. Nor can 66 judged test values certify 0.98 — a perfect 66/66 would bound it at only 0.956, and a 0.98 bound at zero misses needs at least 149 judged values. No threshold of `min_confidence` certifies 0.95 (or 0.90), so the default stays at `min_confidence=0.0`; every positive threshold keeps fewer values and none measures a better bound. Development, where every fix was measured, scores 0.912 on span gold. What was fixed — and what stayed open — is recorded row by row in [known_failures.json](data/processed/field/known_failures.json). The recorded test run predates the office-hours heading and opening-availability rules added since; those were measured on development only, and the test split was not re-scored. Protocol, band table, threshold sweep and the full failure catalogue: [REPORT.md](REPORT.md#task-6-the-extractor--from-a-classified-span-to-a-filled-record).

The raw datasets and computed statistics are in `data/raw/` and `data/inventory.json`. Run the baseline first to write the shared subject split; the real training run writes subject-held-out natural cell examples, tokenizer/config, PyTorch weights and per-class metrics under `artifacts/usc-miniLM/`. See `experiments/README.md` and `docs/research.md` for rationale, license scope and limitations. The local sample command consumes already-loaded local text; all predictions are proposals, not committed academic facts.

## Mobile runtime

The runtime decision was made by measurement. The shipped reader is an **ONNX** export of the MiniLM field classifier plus a flattened document gate, running through `onnxruntime-react-native` inside the Expo app; the TypeScript reader that drives it is the same code the Node parity harness runs. fp32 ships because int8 agreed with it on only 87.5% of probe rows; ExecuTorch and Core ML remain alternatives behind the same `SessionFactory` boundary rather than requirements. Numerical parity is measured on this machine against the Python pipeline (135 checks); **phone latency, memory and the iOS build are not**. See [docs/APP.md](docs/APP.md).

## Data, privacy, and license

`data/raw/` contains downloaded public research data; see [inventory](data/inventory.json) and [research notes](docs/research.md) for dataset-specific licensing caveats. Avoid committing personally identifying data or using real student records. USC instructor fields are excluded from the training examples. The GU syllabus corpus was previously held for rights review only; it is now used for local research training of the field classifier because no permissively licensed syllabus corpus was available — that trade-off is recorded in [research notes](docs/research.md#real-text-field-classifier-corpus-additional-sources-and-their-rights) and it does **not** clear the corpus for redistribution. Respect original-document rights within CU-DocBench despite its dataset-level MIT tag.
