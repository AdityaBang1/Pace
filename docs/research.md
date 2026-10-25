# PaceDecide research and evidence log

Updated: 2026-10-21. Dataset rights discussion is informational, not legal advice. Repository visibility or a card license does not necessarily establish all underlying rights; retain item-level attribution and seek clearance where needed.

## Named real datasets: inspected and downloaded

| Dataset | Downloaded data / inspected statistics | Dataset-card license | Fit and use decision |
|---|---|---|---|
| [anordkvist/gu-course-syllabus](https://huggingface.co/datasets/anordkvist/gu-course-syllabus) | 2,971 natural Göteborg University syllabus rows; one Parquet file, 7,025,801 bytes / 7.03 MB; 11 fields plus index. All-field row token count with MiniLM tokenizer: mean 1,277.82, median 1,209, p90 1,697, p95 1,945, max 8,621. Individual nonempty field cell tokens: 32,681 values, mean 118, median 77, p90 265, p95 343, max 4,677. | No license field/tag and card says “More Information needed.” | Inspected and inventoried, not used for training pending rights clarification. High-value future syllabus dataset; preserve natural fields. If licensed, evaluate each field independently or test natural heading windows based on these actual stats. Do not concatenate whole rows into MiniLM. |
| [USC/USC-Course-Catalog](https://huggingface.co/datasets/USC/USC-Course-Catalog) | 5,571 real course-section CSV rows, 2,133,605 bytes; fields: program, class id/name, catalogue, units, prerequisites, restrictions, section, time, days, type, instructor. Model input cells: median 8 MiniLM tokens, p95 34, max 145. 1,701 explicit prerequisite cells; 1,384 restrictions. | HF card declares Apache-2.0. The card describes contributor-scraped USC Spring 2024 classes; uploader authority not independently confirmed. | Used for bounded field classification; individual natural cells preserved. Exclude instructor names and identifiers from inputs. Seek rights confirmation before commercial redistribution. |
| [iristun/CU-DocBench](https://huggingface.co/datasets/iristun/CU-DocBench) | JSON 4,190,439 bytes; 23 documents, 100 gold pages, 1,065 blocks; four syllabus docs / 12 syllabus pages. Gold page MiniLM tokens: median 266, p90 1,066, p95 1,563, max 3,117. | HF card declares MIT; source PDFs may have separate rights and some transcript pages are mock. | Used only for evaluation of the dataset's bundled `gemini-3-pro-preview` page-block predictions against block IDs/types. No training or schedule extraction claim. |
| [pchristm/PerQA](https://huggingface.co/datasets/pchristm/PerQA) | A deterministic sample of 20 `observable_events` and 20 first `canonicalized_events` rows (the evaluation uses only the 18 canonical IDs linked by the observable sample); stored at [perqa_test_sample.json](../data/samples/perqa_test_sample.json). It checks structured event JSON and ISO-like date/time fields, not question SQL or natural-language extraction. | Dataset card declares CC-BY-4.0; attribute Philipp Christmann and Gerhard Weikum. | Small structured synthetic temporal/event-field smoke test only. Not a syllabus dataset, natural-language task, schedule-scanner evaluation, training set, or calibration data. The local sample is pinned to one revision/config/split; no full corpus download. |

Full stats live in [data/inventory.json](../data/inventory.json). Raw downloads under [data/raw](../data/raw). Regenerate inventory using `python -m pace.inspect_datasets`. Viewer/API cards were checked on 2026-10-19. PerQA rows were fetched from the Dataset Viewer API at the pinned revisions recorded in the manifest and sample JSON, with only bounded offsets. See [GU viewer](https://huggingface.co/datasets/anordkvist/gu-course-syllabus), [USC viewer](https://huggingface.co/datasets/USC/USC-Course-Catalog), and [CU-DocBench viewer](https://huggingface.co/datasets/iristun/CU-DocBench).

## Real-text field-classifier corpus: additional sources and their rights

The academic field classifier (`pace/field_corpus.py`, `pace/field_sources.py`) is trained on **163,797 real rows** drawn from eight public datasets. Six of them are new since the table above. Their cards were read on 2026-10-20.

| Dataset | Rows used | Content | Card license | Rights decision |
|---|---:|---|---|---|
| [twangodev/uwcourses](https://huggingface.co/datasets/twangodev/uwcourses) | 84,034 | UW–Madison course guide: 8,930 current courses plus ~700k real class meetings (building, room, start/end, meeting type, instructor, section capacity, instruction mode, term dates) | **none declared**; card says not affiliated with or endorsed by UW–Madison | Used for local research training. Not cleared for redistribution. Only the non-generated columns are used — every `llm_*` column is ignored. |
| [anordkvist/gu-course-syllabus](https://huggingface.co/datasets/anordkvist/gu-course-syllabus) | 46,338 | 2,971 Göteborg University syllabi × 11 natural field passages | **none declared** | **Changed from the earlier decision in this file.** It was previously "inspected, not used for training". It is now used, because the field classifier needs real syllabus prose and no permissively licensed syllabus corpus was available. Still no redistribution clearance. |
| [chaseharmon/MIT_Course_Catalog](https://huggingface.co/datasets/chaseharmon/MIT_Course_Catalog) | 14,107 | 5,794 MIT catalog rows: class number, name, description, term, level, units, prerequisites, equivalents, lab component | **none declared** | Used for local research training. MIT catalog text is institutionally published; the mirror's own rights are unconfirmed. |
| [Dev-the-dev91/syllabus-cleaned](https://huggingface.co/datasets/Dev-the-dev91/syllabus-cleaned) | 1,638 | 90 real syllabi from 14 universities with labelled section headings, entity spans and structured facts (grading weights, task due dates, class meeting patterns) | **none declared** | Used for local research training. Contains instructor names and, in places, contact details — treat as personal data; not for redistribution. |
| [Dev-the-dev91/syllabus-finetune](https://huggingface.co/datasets/Dev-the-dev91/syllabus-finetune) | 907 | The same 90 syllabi with annotated entity spans (instructor, email, due date, assignment, reading, grading weight, concept) | **none declared** | Used for local research training. Contains names and email addresses — personal data. |
| [alecwatts/umich-course-descriptions](https://huggingface.co/datasets/alecwatts/umich-course-descriptions) | 292 | 469 UMich Atlas course-profile pages, including the published section table | **none declared** | Used for local research training. Contains instructor names. |
| [seansullivan/INT-Business-Syllabus](https://huggingface.co/datasets/seansullivan/INT-Business-Syllabus) | 0 in the final corpus | 55 extracted text chunks from University of Florida Warrington business syllabi | **`other`** (no terms identified) | Loaded and parsed; its sections are covered by the shared heading detector but its rows were deduplicated away in the heading-aware corpus. Not cleared for use. |

**What this means for the artifacts.** None of the six new cards grants a reuse licence. Public downloadability is not permission. The checkpoints and the derived corpus under `artifacts/` and `data/processed/field/` are therefore **local research artifacts only**: do not publish them, do not ship them in a product, and do not use them commercially until rights are cleared for each source document. The MIT and UW catalogues are institutionally published data mirrored by third parties; the four syllabus sets are third-party collections that include instructor names, emails and, in one case, course-instructor contact details. A deployment would need either a licensed corpus or a per-university permission, and the pipeline would need the personal fields stripped (`instructor`, `contact_information`) before any release.

This is a deliberate change of posture from the earlier "inspect only, do not train" rule for the GU corpus. It is recorded here so the trade-off is visible: real training data was needed to measure anything real, and the price is an unresolved rights position on every new source.


Other sources reviewed: MIT OCW permits AI/ML training under CC BY-NC-SA for covered content, with attribution/noncommercial/share-alike and item-specific third-party terms ([terms](https://ocw.mit.edu/pages/privacy-and-terms-of-use/)); one individual CUNY CC BY 4.0 syllabus record was found but PDF access was 403; UIUC public course CSV/API has no verified blanket reuse license; Open Syllabus does not grant blanket underlying-document rights. Public availability alone is not training authorization. See [Open Syllabus](https://docs.opensyllabus.org/), [UIUC course catalog](https://discovery.cs.illinois.edu/dataset/course-catalog/), [US Copyright Office AI materials](https://www.copyright.gov/ai/).

## What a field classifier actually learns (measured, 2026-10-21)

The academic field classifier answers "which field did this text come from?" over 42 types. The interesting result is not the in-distribution accuracy (0.98926 ± 0.00009, macro-F1 0.93260 ± 0.00079, ECE 0.0021 on a group-held-out test split) but what happens when the *source institution* changes.

**A value does not carry its field.** `"3.0 units"` is `credits` only because of the column it came from. So a text-only model trained on loose values learns each institution's layout, and two leak-free protocols say so:

- *Leave-one-source-out* (a whole dataset removed from training, then evaluated on it): MIT 0.6294, USC 0.4643, UW 0.4104, real syllabi 0.2149, GU 0.1054 accuracy. With GU held out, 5,387 `entry_requirements`, 5,246 `course_content` and 4,812 `teaching_form` rows are all read as `course_description` — the model learned "long prose = course_description" from the one source with that shape.
- *Cross-institution* (14 universities dealt into 3 disjoint folds, each fold training its own model, the early-stopping epoch chosen on a slice of the **training** universities): 0.65 accuracy / 0.42 macro-F1 pooled over 1,638 real syllabus rows, with ECE rising from 0.002 to 0.22–0.33. The percentages the model reports on an unfamiliar document are overconfident.

**The heading is the transferable signal.** 446 real labelled sections from 83 real syllabi were scored against the truth implied by the section heading itself — a label that does not depend on the model:

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| heading only (`"Office Hours"`) | **0.8812** | 0.7428 |
| body only (`"Wednesdays 3-4 PM"`) | 0.6928 | 0.5006 |
| heading + body, naive concatenation | 0.7578 | 0.5509 |

Naive concatenation is *worse* than the heading alone, because the body dominates the token budget. A second model trained on heading-plus-value pairs (`"Office Hours: Wednesdays 3-4 PM"`) reaches 0.8453 / 0.6920 on the same sections, and — the point — it is far more accurate on universities it has never seen. A heading-first cascade (classify the heading; fall back to heading+value only when the heading's calibrated confidence is below 0.8) gives 0.8879 accuracy / 0.7788 macro-F1 with a 7.4% fallback rate.

Running the heading-aware model through the *same* three cross-institution folds, read from the *same* fold plan, gives mean accuracy 0.8701 ± 0.0284 and macro-F1 0.6739 ± 0.0194 against the bare model's 0.6509 / 0.4131, and ECE falls from 0.253 to 0.101. The two corpora are different sizes, so the decisive comparison is the paired one: on the 1,326 rows both corpora contain, the bare model scores 0.5904 accuracy / 0.3640 macro-F1 and the heading-aware model scores **0.8727 / 0.6849** — **+28.2 accuracy points and +32.1 macro-F1 points on identical rows.**

The corollary is a design rule for anything that reads real academic documents: **read the document's structure, not its loose values.** The value alone is a badly-posed input.

Artifacts: [field classifier report](../REPORT.md#task-5-academic-field-classifier-real-text-42-field-types), [model card](../artifacts/field-classifier-minilm/MODEL_CARD.md), [leave-one-source-out](../artifacts/field-ood/ood_report.json), [cross-institution](../artifacts/field-crossinstitution/cross_institution_report.json), [heading vs body vs cascade](../artifacts/field-ood/context_eval_cascade.json), [paired bare-vs-heading comparison](../artifacts/field-ood/paired_comparison.json).

## Small multi-dataset evaluation (tasks are deliberately separate)

Reproduce the new lightweight evaluation offline with `python -m pace.evaluate_datasets --output artifacts/multi-dataset/evaluation.json`. It reads the existing local CU-DocBench JSON and the checked-in PerQA row sample; it does not train a model, download the full PerQA corpus, or alter scanner inference/confidence. Before parsing, it verifies the CU-DocBench file's SHA-256 (`657965beddc424193cde29c0463aedb5146478db21c18a61021856d0815c1873`) recorded alongside dataset revision `c00e12b958ad01c0c6a5ae10d8f407160fbd0c20` in the manifest. A different or reformatted input is rejected rather than evaluated under the pinned provenance. The output has two separate metric objects and no cross-task aggregate.

- **CU-DocBench layout:** scores the bundled Gemini 3 Pro Preview predictions against gold layout block IDs/types on all 23 docs / 100 pages. Page-local block ID matches define association; missing/failed pages count their gold blocks as false negatives, and extra or mistyped blocks count as false positives. This reports layout/type agreement of the supplied VLM predictions—not Pace schedule extraction or a model trained here. Current snapshot: 1,055/1,065 gold blocks matched (99.06% coverage), type accuracy over gold 96.53%, micro-F1 95.49%. These are benchmark-source predictions and may not be independent of benchmark creation.
- **PerQA structured temporal smoke test:** uses config `observable_events` and `canonicalized_events`, split `test`, offset 0, lengths 20 each, pinned to revision `ff40fdce3875a6fc262e04d2cdd5de610b2211a5`. It checks row links, JSON column shape, and ISO date/time validity; it does not evaluate language understanding, question answering, SQL, or syllabus extraction. On this sample, 20/20 event JSON and properties fields parsed, and 78/78 non-null temporal values were valid. Because canonicalization may consolidate or retype observed events, this protocol checks stable persona/split links, not equality of event content.

The PerQA sample is synthetic, though the dataset's questions/tasks describe temporal event queries. It is only a schema/structured temporal-data smoke check. Its exact checked-in sample bytes are pinned by SHA-256 in the manifest and verified at load time; modified or reformatted samples are rejected under the recorded revision. Neither dataset is used to train or calibrate the schedule scanner, and inference confidence is unchanged. Real syllabus schedule calibration remains blocked pending independent, human-labeled, licensed, task-matched data. Dataset card licenses do not necessarily clear rights in underlying source documents; retain attribution and review item-level rights.

## First task and label ambiguity

USC's natural schema supports one narrow task: classify individual natural course cell text as course title, description, credits, prerequisites, restrictions, meeting time, meeting days or class type. This is **not** raw document parsing, OCR evaluation, entity extraction or prerequisite graph construction.

A source audit found identical normalized `TBA` values in both `Days` and `Time`: 1,317 and 1,335 cells respectively. A text-only model has no way to infer the source column from that identical text. The final text-only experiment excludes these 2,652 conflicting cells before the subject split, and the inference API abstains on normalized `tba`. This is not synthetic cleaning or correctness cherry-picking: both label targets are true in source metadata, but their input text is uninformative. The ambiguity exclusion map and counts are recorded in model metrics and [data manifest](../data/manifest.json).

Corrected source-group split: 23,175 train, 5,102 validation and 5,522 test cells from 144/30/31 whole subject prefixes. Each model consumes the same exact `artifacts/usc-real/split_assignment.json`.

## Corrected held-out real data comparison

| Model | Accuracy | Macro-F1 | Weighted-F1 | Artifact | Fit |
|---|---:|---:|---:|---:|---|
| Char 2–5 n-gram TF-IDF + logistic regression | 0.99728 | 0.99742 | 0.99728 | 1,963,196 B | 1.85s CPU |
| Fine-tuned all-MiniLM-L6-v2 | 0.99982 | 0.99986 | 0.99982 | 90,893,327 B checkpoint | ~30–31s/epoch, RTX 4070 |
| Fine-tuned DeBERTa-v3-xsmall | 1.00000 | 1.00000 | 1.00000 | 282,794,003 B checkpoint | ~98–106s/epoch, RTX 4070 |

All are held-out USC course-subject prefixes, not held-out institutions. Remaining errors: sparse 15, MiniLM 1, DeBERTa 0. DeBERTa's score is the best here, but is ~3.1x larger and over 3x slower to train per epoch than MiniLM; MiniLM is the pragmatic neural model while sparse is the small option. One dataset/term/seed is no evidence of general application competence.

Real validation fitted temperature and test risk/coverage, not posthoc test calibration: MiniLM T=0.7242, ECE-10 0.000088, confidence >=.90 coverage 100%, precision 99.982%. DeBERTa T=0.1102, ECE 0.0, coverage/precision 100% on this finite test. These high-confidence results are conditional on this highly structured, exact-cell distribution and should not be extrapolated to OCR/domain shifts.

Artifacts: [comparison](../artifacts/usc-real/comparison.json), [sparse metrics](../artifacts/usc-real/metrics.json), [MiniLM metrics](../artifacts/usc-miniLM/metrics.json), [DeBERTa metrics](../artifacts/usc-deberta-xsmall/metrics.json).

## Synthetic-only multitask document understanding

A second, deliberately separate experiment exercises document-level reading: document-type classification (10 classes) plus token-level entity extraction (19 types) over 30,000 procedurally generated academic documents. It is **synthetic-only**. It has no training or validation on real OCR, PDFs, syllabi or institution documents, and its numbers are not comparable to the real-data tasks above.

- Split: whole synthetic school identities and whole layout/template families held out (42/9/9 schools, 20/10/11 template families; 10,279 / 1,069 / 1,208 documents), evaluated only on school/template intersections. All ten document classes occur in every split.
- Model: shared `all-MiniLM-L6-v2` encoder, 6 layers kept, with a document-type head and a BIO entity head. 22,732,081 parameters; checkpoint 90,958,791 B; dynamic INT8 58,654,218 B (0.6448x) at entity-span F1 0.99826.
- Held-out exact entity-span F1 (type + character offsets, 20,382 gold spans): 0.99968 clean, 0.99909 fully corrupted, 0.99953 with only context corrupted. Document type: 1.000 accuracy and macro-F1 for both the multitask model and an identical-data TF-IDF baseline. A task both a neural model and a bag-of-words baseline solve perfectly is a cue-easy task; these figures say nothing about real documents.
- **Size/accuracy was measured, not asserted.** Three encoders differing only in depth (6/4/2 layers) were trained on the same corpus, seed and hold-out: 22.73M/19.18M/15.63M parameters, corrupted span F1 0.99909/0.99890/0.99674, CUDA batch-1 window p50 5.08/3.58/2.23 ms. The 2-layer model is 31.2% smaller for 0.24 F1 points; the 4-layer model saves 15.6% of bytes for 0.02 points.
- One image-derived transcript (`data/samples/mock_transcript_lepid_rowwise.txt`) was run end to end: document type `transcript` (p = 0.9929), 220/220 model entity spans matched, 38 course records across 8 course subject areas plus 3 assessment sections, computed credits 13.25 matching the reported 13.25, 0 records missing a title or section, GPA absent and reported as null. This is a single document, not a benchmark.

Calibration remains blocked for this checkpoint: there is no independent human-labeled calibration set, so the reported `confidence` is a distribution-sharpness statistic only, and no calibrated claim is made. Artifacts: [model card](../artifacts/synthetic-multitask/MODEL_CARD.md), [metrics](../artifacts/synthetic-multitask/metrics.json), [depth comparison](../artifacts/synthetic-multitask/depth_comparison.json), [transcript report](../artifacts/synthetic-multitask/transcript_report_rowwise.json).

## Current compact encoders

| Candidate | Size and context | Rights / evidence-based status |
|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22.713M encoder parameters measured; sentence encoder, model card truncates >256 wordpieces | Apache-2.0; trained on real catalog cells. |
| [DeBERTa-v3-xsmall](https://huggingface.co/microsoft/deberta-v3-xsmall) | 12 layers, hidden 384, 512 max positions; 70.685M parameters including 128,100 embedding vocabulary | MIT; trained on same task/split, perfect on this narrow test; large cost for tiny marginal score improvement. |
| [MobileBERT](https://huggingface.co/google/mobilebert-uncased) | 24 layers, 512 positions, 128 embedding/512 hidden bottleneck; published ~25M params | Apache-2.0; plausible mobile candidate, not evaluated in training yet. |
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 149M, 8,192 positions | Apache-2.0; unnecessary for short catalog cells, possible later long text candidate. |
| [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) | 4 layers, hidden 312 | No explicit Hub license discovered; not used pending rights. |

Model cards/configs and papers are linked here: [MiniLM](https://arxiv.org/abs/2002.10957), [DeBERTa v3](https://arxiv.org/abs/2111.09543), [MobileBERT](https://arxiv.org/abs/2004.02984), [ModernBERT](https://arxiv.org/abs/2412.13663).

## Measured local inference

For MiniLM, batch size 1 on RTX 4070, 120 natural USC cell examples, including tokenizer and PyTorch forward but excluding model load/OCR/app: p50 2.739ms, p95 3.024ms, ~360 examples/s; input token p50 8, p95 34; forward peak CUDA allocated 103,250,944 bytes. This is a desktop GPU measurement, not an iPhone/Android performance result. Saved at [benchmark.json](../artifacts/usc-miniLM/benchmark.json). There is no phone benchmark, model quantization or mobile runtime package yet.

## Deployment runtime research (decouple architecture)

- [ExecuTorch](https://docs.pytorch.org/executorch/stable/using-executorch-export.html): regular PyTorch model captured by `torch.export`, lowered to backend-specialized `.pte`; mobile options include XNNPACK CPU, Core ML delegate on iOS, Vulkan Android and others. Backend-specific export artifact may be needed per hardware target.
- [ExecuTorch quantization](https://docs.pytorch.org/executorch/stable/quantization-overview.html): torchao PT2E, backend-specific quantizer; calibrate, convert, benchmark task quality before deployment.
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/): ONNX model format with iOS and Android runtime packages; CPU/XNNPACK and optional CoreML/NNAPI execution providers.
- [ONNX quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): dynamic/static INT8; documentation points to dynamic as common first method for transformer models.

No ONNX/ExecuTorch/Core ML artifact or physical iOS/Android test has been produced. Preserve architecture in PyTorch, then compare export numerical parity, package/app size, quantized quality, cold/warm latency, memory and runtime on actual phones. Don't declare a mobile winner from documentation alone.

## Where extraction, loss accounting and measurement live

The extractor now records the values it rejects, which separates three jobs that used to blur together. Later passes should keep this boundary:

- `pace/field_extract.py` decides what the document says: format-specific rules read values with evidence spans and a confidence computed from that evidence. It decides whether a value is good enough for the record and reports the ones that are not; it does not count or group losses.
- `pace/field_ledger.py` owns the loss side and imports nothing from the rest of the project: `Ledger.note()` fixes the record's one loss shape (`field`, `value`, `reason`, `evidence`) and its truncation, `suppressed_values()` is the read side, and `reason_group()`/`summarize()` fold per-item labels (`duplicate of 'Homework'`) to their rule so per-field counts add up to the field's loss total.
- `pace/field_extract_eval.py` measures and reports: it consumes the ledger's read side, composes coverage (emitted vs suppressed vs gold) with the values it actually scored, and renders the tables. It derives no reason strings of its own.

Data flows one way — extraction and measurement both depend on the ledger, the ledger depends on neither — so the ledger can be tested without a checkpoint and the record's shape is changed in one file.

## Known gaps

- USC scraped dataset card declares Apache-2.0 but underlying authorization remains unconfirmed. Six further corpus sources declare **no** licence at all, and the GU syllabus has none either, so the field-classifier corpus and its checkpoints are local research artifacts only — not cleared for redistribution or commercial use. See [the rights table above](#real-text-field-classifier-corpus-additional-sources-and-their-rights).
- The field classifier is measured on *sections and cells*, not on end-to-end document parsing: there is no OCR path, no table-structure recovery, and no joint field+value extraction from a scanned syllabus.
- Cross-institution accuracy is 0.65 for the bare-value model. The heading-aware model and the heading-first cascade close most of that gap, but both need a *document structure* to read headings from, so they do not apply to a bare list of values.
- No real schedule-scanner calibration on human-labelled licensed data. CU-DocBench has only four syllabus documents and underlying source rights need checking. PerQA is synthetic structured-event data and is not a substitute.
- No OCR/PDF image pipeline and no real-document NER or entity-matching task has been trained. The grounded course-mention relation classifier and the entity extractor both exist, but the relation classifier is trained on real catalog text and the entity extractor is synthetic-only; neither has been evaluated on a real scanned document.
- No human-adjudicated span gold exists for the synthetic entity extractor, and the generator is demonstrably easier than reality (both document models are perfect on it), so its held-out F1 must not be read as a real-document score.
- The extractor's per-value percentage is not yet a probability of correctness. On 12 hand-labelled documents from two institutions absent from the training corpus, precision is 0.955 (95% lower bound 0.887) after the score was moved off the section softmax onto each value's own evidence, but it is still not monotone — the remaining errors sit at 0.90–0.95 with the true values — and no `min_confidence` setting certifies 0.90 or 0.95 precision, so the field must not be presented as a calibrated confidence. The record also shows what it discarded, without a cap: 214 values emitted against 794 suppressed on the development split, per field and reason in [REPORT.md](../REPORT.md#what-the-record-throws-away--the-coverage-ledger). One named rule (`grading_sum_consistent`, 100 ± 5) decides whether a breakdown adds up, for the per-value flag and the record's flag alike.
- Inference on arbitrary sentence fragments can be confidently wrong; local API echoes evidence and marks proposal-only, but its score is not a fact guarantee.
- No multi-institution, student-owned, or source-shifted evaluation; no phone/mobile benchmarks.
