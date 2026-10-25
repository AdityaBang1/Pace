# Model-layer audit: what is actually implemented

Measured on this checkout on 2026-10-24 with `scripts/audit_model_layer.py` and
`pace.api`. Raw output is beside this file as `00_inventory.json` …
`06_api_e2e.json`. Nothing below is quoted from a plan, a README or an earlier
conversation; where a number comes from a recorded artifact, the artifact is
named.

---

## 1. EXECUTIVE VERDICT

**Yes with small blockers — but only for one vertical slice.**

The repository can read a syllabus into a grounded, versioned record with
per-value evidence. It cannot tell you what kind of document it is looking at,
cannot match two mentions of one course, and has no relation layer on real
documents. Application work can start if the app asks the user for the document
type and confirms every field. It cannot start if the app is supposed to detect
the document type, import a schedule, or link a syllabus to a transcript.

## 2. CURRENT BEST MODEL

There are three field checkpoints and one document-type model. They do not share
a tokenizer and they were not trained on the same rows.

| | jev-field-v2 | field-classifier-minilm | field-classifier-context |
|---|---|---|---|
| Architecture | ELECTRA-shaped choice model (12×320, 4 heads, ffn 1024) | all-MiniLM-L6-v2 + linear head | same as minilm |
| Parameters | 15,331,563 | 22,729,386 | 22,729,386 |
| Checkpoint | `model.pt` 61.4 MB fp32 | 90.9 MB fp32 | 90.9 MB fp32 |
| Tokenizer | WordPiece 16 K (`tokenizers.Tokenizer`) | `BertTokenizerFast` | `BertTokenizerFast` |
| Input | `[CLS] text [SEP] question [SEP]`, lowercased, **127 text tokens** | raw text, **256 tokens** | `heading: body`, 256 tokens |
| Output | 42 label probabilities + `top` + temperature | same shape | same shape |
| Temperature | 0.6255 | 0.6467 | 0.5290 |
| Trained | 200,985 rows, extended corpus | 135,057 rows, older corpus | heading+value corpus |
| Test set | `field_mix_ext/test.jsonl`, 7,571 rows | `field_mix/test.jsonl`, 16,896 rows | own set |

**Same-row comparison** (7,571 rows, scored now with `pace.jev_eval`):

| Model | Accuracy | Macro-F1 | ECE-10 |
|---|---:|---:|---:|
| jev-field-v2 | **0.9982** | **0.9711** | **0.0016** |
| bge-small-en-v1.5 (recorded baseline) | 0.9982 | 0.9713 | 0.0012 |
| field-classifier-minilm (deployed reader) | 0.9464 | 0.8486 | 0.0166 |

That last row is not a fair architecture comparison — the MiniLM checkpoint was
trained on the older corpus and has never seen the Columbia/Coursera rows in
this test set — but it is the honest description of the model the extraction
pipeline currently uses.

**Which one is the app's model?** `artifacts/field-classifier-minilm` plus
`artifacts/field-classifier-context`, because that is the configuration the
extraction gold set was measured with (0.955 precision on 12 held-out
documents). jev-field-v2 is the better field classifier, 30 MB smaller, and
loads in 0.07 s against 0.29 s, and on 12 real syllabi it extracts the same counts
(9 titles, 7 codes, 8 instructors, 8 breakdowns, 6 exact sums, 2 meeting
blocks). It is the recommended next swap, not the shipped default, because its
title scoring has not been re-measured field by field.

The document-type model (`artifacts/synthetic-doc-type`, 22.7 M) is **not
usable**: it is trained only on procedurally generated documents, and its
artifact does not load against the current class (see §3).

## 3. ACTUAL IMPLEMENTED CAPABILITIES

**Document classification — missing.** `artifacts/synthetic-doc-type` reports
0.9997 accuracy, but every row is generated text and a TF-IDF baseline also
scores 1.0, so the number measures the generator. The artifact fails to load
(`Missing key(s): doc_head.*, entity_head.*; unexpected: classifier.*`). The
loadable sibling (`artifacts/synthetic-multitask`) called a bank statement a
transcript at 0.9975 confidence with 25 "entities", a menu "other", a syllabus
"syllabus" at 0.9423.

**Section typing — real, and the strongest model capability.** 42 field types,
calibrated probabilities, ECE 0.0016 in distribution. On a real Yale syllabus:
`Instructor:` 0.9956, `Office Hours:` 0.9997, `Email:` 0.9925. It also called
`Learning goals` "course_title" at 0.9528 — a wrong label with a high band.

**Section segmentation — deterministic and closed-vocabulary.**
`detect_document_sections` only accepts a line whose normalised text is a key in
`SYLLABUS_SECTION_MAP`. A heading a registrar writes differently is not a
section; the paragraph then belongs to whichever heading preceded it.

**Value extraction — rules, not the neural model.** `pace/field_extract.py` is
~1,900 lines of patterns: percentages, course codes, clock ranges, day groups,
credit forms, title plausibility, nested-weight detection. The model is consulted
at only two points (title scoring and section typing). This is the part that
works, and it is mostly not ML.

**Relations — one task, on catalog cells only.** `pace/real_relations.py` labels
a course mention against a prerequisite clause: 5 labels, 3,212 USC rows, 652
test rows, MiniLM encoder macro-F1 0.9193 vs sparse-TF-IDF 0.8079. Tested live:

```
Prerequisite: CSCI 103 or CSCI 104; Corequisite: MATH 225  (self = CSCI 270)
  prerequisite_of -> CSCI 103  0.996
  prerequisite_of -> CSCI 104  0.996
  corequisite_of  -> MATH 225  0.993
```

The label distribution is 2,858 `prerequisite_of` against 13 `corequisite_of`,
36 `related_reference`, 146 `recommended_for`. It needs the cell and the course
id supplied; it does not find them in a document.

**Entity resolution — none.** There is no matcher. `canonical_course_code` only
normalises `subject + number`. Nothing compares "AP Chem" with "AP Chemistry",
and nothing rejects "AP Calculus AB" vs "AP Calculus BC".

**Transcripts — deterministic and genuinely good, and not a neural model.**
`pace.cli parse-transcript` read 8 courses from the sample, computed 7.5 credits
matching the printed total, produced GPA 4.133, flagged the two footer figures as
`unverified`, and attaches per-cell OCR verification when a backend is present.

## 4. FEATURE READINESS MATRIX

| Feature | Status | Evidence | Current quality | Blocker | Action |
|---|---|---|---|---|---|
| document classification | **MISSING** | doc-type artifact won't load; synthetic-only; bank statement → transcript 0.9975 | none on real docs | no trained real-doc model | UI asks the user |
| syllabus classification | **MISSING** | same artifact; 0.9423 on a real syllabus, no held-out number | unmeasured | same | UI asks the user |
| schedule classification | **MISSING** | synthetic only | none | same | defer |
| transcript classification | **PARTIAL** | `looks_like_transcript` rule; sample parsed 8/8 courses, credits match | good on clean text | rule only, no held-out set | keep, add tests |
| gradebook classification | **MISSING** | synthetic only | none | none | defer |
| course catalog classification | **MISSING** | synthetic only | none | none | defer |
| entity extraction (spans) | **PARTIAL** | synthetic tagger emits 58 spans on a syllabus incl. `STUDENT`, `WEIGHT`; uncalibrated | hallucinates | no real span-tagger | use rules |
| course title | **PARTIAL** | 9/12 real docs | 2 wrong ("Teaching Assistant: TBD", "Paul Kockelman, Department of Anthropology") | title heuristic | confirm in UI |
| course code | **PARTIAL** | 7/12 | missing when code is glued to the title line | rule | confirm |
| teacher | **PARTIAL** | 8/12 | 3 contaminated ("Dr. Perihan Saygin Fall 2018", email appended) | rule | confirm/strip |
| grade category | **PARTIAL** | 8/12 breakdowns | nested sub-weights misread ("participation 10%" of 20%) | rule | show sum warning |
| weight | **PARTIAL** | 8/12; 6 sum to exactly 100 | exam dates inside grading block; components dropped | rule | confirm + sum check |
| assessment | **PARTIAL** | categories only | no type/date/points | no schema | defer |
| date | **PARTIAL** | regex `extract_candidates` | found, never linked to an assessment | no relation | defer |
| credit | **PARTIAL** | 3/12 | only when printed as "3 credits"/"3.0" | rule | confirm |
| prerequisite extraction | **PARTIAL** | relation engine on catalog cells | 0.919 macro on 652 rows | needs supplied cell+id | catalog flow only |
| relation extraction | **PARTIAL** | 5 labels, catalog-cell scope | 0.919 macro; rare labels weak | no document-level relation model | defer |
| WEIGHT_OF | **READY** | `grading[] = {category, weight, evidence}` | 8/12 breakdowns, 6 exact | none in record | ship |
| DATE_OF | **MISSING** | no object links a date to an assessment | — | no relation model | defer |
| PREREQUISITE_OF | **PARTIAL** | live demo above; 0.996 conf | works when given the cell | input contract | catalog flow only |
| entity resolution | **MISSING** | no matcher in the tree | — | none | defer; never auto-merge |
| course matching | **MISSING** | `canonical_course_code` normalisation only | exact-string only | none | defer |
| confidence output | **READY** | every section and value carries confidence + `confidence_source` | — | none | ship |
| confidence calibration | **PARTIAL** | in-dist ECE 0.0016; unseen-university ECE 0.1665 (jev) / 0.256 (minilm) | overconfident OOD | distribution shift | never auto-accept |
| unknown / OTHER detection | **PARTIAL** | field menu has `other` (receipt 71.9%); "Late work loses 10% per day." → `course_description` 99.25% | unreliable | no real OOD training | UI asks type |
| OCR-noise robustness | **PARTIAL** | 400 rows, glyph swaps 0→O, 1→l, m→rn: jev 1.000→0.995 @30%, minilm 0.930→0.925 | strong on short strings | no page-image eval for this model | ingest layer separately tested |
| long-document support | **PARTIAL** | detector segments; model window 127/256; 400-word input silently cut to 127 tokens with same answer | whole syllabus as one input → `office_hours` 99.69% | no chunk aggregation | section-first only |
| Academic IR | **READY** | `pace.academic-ir/0.2` in `pace/api.py` | new this audit | no document type/relations | ship |
| provenance | **READY** | every value carries `source_text`, `start`, `end`, `grounded_in_source`; 72/73 grounded | page/bbox only in `ingest` | not carried into record | ship |
| stable inference API | **READY** | `understand_academic_document(text)` / `pace understand PATH` | new this audit | none | ship |
| mobile deployment readiness | **PARTIAL** | PyTorch 91 MB fp32; ONNX int8 exists only for other checkpoints (`field-classifier-mix`, `usc-miniLM`) | desktop CPU 3.6 ms/value | no export for the shipped checkpoint, no runtime test | export before shipping |

## 5. REAL END-TO-END TEST RESULTS

12 syllabi from Yale Summer Session and UF's published listings — the untouched
value-gold documents — through `understand_academic_document`:

| Read | Count | Note |
|---|---:|---|
| course title | 9 / 12 | 2 wrong, 1 absent |
| course code | 7 / 12 | |
| instructor | 8 / 12 | 3 carry term/email contamination |
| grading breakdown | 8 / 12 | |
| weights summing to exactly 100 | 6 / 12 | one is a genuine nested-sub-weight document |
| class meetings | **2 / 12** | weakest field |
| evidence spans grounded | 72 / 73 | 1 instructor span not found in its own slice |
| mean read time | 0.16 s / document | CPU, model warm |

One document in full (Yale PHYS S102E, 7,454 chars):

```
sections 7 | grading Exam1 20, Exam2 20, Exam3 20, participation 10 (sum 70, flagged)
meeting  MO WE 10:00-11:30 | course Hollywood Astrophysics | instructor Frank Robinson
warning  "grading weights sum to 70, not 100: ... nested sub-weights or a missed row"
```

The warning is correct: the document prints `20% Exam 1 = written (15%) + oral
(5%)`, and the reader drops the components rather than double-counting them.

**Non-academic input** (same reader, one text in):

| Input | Field menu | Document type |
|---|---|---|
| receipt | `other` 71.9% | transcript 0.9211, 9 entities |
| menu | `class_meeting_room` 84.0% | `other` 0.5521 |
| news article | `course_description` 99.6% | gradebook 0.3145 |
| bank statement | `office_hours` 96.2% | transcript 0.9975, 25 entities |
| "Late work loses 10% per day." | `course_description` 99.25% | — |

The field menu sometimes abstains into `other`; the document-type model never
does. That asymmetry is why the record exposes no document type.

## 6. REAL MODEL METRICS

* Section reader, in distribution: 7,571 rows, accuracy 0.9982, macro-F1 0.9711,
  weighted 0.9981, ECE-10 0.0016, per-class F1 0.571 (`exam_schedule`, 5 rows)
  to 1.000.
* Section reader, unseen universities (fold 0: MIT, Penn State, Ohio State,
  Berkeley, UCLA; 447 rows): jev v2 accuracy 0.7830, macro 0.5659, ECE 0.1665;
  jev v1 0.8322/0.6391; bge-small 0.8300/0.6165.
* Deployed MiniLM reader, its own test set: 16,896 rows, 0.9892 / 0.9334.
  On the 7,571-row set: 0.9464 / 0.8486.
* Relation encoder: 652 rows, macro-F1 0.9193, 4 errors, accuracy 0.9939.
* Thesis of the ablated runs: a 149 M encoder transferred no better than a 33 M
  one, so size is not the transfer constraint.

## 7. CURRENT INFERENCE CONTRACT

```python
from pace.api import understand_academic_document
record = understand_academic_document(text)          # CLI: python -m pace.cli understand file.txt
```

```json
{
  "schema_version": "pace.academic-ir/0.2",
  "document": {"characters": 7454, "sections_detected": 7,
               "document_type": null, "document_type_unavailable_because": "..."},
  "course": {"course_title": "...", "course_code": "...", "instructor": "..."},
  "grading": [{"category": "Exam 1", "weight": 20, "evidence": {"text": "...", "start": 2431, "end": 2499}}],
  "grading_total_percent": 70, "grading_total_consistent": false,
  "meetings": [{"days": ["MO","WE"], "start_time": "10:00", "end_time": "11:30", "room": null}],
  "policies": {"office_hours": {"value": "...", "confidence": 0.85, "evidence": {}}},
  "sections": [{"heading": "...", "label": "office_hours", "confidence": 0.9997, "band": "high"}],
  "evidence": [{"field": "grading", "value": "Exam 1 20%", "confidence": 0.9,
                "confidence_source": "rule", "band": null,
                "source_text": "...", "start": 2431, "end": 2499,
                "grounded_in_source": true, "requires_confirmation": true}],
  "confidence": {"sections_in_distribution_ece": 0.0016, "sections_unseen_university_ece": 0.1665, "...": "..."},
  "model": {"checkpoint": "artifacts/field-classifier-minilm", "parameters": 22729386,
            "max_state_tokens": 256, "temperature": 0.6467},
  "warnings": ["..."],
  "requires_confirmation": true
}
```

Bands apply to **section labels only** (they are model probabilities). Values are
read by rules, so they carry `confidence_source: "rule"` and no band; their
measured precision is 0.955 on 12 held-out documents.

## 8. PRE-UI BLOCKERS

1. **The app must ask the user what the document is.** No model can do it.
2. **Every value must be confirmed.** On unseen institutions 0.98 confidence
   measured 0.828 precision; there is no safe auto-accept threshold.
3. **Do not promise schedule import from a syllabus.** Meetings were found in
   2 of 12 documents.
4. **Do not merge courses across documents.** There is no entity resolution.

Everything else on the matrix is deferred, not blocking.

## 9. FIXES PERFORMED DURING THIS AUDIT

* `pace/api.py` — one entry point, one versioned record, evidence on every value,
  explicit refusal to claim a document type, measured calibration block,
  warnings for inconsistent weight sums.
* `pace.cli understand PATH` — the same record from the command line.
* `scripts/audit_model_layer.py` — the probe that produced every number here.
* `tests/test_api_schema.py` — 8 tests fixing the contract, evidence grounding,
  band semantics, the unclaimed document type and input refusal.
* Full suite re-run: **359 tests, exit 0**.

## 10. FIRST UI VERTICAL SLICE

**Syllabus → course identity + grading breakdown.**

Trust in the first UI, with confirmation:

* course title and code (9/12 and 7/12 — preselect, let the user fix)
* instructor (8/12 — strip trailing term/email before showing)
* grading categories and weights, with the sum displayed and the inconsistency
  warning surfaced (8/12, 6 exact)
* office hours / policy values (the strongest section class)
* source span for every one of them — the reader already proves it was not
  invented

Do not show in the first UI: class meetings, exam dates, prerequisites,
document type, any cross-document link.

## 11. DEFERRED FEATURES

Not blockers: schedule/gradebook/catalog classification, entity extraction from
free text, DATE_OF, entity resolution, pathway graphs, GPA and What-If engines,
mobile export, K-12/SIS corpus, quantization.

Distinguish carefully: **document classification and entity resolution are
deferred because the UI can replace them with a picker and a manual link.**
Relations are deferred because nothing in the first slice needs them. None of
these should stop the build; all of them must stay off the feature list.

## 12. FINAL GO / NO-GO

**GO TO UI — scoped to the syllabus import slice, with the user choosing the
document type and confirming every field.**

Not ready, and must not be advertised: automatic document classification,
schedule import, prerequisite graphs, cross-document course matching, automatic
acceptance of any extracted value.
