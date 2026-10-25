# PACEDECIDE — MODEL COMPLETION, RETRAINING, GENERALIZATION, AND HARDENING GOAL

> Saved verbatim on 2026-10-24 as the working charter for the model system.
> The audit in `artifacts/model-audit/REPORT.md` is the baseline, not a completion report.

The sole objective of this work is the **PaceDecide model system**.

Do not work on application UI.

Do not optimize around a demo.

Do not stop merely because one narrow vertical slice works.

Do not reinterpret incomplete functionality as acceptable because surrounding software could compensate for it.

The goal is to produce the strongest practical compact academic-document understanding model system possible from the current repository, training infrastructure, available data, compute, and time.

The existing repository has already undergone a detailed audit. Treat that audit as a baseline, not as a completion report.

The current implementation has meaningful strengths, but it is not yet the intended final model.

The agent is explicitly authorized to:

- retrain models
- replace existing checkpoints
- replace base architectures
- change tokenization
- alter task formulations
- build new datasets
- obtain additional legitimate public datasets
- add targeted synthetic data
- redesign heads
- split one model into several specialized models
- merge specialized models into a shared encoder
- build better confidence calibration
- add entity-resolution training
- add document classification
- add relation extraction
- improve long-document processing
- replace weak rule/model boundaries
- benchmark entirely different approaches
- remove components that are not helping

Do not preserve an existing architecture merely because work has already been invested in it.

Evidence determines the architecture.

The desired result is not literally a mathematically "perfect" model; no real model is perfect.

The intended meaning of "do not leave this as an imperfect model" is:

> Do not knowingly stop at obvious, material weaknesses that can reasonably be improved through better data, task design, training, architecture, calibration, or evaluation.

The final stopping point should be justified by empirical evidence showing that remaining errors are difficult, rare, understandable, and no longer caused by obvious missing capabilities or weak training methodology.

============================================================
1. CURRENT BASELINE — DO NOT LOSE THIS CONTEXT
============================================================

The current audit established roughly the following state.

There are currently several field-classification models.

The strongest measured field model is approximately:

jev-field-v2
- approximately 15.3M parameters
- ELECTRA-like architecture
- 127 text-token state window
- 42 label probabilities
- trained on approximately 200,985 rows
- same-row test:
  - accuracy approximately 0.9982
  - macro-F1 approximately 0.9711
  - ECE approximately 0.0016

However, unseen-university generalization is substantially worse.

On one held-out-university fold:

- accuracy approximately 0.783
- macro-F1 approximately 0.566
- ECE approximately 0.167

This distribution-shift failure is one of the most important current weaknesses.

The deployed MiniLM reader also performs substantially worse when evaluated outside the corpus it was trained on.

The current system additionally has:

- deterministic syllabus extraction
- approximately 0.955 precision on a very small held-out 12-document extraction set
- provenance/evidence support
- an Academic IR
- one narrow prerequisite/corequisite relation model
- transcript parsing that is primarily deterministic
- no usable general document classifier
- no robust entity-resolution system
- no general document-level relation system
- no reliable unknown/out-of-domain rejection
- incomplete long-document handling
- no mature mobile deployment artifact for the selected final model

The current document classifier is especially unacceptable:

- synthetic-only training
- extreme synthetic test accuracy that does not transfer
- artifact compatibility problems
- severe out-of-distribution false positives

Examples observed during the audit included non-academic documents being classified as transcripts with extremely high confidence.

The current system also lacks:

- AP Chem ↔ AP Chemistry entity matching
- AP Calculus AB versus AP Calculus BC discrimination in a dedicated matcher
- DATE_OF relationships
- general assessment/date linking
- reliable schedule understanding
- general free-text entity extraction
- safe high-confidence auto-accept thresholds on unseen institutions

These are not theoretical future features.

They are core weaknesses to investigate and improve.

============================================================
2. PRIMARY GOAL
============================================================

Build a compact local academic semantic model system that can reliably transform academic document text into structured understanding.

The core model system should ultimately be capable of performing, either through one model or a small set of tightly integrated compact models:

1. document classification
2. section classification
3. entity extraction
4. relation extraction
5. entity resolution
6. uncertainty estimation
7. out-of-domain rejection
8. OCR-noise robustness
9. long-document processing
10. structured academic output

The model system should understand academic meaning rather than memorize templates.

============================================================
3. DO NOT CONFUSE "ONE MODEL" WITH "ONE CHECKPOINT"
============================================================

"PaceDecide" refers to the complete compact model system.

It does NOT require every task to live inside one neural checkpoint.

The final design may be:

- one multi-task encoder
- one encoder with several heads
- one shared encoder plus specialized heads
- several small specialized models
- a router plus task-specific compact models
- neural models combined with deterministic extraction where deterministic methods are objectively superior

For example:

Document classifier
        +
Academic field model
        +
Relation model
        +
Entity-resolution model

may be better than forcing everything into one encoder.

Conversely, a shared multi-task model may improve representation transfer.

Benchmark both philosophies where reasonable.

Optimize for:

- quality
- generalization
- model size
- latency
- maintainability
- mobile feasibility

not conceptual purity.

============================================================
4. FIRST PRIORITY: FIX DISTRIBUTION SHIFT
============================================================

The largest warning in the audit is the enormous difference between in-distribution performance and unseen-institution performance.

A model that scores 99% on familiar distributions but approximately 78% on unseen universities is not sufficiently general.

Investigate why.

Possible causes include:

- lexical/template memorization
- source-specific formatting
- narrow institution coverage
- synthetic artifacts
- class imbalance
- duplicated or near-duplicated data
- vocabulary leakage
- heading memorization
- insufficient semantic variety
- incorrect train/test structure
- base-model limitations
- short context
- task formulation
- calibration failure

Perform proper diagnostics.

Do not simply add more epochs.

Create institution-level and template-level cross-validation where possible.

Evaluation should intentionally ask:

> Can the system understand a school or university whose formatting and vocabulary it has never seen?

This is a major success criterion.

============================================================
5. SEARCH FOR MORE REAL DATA
============================================================

Before generating massive synthetic corpora, search aggressively for existing real academic datasets.

Search:

Hugging Face
Kaggle
GitHub
university repositories
public-school websites
state education websites
open research corpora
academic NLP datasets

Look for:

- syllabus datasets
- course syllabi
- course catalog datasets
- prerequisite datasets
- class schedule datasets
- transcripts/sample transcripts
- gradebook-like data
- report cards/sample report cards
- graduation requirement documents
- degree requirements
- curriculum guides
- educational document understanding datasets
- academic table datasets

Previously identified examples include datasets such as:

- public syllabus corpora on Hugging Face
- USC course-catalog data
- academic document benchmark datasets

Do not assume these are the best available options.

Search again.

Record:

- source
- license
- size
- schema
- academic domain
- institution diversity
- document diversity
- usefulness
- final usage decision

Prefer diverse real academic language whenever legally usable.

============================================================
6. DATASET DIVERSITY MATTERS MORE THAN RAW ROW COUNT
============================================================

Do not celebrate "200,000 training rows" if they come from narrow or repetitive distributions.

Measure diversity.

Report:

- number of institutions
- number of document families
- number of unique templates
- number of distinct courses
- vocabulary diversity
- source diversity
- real versus synthetic ratio
- average and percentile sequence lengths
- class distribution
- duplicate rate

Investigate near-duplicate leakage.

Use semantic or hashing approaches to detect duplicated content where appropriate.

============================================================
7. USE SYNTHETIC DATA SURGICALLY
============================================================

Synthetic generation should now be driven by measured failures.

Examples:

If grading categories fail:
generate difficult grading-policy variants.

If DATE_OF fails:
generate assessment/date ambiguity.

If entity resolution overmerges:
generate hard negative course pairs.

If OCR errors hurt:
generate realistic OCR corruption.

If document classification confuses bank statements with transcripts:
generate hard negative structured non-academic documents.

Do NOT generate millions of easy synthetic examples just to increase dataset size.

The synthetic generator should attack known blind spots.

============================================================
8. DOCUMENT CLASSIFICATION MUST BE REBUILT
============================================================

The current synthetic document classifier is not acceptable.

Build a real document-classification solution.

Target classes may include:

- syllabus
- schedule
- transcript
- gradebook
- report card
- course catalog
- graduation requirements
- assignment sheet
- course-selection document
- other / unsupported

This taxonomy may be refined.

Critical requirement:

OTHER must be meaningful.

Train with real negative documents.

Potential negatives:

- receipts
- invoices
- menus
- bank statements
- résumés
- articles
- medical forms
- legal forms
- business documents
- random web pages
- generic tables
- unrelated PDFs

The classifier should prefer abstention/OTHER over confidently hallucinating an academic document type.

Evaluate:

- macro-F1
- per-class precision/recall
- AUROC where useful
- OOD detection
- false academic classification rate
- calibration
- high-confidence error rate

============================================================
9. GENERAL ENTITY EXTRACTION
============================================================

Build or substantially improve general entity extraction.

The desired semantic vocabulary likely includes:

COURSE
COURSE_CODE
TEACHER
PERIOD
ROOM
TERM
SEMESTER
SCHOOL_YEAR
GRADE
LETTER_GRADE
NUMERIC_GRADE
GRADE_CATEGORY
WEIGHT
ASSESSMENT
ASSIGNMENT
EXAM
QUIZ
PROJECT
LAB
DATE
TIME
CREDIT
REQUIREMENT
PREREQUISITE
COREQUISITE
GRADING_SCALE

Do not blindly keep every label if it hurts quality.

Some labels may be hierarchical.

For example:

ASSESSMENT
  ├── EXAM
  ├── QUIZ
  ├── PROJECT
  └── LAB

Research whether hierarchical classification improves data efficiency.

Compare:

- BIO token tagging
- span classification
- candidate extraction + classification
- QA-style extraction
- hybrid rule candidate generation + neural semantic classification

The current pattern-heavy extraction system is useful but should no longer be the unquestioned endpoint.

If deterministic extraction remains better for certain fields, keep it.

============================================================
10. DO NOT REPLACE GOOD RULES JUST TO SAY "AI"
============================================================

Some fields are naturally deterministic.

Examples:

- percentages
- times
- dates
- credits
- course codes

The semantic model may answer:

"Is this percentage a grade-category weight?"

while deterministic code extracts:

40%

This is desirable.

The model should solve semantic ambiguity.

Rules should solve mechanical parsing when rules are more reliable.

The final system should be hybrid where justified.

============================================================
11. BUILD REAL RELATION EXTRACTION
============================================================

The current relation model is too narrow.

Expand relation understanding.

Important relations include:

WEIGHT_OF
DATE_OF
GRADE_OF
BELONGS_TO
TAUGHT_BY
CREDIT_OF
PREREQUISITE_OF
COREQUISITE_OF
SATISFIES
SAME_AS

Examples:

Tests → WEIGHT_OF → 40%

Final Examination → DATE_OF → December 16

AP Chemistry → TAUGHT_BY → Dr. Johnson

Precalculus → PREREQUISITE_OF → AP Calculus BC

Course → CREDIT_OF → 1.0

Assessment → GRADE_OF → 94%

Do not assume every relation requires a neural model.

Benchmark:

- contextual neural relation classification
- geometry/layout features
- deterministic table pairing
- distance heuristics
- hybrid systems

Measure relation F1 per class.

Rare relation classes must not disappear inside a high overall accuracy number.

============================================================
12. BUILD ENTITY RESOLUTION AS A FIRST-CLASS CAPABILITY
============================================================

This is currently entirely missing and must be developed seriously.

PaceDecide should understand that:

AP Chem
AP Chemistry
Chemistry AP
Advanced Placement Chemistry

may be the same course.

It should also understand that:

AP Calculus AB
AP Calculus BC

are NOT the same course.

Likewise:

Biology
AP Biology

may or may not be equivalent depending on context.

Build a dedicated pairwise or retrieval-based entity-resolution dataset.

Positive examples:

- abbreviation variants
- punctuation variants
- reordered names
- OCR-corrupted names
- course-code aliases
- teacher initials
- common institutional shorthand

Hard negatives:

- honors versus AP
- AB versus BC
- I versus II
- regular versus advanced
- similar teacher surnames
- same course title across different terms where identity should remain separate

Use context:

- course code
- instructor
- period
- semester
- credit
- school
- nearby text

Metrics should include:

- precision
- recall
- F1
- false merge rate
- false split rate

False merges are especially dangerous.

Optimize accordingly.

============================================================
13. CONFIDENCE MUST GENERALIZE
============================================================

Current confidence is excellent in-distribution and substantially worse out-of-distribution.

Fix this.

Investigate:

- temperature scaling
- class-specific calibration
- vector scaling
- isotonic calibration
- calibration on institution-held-out sets
- confidence based on ensembles
- uncertainty from agreement/disagreement
- OOD scoring
- energy-based confidence
- embedding-distance confidence
- conformal approaches if practical

The final confidence score should mean something useful on unseen institutions.

Measure:

- ECE
- Brier score
- reliability curves
- precision at threshold
- coverage at threshold

The most important operational statistic is:

> When the system claims high confidence, how often is it actually correct on unseen real academic documents?

============================================================
14. OUT-OF-DISTRIBUTION REJECTION
============================================================

Treat OOD as a core model task.

The model should not be forced to choose a known academic label when evidence is weak.

Test:

- restaurant menus
- receipts
- invoices
- bank statements
- résumés
- articles
- medical documents
- unrelated course-adjacent web pages
- generic tables

Measure:

- false positive rate
- confidence on OOD
- OTHER precision
- OTHER recall
- maximum-confidence distribution

If necessary, build a separate lightweight academic/non-academic gate before detailed classification.

============================================================
15. OCR ROBUSTNESS
============================================================

Continue strengthening OCR robustness.

Train/evaluate examples such as:

Chemistry → Chem1stry
Johnson → J0hnson
40% → 4O%
December → Decernber
AP Calculus → AP CaIculus

Also:

- missing spaces
- repeated OCR lines
- merged columns
- broken rows
- missing punctuation
- shuffled text order
- duplicated headers
- character deletion
- corrupted course codes

Use actual OCR outputs when available.

Synthetic character substitution alone is not enough.

============================================================
16. CONTEXT LENGTH AND LONG DOCUMENTS
============================================================

The current 127-token model is surprisingly strong locally, but the system should investigate whether this context window is unnecessarily restrictive.

Do not assume the answer is simply "use 512."

Benchmark reasonable sequence lengths.

Examples:

128
256
384
512
possibly longer if a suitable compact encoder supports it efficiently

Measure:

- field quality
- relation quality
- latency
- memory
- truncation rate
- cross-sentence understanding

If longer context materially improves semantic understanding, use it.

If not, keep the smaller window.

============================================================
17. LONG DOCUMENT UNDERSTANDING
============================================================

Build a robust long-document strategy.

The solution does not need to feed an entire catalog through one transformer.

Potential pipeline:

document
  ↓
layout / sections
  ↓
semantic chunks
  ↓
local inference
  ↓
structured state
  ↓
cross-chunk reconciliation

Prefer natural section boundaries over fixed token chopping.

Use:

- headings
- tables
- course-entry boundaries
- transcript rows
- paragraphs
- grading sections

Use overlapping windows only where needed.

Test on genuinely long course catalogs.

============================================================
18. STRUCTURED MEMORY
============================================================

Once facts are confidently extracted, downstream processing should not repeatedly reread already-understood raw text.

Example:

KNOWN COURSE: AP Chemistry
KNOWN TEACHER: Allison Johnson
CURRENT SECTION: Final Examination

This can provide relevant global context cheaply.

Evaluate whether structured context improves subsequent decisions.

============================================================
19. MODEL ARCHITECTURE SEARCH
============================================================

Do not assume the current ELECTRA-like model is optimal.

Research and benchmark current compact encoders.

Candidates may include:

- ELECTRA variants
- MiniLM variants
- ModernBERT compact variants
- DeBERTa small/xsmall
- MobileBERT
- TinyBERT
- DistilBERT
- BGE small encoders
- E5 small encoders
- compact embedding models
- newer efficient encoder architectures

The previous audit already suggests that simply making the encoder much larger does not guarantee transfer improvement.

One recorded result indicated that a roughly 149M encoder did not transfer better than a roughly 33M encoder.

Therefore focus heavily on:

- data diversity
- task design
- regularization
- domain generalization
- calibration

not only parameter count.

============================================================
20. TRY MULTI-TASK LEARNING SERIOUSLY
============================================================

Test whether shared academic representations help.

Possible architecture:

Shared encoder
    │
    ├── document head
    ├── section head
    ├── entity head
    ├── relation head
    └── matching head

Potential advantages:

- common academic semantics
- shared representations
- smaller total deployment footprint
- transfer across scarce labels

Potential disadvantages:

- negative task interference
- difficult balancing
- weaker specialist performance

Benchmark rather than assume.

If specialist models win, use specialist models.

============================================================
21. HARD NEGATIVE MINING
============================================================

Make hard-negative mining a first-class training process.

Examples:

Tests 40%
versus
Test 4: 40/50

Final Exam Review — Dec 12
versus
Final Examination — Dec 16

AP Calculus AB
versus
AP Calculus BC

Biology
versus
AP Biology

Period 4
versus
Chapter 4

Wrong confident predictions should automatically become candidates for future hard-negative datasets.

============================================================
22. FAILURE-DRIVEN TRAINING LOOP
============================================================

After each serious run:

1. evaluate
2. inspect failures
3. categorize failures
4. generate/find targeted examples
5. retrain
6. reevaluate

Do not merely train longer.

Fix the cause.

============================================================
23. REAL-WORLD EVALUATION MUST EXPAND
============================================================

The current 12-document held-out syllabus set is useful but far too small to justify broad claims.

Expand real evaluation substantially.

Gather diverse public documents from many institutions.

Prefer a test set containing:

- high schools
- universities
- community colleges
- school districts
- multiple states
- different document styles
- PDFs
- OCR'd images
- tables
- screenshots where legally obtainable

Keep final test documents isolated from training decisions.

============================================================
24. INSTITUTION-HOLDOUT EVALUATION
============================================================

This should become one of the project's central benchmarks.

For example:

Train:
institutions A–Z except five

Test:
five completely unseen institutions

Rotate folds where practical.

Measure:

- document classification
- section classification
- entity extraction
- relation extraction
- calibration
- OOD rejection

This metric matters more than random-row splits for this project.

============================================================
25. TEMPLATE-HOLDOUT EVALUATION
============================================================

Synthetic and real template families should not leak across splits.

If nearly identical syllabus templates appear in train and test, results are misleading.

Group by template/source/institution before splitting.

Document the strategy.

============================================================
26. BASELINES MUST REMAIN
============================================================

Keep strong simple baselines.

Examples:

- TF-IDF
- regex
- fuzzy string matching
- sentence embedding similarity
- deterministic table matching

The final ML system should outperform them where ML is used.
If a baseline remains superior for one task, keep the baseline.

============================================================
27. ACADEMIC INTERMEDIATE REPRESENTATION
============================================================

The existing Academic IR is useful. Continue improving it as the neutral output contract.

The model system should eventually produce structured information resembling:

{
  document_type,
  sections,
  entities,
  relations,
  evidence,
  confidence,
  warnings
}

Every extracted claim should retain evidence where possible.

Do not allow model-specific implementation details to leak into the semantic schema unnecessarily.

============================================================
28. PROVENANCE
============================================================

Retain and improve provenance.

A fact should ideally know:

- original source text
- character span
- page
- bounding box
- document identifier
- OCR confidence
- model confidence
- extraction method
- relationship evidence

This is valuable both scientifically and for debugging.

============================================================
29. TRAINING INFRASTRUCTURE
============================================================

Preserve reproducibility.

Every run should capture:

- git commit
- dataset version
- config
- random seed
- base checkpoint
- tokenizer version
- preprocessing version
- task weights
- augmentation settings
- hardware
- library versions
- metrics

Avoid untracked notebook-only experiments.

============================================================
30. EXPERIMENT TRACKING
============================================================

Maintain a structured experiment table.

At minimum:

run_id
architecture
parameters
context length
dataset
tasks
accuracy
macro-F1
OOD metrics
institution-holdout metrics
ECE
model size
latency
notes

This should make architectural decisions evidence-based.

============================================================
31. TOKENIZATION
============================================================

Review whether the current tokenizer is appropriate for academic text.

Academic documents contain:

- course codes
- abbreviations
- percentages
- grade symbols
- semesters
- department codes
- room codes
- unusual capitalization

Inspect fragmentation.

Compare tokenizers if necessary.

Do not rebuild a tokenizer without evidence that it helps.

============================================================
32. CLASS IMBALANCE
============================================================

The existing prerequisite relation dataset contains severe rare-class imbalance.

Analyze all task distributions.

Potential techniques:

- weighted loss
- focal loss
- oversampling
- targeted data collection
- targeted generation

Macro-F1 and per-class metrics matter.

Do not allow majority classes to conceal weak rare relations.

============================================================
33. LABEL QUALITY
============================================================

Investigate noisy or ambiguous labels.

A model cannot outperform broken ground truth indefinitely.

Create tooling for:

- random label audits
- disagreement inspection
- impossible examples
- duplicate-label inconsistencies
- ambiguous cases

Correct training data where justified.

============================================================
34. MODEL SIZE IS A CONSTRAINT, NOT THE OBJECTIVE
============================================================

The target remains a compact locally deployable model system.

However, do not sacrifice major quality to save a small amount of storage.

A 15M model is excellent if sufficient.

A 30M model is fine.

A 60M model may also be acceptable.

Even somewhat larger models may be justified if they produce a large quality improvement and remain practical on modern phones.

Measure instead of guessing.

============================================================
35. MOBILE DEPLOYABILITY
============================================================

Training should remain framework-first, likely PyTorch unless evidence supports another framework.

Do NOT constrain the architecture around Core ML.

Once strong checkpoints exist, evaluate practical mobile runtimes such as:

- ExecuTorch
- ONNX Runtime Mobile
- another current suitable runtime

Deployment should be selected after model quality is established.

But do periodically verify that architectural choices are not impossible to export.

============================================================
36. QUANTIZATION
============================================================

After model quality stabilizes, benchmark:

- FP16
- INT8
- lower precision only if supported and worthwhile

Measure:

- metric changes
- calibration changes
- model size
- latency
- memory

Do not assume quantization preserves confidence behavior.

Recalibrate if required.

============================================================
37. PERFORMANCE
============================================================

Eventually report:

- model load time
- warm latency
- tokenization time
- batch latency
- memory
- throughput
- long-document throughput
- model size

Desktop measurements must be labeled as desktop.

Mobile measurements must actually come from mobile or representative runtime tests.

============================================================
38. DO NOT OVERFIT TO CURRENT SYLLABUS EXTRACTION
============================================================

The current syllabus pipeline is useful, but do not let it constrain the final model.

The long-term model system should generalize across academic documents.

Syllabus understanding can remain the strongest initial benchmark, but improvement should expand toward:

- schedules
- transcripts
- gradebooks
- report cards
- catalogs
- requirements

Incrementally add document families when sufficient real training data exists.

============================================================
39. TRANSCRIPTS
============================================================

The current deterministic transcript parser appears promising.

Do not replace it blindly.

Instead evaluate whether neural semantics improve:

- course-row detection
- ambiguous columns
- term boundaries
- course level
- repeated courses
- transfer credits
- unusual formatting

Use ML only where it improves the deterministic parser.

============================================================
40. COURSE CATALOGS
============================================================

Course catalogs are especially important for:

- course identification
- credit extraction
- prerequisites
- corequisites
- recommendations
- restrictions

Use real catalog datasets extensively.

The existing USC relation model is a useful baseline.

Expand to multiple institutions.

Avoid learning USC-specific phrasing.

============================================================
41. SCHEDULES
============================================================

Build actual schedule understanding if enough data can be obtained.

Extract:

- course
- teacher
- period
- room
- days
- start/end time
- semester/term

Schedules often contain strong layout structure.

Consider whether table/layout reconstruction should precede semantic inference.

============================================================
42. GRADEBOOKS
============================================================

Research whether usable public or synthetic gradebook data can support:

- course name
- assignment
- category
- score
- points possible
- due date
- missing/excused status

Do not promise gradebook understanding without real evaluation.

============================================================
43. LAYOUT
============================================================

Investigate lightweight layout features before using large document-vision architectures.

Potential features:

- line index
- row/column membership
- normalized bounding box
- page index
- heading state
- table-cell adjacency
- relative spatial distance

Benchmark whether these materially improve relation/entity extraction.

============================================================
44. VISION SHOULD REMAIN OPTIONAL TO THE SEMANTIC CORE
============================================================

The central model can primarily consume OCR-derived structure.

Do not unnecessarily turn PaceDecide into a giant vision-language model.

If lightweight visual/layout information helps substantially, use it.

Keep semantic inference compact.

============================================================
45. HARD TEST SUITE
============================================================

Create an adversarial permanent benchmark containing cases such as:

- nested grading weights
- multiple dates
- misleading percentages
- repeated course names
- AB versus BC
- Honors versus AP
- OCR corruption
- broken tables
- unusual headings
- unrelated documents
- multi-course pages
- cross-page sections

This suite should grow from real failures.

============================================================
46. AUTOMATED REGRESSION GATES
============================================================

Do not accept a new checkpoint because one metric improves.

Establish regression gates.

Example concept:

A new model cannot become the preferred checkpoint if:

- institution-holdout macro-F1 drops materially
- OOD false-positive rate worsens
- entity-resolution false merges increase
- calibration worsens substantially
- real syllabus extraction regresses

Use a balanced model-selection score or Pareto reasoning.

============================================================
47. MODEL SELECTION SHOULD WEIGHT REAL GENERALIZATION
============================================================

Do not select the "best" model using random-row validation accuracy.

Priority should roughly favor:

1. real unseen-institution performance
2. hard-case performance
3. OOD behavior
4. calibration
5. relation/entity quality
6. normal in-distribution quality
7. size/latency

A 99.8% familiar-distribution model that fails unseen schools is not superior to a 98% model that transfers reliably.

============================================================
48. POTENTIAL ENSEMBLES
============================================================

If two small models have complementary strengths, test lightweight ensembling.

For example:

small semantic model
+
deterministic parser
+
OOD detector

or:

two compact encoders whose agreement provides confidence

Do not create an unnecessarily expensive ensemble.

Use only if measurable gains justify it.

============================================================
49. TEACHER DISTILLATION
============================================================

Consider teacher-model assistance where useful.

Possible use:

- labeling difficult public documents
- generating paraphrases
- detecting mislabeled training data
- generating hard negatives
- producing soft targets
- distillation

But keep ground truth grounded.

Do not allow a teacher model to hallucinate labels into the training set unchecked.

============================================================
50. ACTIVE LEARNING
============================================================

If useful, implement an active-learning loop.

Identify examples where:

- confidence is low
- models disagree
- predictions are unstable
- a deterministic parser disagrees with neural semantics

Send those examples for human/teacher review and targeted training.

============================================================
51. STOPPING CRITERIA
============================================================

Do not stop because:

- training finished
- one benchmark is high
- a checkpoint exists
- inference works
- the repository has many tests

Stop only when evidence supports that the major architectural weaknesses have been addressed.

A strong stopping point should include:

A. DOCUMENT CLASSIFICATION
Reliable on real held-out institutions and real negative documents.

B. SECTION UNDERSTANDING
Strong institution-held-out performance.

C. ENTITY EXTRACTION
Reliable for core academic entities.

D. RELATIONS
Core relations work beyond one narrow catalog dataset.

E. ENTITY RESOLUTION
Implemented and benchmarked with difficult positives/negatives.

F. OOD
Non-academic documents are reliably rejected.

G. CONFIDENCE
Confidence is meaningfully calibrated on held-out institutions.

H. OCR
Reasonable robustness demonstrated.

I. LONG DOCUMENTS
A real strategy exists and works on large documents.

J. STRUCTURED OUTPUT
Academic IR remains stable and grounded.

K. DEPLOYABILITY
The selected model architecture is realistically deployable on modern mobile hardware.

============================================================
52. IDEAL PERFORMANCE TARGETS
============================================================

Do not treat these as rigid requirements, but use them as ambitions.

For critical core tasks on diverse real held-out data, aim toward:

document classification:
>95% accuracy where feasible

core entity classes:
>95% precision with strong recall

core relation extraction:
>90% macro-F1

entity resolution:
very high precision, especially low false-merge rate

high-confidence predictions:
approximately 99% precision where realistic

OOD academic/non-academic gate:
extremely low false academic rate

institution-holdout performance:
as close as possible to normal IID performance

The key target is reducing the generalization gap.

A slightly lower IID score is acceptable if unseen-institution performance improves significantly.

============================================================
53. CURRENT MODEL SWAP QUESTION
============================================================

The audit found jev-field-v2 promising:

- smaller than the currently deployed MiniLM reader
- faster load time
- excellent same-row metrics
- similar extraction counts on real syllabi

But it has not been fully re-measured field-by-field for the production extraction task.

Perform that comparison properly.

If jev-field-v2 wins on:

- real syllabus extraction
- institution holdout
- calibration
- latency
- size

promote it.

If not, do not.

Do not preserve MiniLM simply because it is currently wired into the reader.

============================================================
54. REVISIT THE 127-TOKEN LIMIT
============================================================

The current Jev-style model uses approximately 127 text tokens.

Determine whether that is genuinely optimal.

Train controlled variants if appropriate.

For example:

127
256
384
512

Compare on identical datasets.

This experiment is important because many previous assumptions about long documents were architectural rather than empirically established.

============================================================
55. FINAL SYSTEM SHOULD ANSWER TYPED QUESTIONS
============================================================

Preserve the desirable Jev-like philosophy.

The semantic model should primarily make bounded decisions.

Examples:

DOCUMENT_TYPE?
→ syllabus 0.98

ENTITY_TYPE("AP Chemistry")?
→ course 0.997

RELATION("Tests", "40%")?
→ weight_of 0.991

SAME_COURSE("AP Chem", "AP Chemistry")?
→ yes 0.984

IS_ACADEMIC_DOCUMENT?
→ yes 0.999

This is preferable to generating ungrounded prose.

============================================================
56. NO HALLUCINATED ACADEMIC FACTS
============================================================

The model should be rewarded for:

UNKNOWN
OTHER
NONE
LOW CONFIDENCE

when appropriate.

It should never be forced to invent a course, date, grade, teacher, or prerequisite.

============================================================
57. DOCUMENT ALL REAL IMPROVEMENTS
============================================================

Maintain a progress report showing the baseline audit versus the final system.

Example:

                         OLD          NEW
Unseen institution F1   0.566        ...
Document classifier     unusable     ...
Entity resolution       missing      ...
OOD rejection           poor         ...
DATE_OF                  missing      ...
Relation macro-F1        narrow .919  ...
Calibration ECE          .1665        ...
Context                  127          ...
Model size               ...          ...

Every claim should come from actual evaluation.

============================================================
58. DO NOT DESTROY WORKING CAPABILITIES
============================================================

Keep the current repository usable throughout iteration.

Create new checkpoints/configs rather than overwriting the only working model.

Preserve:

- working extraction
- Academic IR
- provenance
- tests
- audit scripts

Replace defaults only after a new model passes regression gates.

============================================================
59. TEST SUITE
============================================================

Expand the existing 359-test suite as new capabilities are added.

Add tests for:

- document classifier
- OOD rejection
- entity resolution
- relation output
- calibration loading
- long input
- hard negatives
- new Academic IR relationships
- checkpoint loading
- tokenizer parity
- quantized parity when applicable

============================================================
60. FINAL MODEL AUDIT
============================================================

When the agent believes PaceDecide is complete, run another audit at least as strict as the current one.

It must test:

- real unseen academic documents
- multiple institutions
- non-academic documents
- OCR corruption
- difficult entity aliases
- difficult relation cases
- long documents
- calibration
- actual selected checkpoint
- actual inference pipeline

Do not evaluate unused experimental checkpoints and call those production metrics.

============================================================
61. FINAL DELIVERABLES
============================================================

At completion, the repository should contain:

- final selected model architecture
- final checkpoint(s)
- tokenizer(s)
- training configs
- complete dataset manifests
- real dataset inventory
- synthetic-data inventory
- train/validation/test split definitions
- institution-holdout benchmarks
- document classifier
- entity extractor
- relation extractor
- entity resolver
- OOD handling
- confidence calibration
- OCR robustness evaluation
- long-document pipeline
- Academic IR
- provenance
- stable inference API
- model export path
- benchmarks
- regression tests
- failure-analysis tooling
- model card
- research documentation
- licenses
- final audit report

============================================================
62. FINAL REPORT
============================================================

The final report must provide:

1. FINAL ARCHITECTURE

2. WHY IT WAS SELECTED

3. MODELS TESTED

4. DATASETS USED

5. DATASET DIVERSITY

6. SYNTHETIC DATA USED

7. DOCUMENT CLASSIFICATION METRICS

8. ENTITY METRICS

9. RELATION METRICS

10. ENTITY-RESOLUTION METRICS

11. OOD METRICS

12. OCR ROBUSTNESS

13. IID PERFORMANCE

14. UNSEEN-INSTITUTION PERFORMANCE

15. CALIBRATION

16. HIGH-CONFIDENCE PRECISION/COVERAGE

17. LONG-DOCUMENT PERFORMANCE

18. PARAMETER COUNT

19. MODEL SIZE

20. LATENCY

21. MOBILE-DEPLOYMENT STATUS

22. QUANTIZATION RESULTS, IF PERFORMED

23. FAILURE ANALYSIS

24. REMAINING LIMITATIONS

25. EXACT COMPARISON AGAINST THE PREVIOUS AUDIT

============================================================
63. MOST IMPORTANT PRINCIPLE
============================================================

Do not optimize for making the existing audit look good.

Attack the weaknesses the audit exposed.

The existing system already proves that compact academic classification can work extremely well inside a familiar distribution.

The next challenge is much harder:

> Make it work on academic documents it has never seen before.

A genuinely strong PaceDecide system should not depend on recognizing the wording and formatting of a handful of known institutions.

It should learn the concepts:

course
teacher
grading weight
exam
date
credit
prerequisite
requirement
schedule
transcript
syllabus

independently of how one school chooses to write them.

============================================================
64. FINAL INSTRUCTION
============================================================

Continue iterating until the system is no longer obviously incomplete.

Do not settle for:

"the UI can ask the user."

Do not settle for:

"the rule-based parser handles it."

Do not settle for:

"the synthetic benchmark is 99%."

Do not settle for:

"the current model is good enough for a demonstration."

Those may be useful temporary fallbacks, but they are not the goal of this agent.

The agent's sole responsibility is the model system.

Research.

Collect data.

Train.

Evaluate.

Break it.

Inspect the failures.

Improve the data.

Retrain.

Try better architectures.

Measure transfer.

Calibrate confidence.

Build missing semantic capabilities.

Repeat.

The desired endpoint is a compact academic semantic system that is difficult to break with ordinary real-world academic documents and whose remaining limitations are demonstrated empirically rather than merely assumed.

The system does not need to be literally flawless.

It does need to be something its developer can reasonably defend as:

**the strongest model they could build after systematically identifying and eliminating every major measurable weakness.**
