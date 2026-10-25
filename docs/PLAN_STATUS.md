# Plan status, section by section

Status of every numbered section of [MODEL_COMPLETION_PLAN.md](MODEL_COMPLETION_PLAN.md).
Legend: **DONE** = measured and written down, **PARTIAL** = measured with a known
gap, **BLOCKED** = attempted, blocked by a named cause, **OPEN** = not started.

| § | requirement | status | evidence |
|---:|---|---|---|
| 1 | baseline recorded, not treated as completion | DONE | `artifacts/model-audit/REPORT.md`; v2 numbers re-measured in `artifacts/transfer-eval/v2-mix.json` |
| 2 | primary goal: compact academic semantic system | PARTIAL | field reader done; relations/taxonomy still short (see §11, §33) |
| 3 | one system ≠ one checkpoint | DONE | v2 primary reader + v3 district specialist + `pace/doc_type.py` gate + `pace/field_shapes.py` |
| 4 | fix distribution shift | DONE (diagnosed) | school-holdout = 0.388 macro-F1 vs 0.400 family-holdout: gap is document family, not school (`TRANSFER.md`) |
| 5 | search more real data | PARTIAL | Infinite Campus/OneRoster corpus added (3,296 cells from real catalog values); no new public syllabus corpus found with usable licence |
| 6 | diversity over row count | DONE | `pace.transfer.diversity()`; syllabus natural rows = 1,638 of 200 k — the binding constraint |
| 7 | synthetic data only against measured failures | DONE | shapes built for measured probe failures (`scripts/build_pattern_rows.py`) and rejected when they cost -6 F1 (`jv7-field`) |
| 8 | document classification rebuilt | DONE | 302 test rows, 0.9967 acc / 0.9883 macro-F1, `other` precision+recall 1.0 on 239 real negatives (`artifacts/doc-type/metrics.json`) |
| 9 | general entity extraction | PARTIAL | 42 described fields; **value-only (header hidden) accuracy 0.9473 / macro-F1 0.8172** on 7,571 rows for v4 (v2: 0.8991 / 0.7000); `work` family still weak on unseen syllabus prose |
| 10 | keep good rules | DONE | `pace/field_shapes.py` (15/15 shapes, 0/10 ambiguous fired), SIS header table, extractor untouched |
| 11 | real relation extraction | PARTIAL | catalog prerequisite/corequisite model exists (`artifacts/usc-relations`); WEIGHT_OF/DATE_OF remain rule-based |
| 12 | entity resolution first-class | DONE | 2,954 test pairs: precision 0.9969, recall 0.917, F1 0.9553, false merges 5 (0.42%) |
| 13 | confidence generalises | PARTIAL | v4 cuts shifted-document ECE from 0.321 to **0.1325** and bare-value ECE to 0.0348; unseen prose is still under-confident-inducing, so gating stays required |
| 14 | OOD rejection | DONE | academic/non-academic gate, zero false-academic on 239 negatives at the chosen threshold |
| 15 | OCR robustness | PARTIAL | measured −2.1 (lookalikes), −4.2 (joined words), −6.0 combined; not trained on |
| 16 | context length investigated | DONE | 127/256/384/512 sweep, no monotone gain → 127 kept (`AUDIT.md` §1) |
| 17 | long-document strategy | DONE (new) | `pace/long_document.py`: 0 → 679 pieces on a 397 k doc, 0.42 coverage; tests added |
| 18 | structured memory | OPEN | not implemented; no evidence it would help yet |
| 19 | architecture search | PARTIAL | ELECTRA-shaped 15.3 M vs MiniLM/bge baselines measured; no new backbone trained this pass |
| 20 | multi-task learning | OPEN | single-task choice model only |
| 21 | hard-negative mining | PARTIAL | entity-resolution hard negatives (AB/BC, honors/AP, I/II); no automated loop |
| 22 | failure-driven loop | DONE | each run's failures read and used (control arm, domw, jv5, jv7 decisions documented) |
| 23 | real-world evaluation expands | PARTIAL | 1,638 unseen-syllabus rows + 354 SIS rows + 302 doc-type rows + **new leave-one-publisher-out benchmark** (`scripts/build_source_holdout.py`, 5 publishers, 103 k held-out rows); still 14 schools |
| 24 | institution-holdout benchmark | DONE | `scripts/build_school_folds.py`, 3 grouped folds, pooled 1,638 rows, leakage reported |
| 25 | template-holdout | DONE | composed families excluded from every fold; exact-overlap rate reported per fold |
| 26 | keep baselines | DONE (new) | re-scored on the same rows: IID tie (jev 0.99815 acc / 0.97112 macro-F1 vs bge-small 0.99828 / 0.97088, `artifacts/jev-field-v2/baseline_compare.json`); **on unseen schools the compact model loses**: 0.491 acc / 0.388 macro-F1 vs bge-small 0.651 / 0.413 over the same 1,638 rows (`artifacts/field-crossinstitution`) |
| 27 | Academic IR | DONE (pre-existing) | `pace/api.py` schema `academic-ir/0.2` |
| 28 | provenance | DONE (pre-existing) | evidence spans retained per field; long-doc pieces carry char spans |
| 29 | training infrastructure | DONE (new) | `scripts/training_manifest.py` writes `artifacts/MANIFEST.json`: 28 checkpoints with architecture, parameter count, weight hash, per-epoch history and the four training corpora with row counts and SHA-256. Still no git in this checkout (`.git` absent), which is a host limit, not a gap in the record |
| 30 | experiment tracking | DONE | `TRANSFER.md` table + `artifacts/transfer-eval/*.json` |
| 31 | tokenization review | PARTIAL | 16 k vocab inspected via failures; no tokenizer retrain |
| 32 | class imbalance | DONE | sampler power 0.5 + macro-F1 reporting per class |
| 33 | label quality | DONE (audited) | 177/534 errors read: majority are gold conflicts or family ambiguity, documented with examples |
| 34 | size is a constraint not the objective | DONE | 15.3 M kept after larger encoder showed no transfer gain (audit) |
| 35 | mobile deployability | PARTIAL | int8 weights exist and are measured (36.6 MB, 1.98 ms/row, `artifacts/jev-field-v2/quantization.json`); no ONNX/phone artifact: traced graph bakes batch/seq, scripting hits `prim::is_nested` |
| 36 | quantization | DONE (new) | torch dynamic int8: 61.4 → 36.6 MB (1.68×), 2.62 → 1.98 ms/row (1.32×), argmax agreement 1.000, probability MAE 3.2e-6; parity test added (`scripts/measure_quantization.py`) |
| 37 | performance | DONE | 15,331,563 params, 61.3 MB, 14.7 ms/item, 97 items/s (desktop CPU) |
| 38 | don't overfit to syllabus extraction | DONE | field corpus + catalogs + SIS + doc-type all scored |
| 39 | transcripts | DONE (pre-existing) | deterministic transcript path with cell verification tests |
| 40 | course catalogs | DONE | catalog rows dominate training; per-class F1 reported |
| 41 | schedules | PARTIAL | schedule family in corpus + doc-type `schedule` class; no layout reconstruction |
| 42 | gradebooks | OPEN | no real public gradebook text; deliberately not synthesised (plan §8 mistake) |
| 43 | layout features | OPEN | no bbox/row features in the field model |
| 44 | vision optional | DONE | semantic core reads text only |
| 45 | hard test suite | DONE (new) | `tests/test_reader_hard_cases.py`: empty/whitespace/numeric/CJK/Thai/5 k-char inputs still return a proper distribution, determinism, truncation past the state window, headerless values, a rewritten menu, and a check that the published 0.9 threshold really separates on bare values. Plus the shape/AB-BC/OCR probes from earlier |
| 46 | automated regression gates | DONE | `scripts/regression_gate.py` gates field corpus (−0.005), unseen schools (−0.002) and SIS holdout (−0.020); it **rejects** v3 on unseen schools (−0.0098), which is the demotion recorded in `artifacts/transfer-eval/gate-v3.json` |
| 47 | model selection weights real generalisation | DONE | v3 demoted on school-holdout evidence; v2 kept |
| 48 | ensembles | DONE | soups measured (v4, v6, v8); accepted only when all metrics held |
| 49 | distillation | DONE (new) | `pace/jev_distill.py` into the 10.3 M architecture (KL T=2 + CE, pretrain init, temperature refit): 0.9958/0.8435 on the field corpus, 0.588/0.315 on unseen documents — **rejected**. Soft labels do not beat the teacher at this budget, and the earlier direct fine-tune of the same size scores higher (0.9155). Recorded in `TRANSFER.md` |
| 50 | active learning | DONE (new) | one measured round (`scripts/active_learning_round.py`, 1,500 validation rows added, retrained from v4): uncertainty-selected 0.854/0.714 on unseen documents vs random-budget 0.858/0.710 vs v4 0.839/0.659 — labels help, entropy selection does **not** beat random, and the AL arm costs SIS macro-F1 (0.485 vs 0.554). Selection criterion rejected, extra-data effect confirmed |
| 51 | stopping criteria | PARTIAL | A–J assessed; K blocked by ONNX |
| 52 | performance targets | PARTIAL | met for IID/doc-type/entity-resolution and now for shifted-document ECE (0.133); still missed for unseen-school field F1 |
| 53 | model swap question | DONE | **v4 is now the primary reader** (regression gate `accept`: field macro-F1 +0.002, unseen-syllabus +0.196, SIS −0.016); v3 stays the district-format specialist; v2 kept as the audited baseline (`jev-field-v4/MODEL_CARD.md`) |
| 54 | revisit 127-token limit | DONE | sweep run; 127 kept |
| 55 | typed questions | DONE | `JevEngine.classify` / `score_against` with described options |
| 56 | no hallucinated facts | DONE (new) | calibrated abstention measured (`scripts/measure_abstention.py`, `artifacts/jev-field-v4/abstention.json`): at ≥0.90, unseen syllabus documents keep 86.3% coverage at 90.2% precision, unseen catalogue cells 97–99% coverage at 94–97%, while the two publishers it cannot read reach no threshold at all (abstain entirely). Clopper-Pearson lower bounds are in the file |
| 57 | document real improvements | DONE | `TRANSFER.md` old-vs-new tables; corrections recorded, including a demotion |
| 58 | don't destroy working capabilities | DONE | no production checkpoint overwritten; all new dirs |
| 59 | test suite | DONE | 422 tests pass, including shape, transfer-tool, long-document, int8-parity, source-holdout-sampler and v4-on-bare-values tests |
| 60 | final audit | DONE | `artifacts/transfer-eval/AUDIT.md` |
| 61 | deliverables | PARTIAL | all present except the deployable quantized artifact |
| 62 | final report | DONE | `TRANSFER.md` + `AUDIT.md` + model cards |
| 63 | most important principle | PARTIAL | school-holdout changed a conclusion (the gap is document family, not school); **v4 attacked it and moved 534 unseen documents from 0.648/0.425 to 0.839/0.659**. Leave-one-publisher-out separates catalogue cells (0.93–0.99) from syllabus prose (0.39–0.67 on seen labels), and the v4 recipe *lowers* cross-publisher transfer (0.793 → 0.658), so the concept claim is not finished |
| 64 | final instruction | PARTIAL | gaps are now measured and named (source-holdout prose, ONNX artifact, 14-school ceiling, uncalibrated new-institution confidence); this pass produced a shipped model swap, not only a report |

## The four things still open, in order

1. **ONNX/INT8** — replace `nn.TransformerEncoder` with an explicit attention block
   whose reshapes use `-1`, preserving state-dict names; re-run
   `scripts/export_jev_onnx.py`. Contained, mechanical. (Weights-level int8 is
   already measured and shipped in the card: 36.6 MB, 1.98 ms/row, identical
   decisions — what is missing is the `.onnx` file a phone runtime needs.)
2. **Taxonomy repair** — the `work` family (assignments/readings/schedule/assessment)
   is where 41% of syllabus rows live and 0.459 family accuracy. Needs relabelling
   with section headings, not more epochs.
3. **Section-aware input and header-vocabulary coverage** — heading + value as the
   state text; the measured heading/body disagreement (0.22) says this is the
   highest-value modelling change, and the leave-one-publisher-out result says a
   new publisher's header vocabulary has to be in training for the header path to
   work at all (`artifacts/source-holdout-hd`).
4. **Active learning / hard negatives** (§21, §50) — scripts exist for mining but no
   loop feeding failures back into training automatically.
