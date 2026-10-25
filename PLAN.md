# Plan: a field model that actually holds up

Written 2026-10-21. This is the working plan for the next training round. Every goal has a
number attached, and every number is produced by a command in this repo, not typed by hand.
Status is updated as work lands.

## What is wrong with the current model

The shipped checkpoint (`artifacts/field-classifier-minilm`) is trained on a corpus capped at
**6,000 rows per field type** — 163,797 rows kept out of 1,405,290 available. 23 of 42 field
types sit exactly on that ceiling. Consequences:

- In-distribution accuracy (0.989) is measured on a label-balanced subsample, not a real
  distribution.
- The long tail is *enumerated, not learned*: `drop_add_deadline` 2 rows, `attendance_policy` 8,
  `course_policies` 27, `exam_schedule` 79, `materials` 98.
- Calibration is excellent in-distribution (ECE 0.002) and poor on a new university
  (ECE 0.21–0.33). A percentage the user reads is therefore not trustworthy where it matters.
- "No false positives" has never been measured as an operating point. Nobody knows the
  precision/coverage curve at a 0.95 or 0.99 threshold, on familiar or unfamiliar documents.

## Goals and acceptance criteria

| # | Goal | Measured by | Acceptance |
|---|---|---|---|
| G1 | Train on the uncapped corpus, no per-label ceiling | `data/processed/field_uncapped`, per-class support and F1 vs the capped model on the same labels | Long-tail classes move from "unmeasured" to a measurable F1; no regression > 0.5 pt on common classes |
| G2 | Probabilities that mean something | reliability bins + ECE, in-distribution and on unseen universities | ECE reported per protocol; a calibrated threshold whose observed precision is within 0.02 of its nominal value |
| G3 | A no-false-positive operating point | precision / coverage / false-positive rate sweeps at 0.50–0.995 | Report the threshold where precision ≥ 0.99, the coverage there, and the abstention rate |
| G4 | Breadth check against what a person needs | label coverage audit vs syllabus/catalog/transcript fields | Every field a person would extract is either a trained label or explicitly out of scope |
| G5 | Transcripts | field model applied to transcript rows, plus the rule-based parser | Report per-field accuracy on transcript rows; say plainly what is synthetic |
| G6 | The docs cannot drift again | regenerated artifacts + a test that fails when a headline number disagrees | `pytest` covers the calibration report and the headline figures |

## Execution order

1. Rebuild the corpus with no cap into `data/processed/field_uncapped`.
2. Train one MiniLM on the uncapped group split, 3 epochs, same recipe otherwise.
3. Add `pace/field_calibration.py`: reliability bins, ECE, precision/coverage/FPR sweep,
   per-label precision at a chosen threshold, abstention.
4. Evaluate: uncapped test split, capped test split (control), and the three existing
   cross-institution fold checkpoints (unseen universities, no retraining needed).
5. Pick the operating threshold on validation only, then report its test behaviour.
6. Transcript coverage pass.
7. Update `REPORT.md`, `MODEL_CARD.md`, `README.md` with measured numbers; add tests.

## Honest limits, fixed in advance

- The corpus is real university text, but the 42 labels are a harmonised vocabulary. A field
  type with 8 real rows cannot be "learned"; it can only be reported as thin.
- Uncapping adds mostly catalog rows. Transfer to a new *university* is a syllabus problem, so
  G1 is expected to help the long tail far more than it helps cross-institution accuracy. If
  that is what the numbers say, that is what gets reported.
- Transcripts in this repo are synthetic mocks unless a real labelled source is found; the
  parser's accuracy on them is not evidence about real transcripts.

## Status

| Step | State |
|---|---|
| Plan written | done |
| Uncapped corpus | done — 1,405,290 rows, 26,818 groups, `was_capped: false` |
| Calibration tool | done — `pace/field_calibration.py`, 7 unit tests |
| Uncapped training | done — 4 checkpoints (capped, uncapped, +oversampled 3 ep, +oversampled 6 ep) |
| Transcript audit | done — `pace/field_transcript_audit.py` |
| Evaluations | done — calibration, precision sweeps, per-class comparison |
| Docs + tests | done — REPORT, MODEL_CARD, README, 7 new tests |

## Outcome

| Goal | Result |
|---|---|
| G1 uncapped training | **Failed to improve.** Three uncapped variants score macro-F1 0.9183 / 0.9283 / 0.9262 against 0.9518, and tail accuracy 0.915 / 0.927 / 0.921 against 0.976, on identical rows. Cap retained, now justified by measurement. Caveat: the uncapped runs used batch 256 and one seed. |
| G2 true probabilities | Done. ECE 0.0024 in-distribution, 0.2559 bare / 0.1008 heading+value on unseen universities. |
| G3 no-false-positive point | **Not reachable on unseen universities** (best 94.7% precision). Certified on familiar data: 99.68% lower bound at the 0.99 threshold. |
| G4 breadth | Done. 42 labels audited; transcript date has no field, `grades`/`grading_policy` conflated. |
| G5 transcripts | Done and negative: bare cells fail; the parser, not the classifier, is the right tool. |
| G6 drift | Partly: artifacts and commands are documented. A test that cross-checks report numbers is still the next step. |

## What a follow-up should do

1. Add transcript-specific labels (`mark`, `credit_earned`, `completion_date`, `student_standing`) with real rows, then a column-aware model.
2. Cross-institution retrain on the uncapped corpus was **not** run; the syllabus component is unchanged, but that is an untested assumption, not a result.
3. The 99% ceiling on new universities needs either more syllabus-hosted labels or a stronger encoder, not a threshold.

## Findings so far (measured, not estimated)

**A 99% precision promise is impossible on a new university with the current model.**
On the three held-out university folds, the bare-value model accepts 93.8% of rows at a 0.50
threshold and is right 67.4% of the time; even at 0.995 it is only 91.5% precise on 50.3%
of rows. No threshold reaches 99%. Pooled ECE is 0.256 and every confidence bin is
overconfident — the 0.7–0.8 bin claims 75% and delivers 21%.

**The heading-aware model is much closer.** Same folds, 1,295 rows: accuracy 0.870, ECE
0.101, and at a 0.99 threshold it covers 88.2% of rows at 94.7% precision (FPR 4.7%). Still
short of 99%, but the failures are no longer 25 points of overconfidence.

**Transcripts are not a classifier problem, and the classifier fails on them.** On 38 parsed
course records: marks (`grades`) 0.868 agreement, titles (`course_title`) 0.474, and the
numeric cells collapse — a credit value `0.250` is read as `course_code` 38/38, a grade level
`09` as `class_section` 25/38. There is also no vocabulary field at all for a transcript
date. The rule-based parser is the right tool for tabular records; the classifier is for prose
and headings. Any claim that the field model "reads transcripts" would be false.

**The sample size needed for a 99% claim is now explicit.** Zero errors in 300 accepted rows
certifies 99.01% precision (95% Clopper-Pearson); zero errors in 100 rows certifies only
97.05%. This is reported per threshold, so a precision claim can fail.

