# The one-model page reader: an experiment, not the production reader

This directory is the surviving one-model experiment: a single network that reads
transcript pages straight from pixels. It is **not** the production reader.
Production remains the two-stage path in `pace/ingest.py` (text layer or OCR, then
cell verification), whose numbers are in `artifacts/ingest/` and whose behaviour
is unchanged by this experiment.

On 2026-10-22 the model was rebuilt onto the page family this project actually
ingests. The previous checkpoint had been trained on invented landscape pages and
read **no field** on a real rendered page; that measurement is kept below as
[HISTORY](#history-the-landscape-model). The current checkpoint reads every field
of the fixture's real page, none refused and none wrong. With two views and the
duplicate rule it is close to a safe reader but not there: some emitted values are
still wrong, not every gold row is read completely, and a wrong character can
still be confident. The tables below are the exact figures.

- Model: `PageSpotter` — 290,161 parameters, fully convolutional, 1224×1584 page
  in, per-position character presence and class at half resolution out.
  No template, no column table and no cell geometry is supplied.
- Vocabulary: upper- and lower-case letters, digits, `.` and `,` (64 classes).
  Course-code validation mirrors the shipped reader's rule —
  three to six digits and a one- or two-letter suffix — so `220725y` is
  representable. Weights must print a decimal point; a numeric string with a
  dangling point is refused rather than read as a number.
- Input frame: portrait US letter at two pixels per printer's point. The old
  1152×640 landscape frame compressed a portrait page into a 494-pixel strip and
  reduced 9 pt type to about 7 px.
- Training: 800 documents → 1,379 pages, 12 epochs, batch 4, **932.2 s** on an
  RTX 4070 (83.5 s of that is rendering). Artifacts: `model.pt`, `metrics.json`.
- Reproduce, from the repository root:

  ```
  python scripts/train_page_spotter.py --eval-only --heldout-documents 60 --report-only
  ```

  This re-scores the saved checkpoint and prints the held-out summary and the
  real-page table without writing: `model.pt` and `metrics.json` keep their
  hashes (`03bc6f52…` and `443193f5…`). It is the whole reproduce path — no
  retraining, no `--train-documents`. Dropping `--report-only` rewrites
  `metrics.json` from the same checkpoint and reproduces the stored file byte for
  byte, so the stored record is this code's output for this checkpoint; the
  `training` block, the seeds and the training-split counts are carried over from
  the stored file rather than taken from this run's arguments. A re-score that
  writes only writes into a directory that already holds a record — pointed at a
  bare copy of `model.pt` it stops and says so instead of inventing a training
  block it did not run. Training itself was seeded but cuDNN is not bitwise
  deterministic, so retraining reproduces the family, not this checkpoint.

## Training distribution

Each document is drawn in the geometry of the project's own renderer
(`pace/transcript_render.py`), with per-document jitter around it:

- portrait US letter, 2 px per printer's point (1224×1584);
- 9–10 pt Arial, the Helvetica-metric face available here, for body rows; the
  renderer's column layout — code at the left margin, a wrapped title, then
  right-aligned mark, weight and credit;
- course codes of five or six digits and one lower-case letter (`220725y`,
  `42075y`), marks 55–100, weights from `{0.25, 0.5, 0.75, 1, 1.5}` printed to
  four decimals, credits printed trimmed;
- header row, term banner and footer totals as page furniture.

Every printed glyph is a presence target, so the model must find all the ink;
only the four measured fields carry a class target. Titles and headings are
therefore located but not transcribed — the task is the four fields.

Documents are drawn whole and **split by document**: none of the 104 held-out
pages belongs to any of the 800 training documents. A quarter of training pages
are left clean; every held-out page is degraded with illumination falloff, blur
and JPEG, the way a capture is.

## Held-out documents (60 documents, 104 pages, floor 0.5)

The confidence floor and peak threshold were fixed before evaluation (0.5 and
0.35, the module defaults); no floor or geometry was tuned on these documents.
The table is the current pipeline: the duplicate-suffix rule plus a second reading
that must agree character for character.

| field | emitted | refused | wrong | wrong share of emitted |
|---|---:|---:|---:|---:|
| course_code | 512 | 56 | 4 | 0.8% |
| mark | 486 | 78 | 1 | 0.2% |
| weight | 438 | 126 | 1 | 0.2% |
| credit | 414 | 150 | 1 | 0.2% |
| **all fields** | **1,850** | **410** | **7** | **0.38%** |

396 of the 564 gold rows (70.2%) were read completely correctly; 112 were partly
read; 56 had no field emitted — against 437, 86 and 41 for the single reading.
937 row records were decoded in total, the difference being column headers, title
continuation lines and footers, which are refused as course rows. Twelve further
rows were seen by only one reading (6 in the framed view, 6 in the shifted one);
they are counted as `unpaired_rows` and never emitted.

### Step 1: duplicate-suffix refusal (2026-10-22)

The worst failure mode was a second presence peak on a code's final letter. The
code rule accepts a one- or two-letter suffix, so `733983w` was emitted as the
legal-looking `733983ww` — a different, confidently wrong code. The decoder now
checks the closing letters of every code: two peaks of the same letter closer than
three-quarters of the run's own character advance are close enough that peak
jitter, not printed spacing, explains the distance, so they are one detection and
the code is refused, not repaired. The check deliberately does not touch digits: a repeated
digit on a degraded page can sit a little inside its advance while still being
printed there, and refusing those would cost correct codes. The confidence floor
and peak threshold are unchanged (0.5 and 0.35), as are the held-out split and
gold.

| held-out field | before: emitted / refused / wrong | after: emitted / refused / wrong |
|---|---:|---:|
| course_code | 543 / 41 / 20 | 537 / 41 / 14 |
| mark | 516 / 48 / 5 | 516 / 48 / 5 |
| weight | 493 / 71 / 12 | 493 / 71 / 12 |
| credit | 449 / 115 / 8 | 449 / 115 / 8 |
| **all fields** | **2,001 / 275 / 45** | **1,995 / 275 / 39** |

Both columns are float64 readings. The after column is the stored single-view
baseline (`heldout_single_view` in `metrics.json`, with its complete wrong-value
list); the before column needs `_duplicated_suffix` disabled, so it survives only
as the numbers above, without a stored list.

Six wrong codes are no longer emitted. Because those six gold rows were already
unmatched (the doubled code did not equal the printed one), the row counts do not
move, and no correct code was refused. The real pages are unchanged (their table
is below).

The check is conservative by construction: this family prints a one-letter
suffix, but a genuinely narrow two-letter suffix (say `ll`) could sit inside its
advance and be refused too. That is a refusal, not a wrong value, and no code on
this split has a two-letter suffix.

### Step 2: second reading agreement (2026-10-22)

The wrongs left after step 1 were single confident glyph errors. No spelling rule
can catch those, so each page is now read twice. The second view is the same framed
page shifted 3 px right and 2 px down and scaled 1.015 about its centre (bilinear,
white fill), so glyphs land on different output cells and are resampled. Both views
are decoded with the same checkpoint, floor 0.5 and peak 0.35; the second view's
rows are mapped back through the transform that produced it and paired only where
the pairing is unambiguous — the row's single candidate inside 4 px must also have
that row as its single candidate — so a neighbouring row closer than the decoding
jitter refuses instead of standing in for the twin. A field is emitted only where
the two readings produced the same raw text **before** validation. Any
disagreement is a refusal: nothing is voted on, repaired or taken from the more
confident view. The duplicate-suffix rule runs inside each reading.

| held-out | before: emitted / refused / wrong | after: emitted / refused / wrong |
|---|---:|---:|
| course_code | 537 / 41 / 14 | 512 / 56 / 4 |
| mark | 516 / 48 / 5 | 486 / 78 / 1 |
| weight | 493 / 71 / 12 | 438 / 126 / 1 |
| credit | 449 / 115 / 8 | 414 / 150 / 1 |
| **all fields** | **1,995 / 275 / 39** | **1,850 / 410 / 7** |

The before column is the stored `heldout_single_view`; the after column is
`heldout_documents`. Both store complete wrong-value lists, so the comparison can
be checked value by value.

Agreement bought 32 fewer wrong values for 145 fewer emitted values — one in 264
emitted values is wrong now, one in 51 before. The catches are uneven: 10 of the
14 wrong codes, 4 of the 5 marks, 11 of the 12 weights and 7 of the 8 credits. No
new wrong value appears: every survivor was already on the earlier list.

The real pages were re-measured with both readings and are unchanged (table
below).

### How it fails now

The wrong values that survive both readings are the same confident mistake made in
each view, so agreement cannot see them: four codes (`71707u`, `10685l`,
`38893l`, `61295l`), one mark (`76` read as `78`), one weight (`1.0000` as
`1.9000`) and one credit (`0.5` as `5.0`). Nothing in the two outputs contradicts
these readings; catching them needs a third, genuinely different signal.

The cost is visible in refusals, and the credit column pays most (26.6% of its
rows): a field of one to three characters is refused whole when one character is
unstable. A stray trailing peak turning `1` into `1.` is still refused by the
format rule.

## Real pages from this tree (floor 0.5, no tuning)

Gold: the fixture's 8 course rows, matched by course code. "Read" means all four
fields equal the fixture.

| Page | rows decoded | read / 8 | partly read | refused | code | mark | weight | credit |
|---|---:|---:|---:|---:|---|---|---|---|
| PDF page, 2x raster (1224×1584) | 15 | 8 | 0 | 0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 |
| rendered PNG (1224×1584) | 15 | 8 | 0 | 0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 |
| 256×202 capture | 57 | 0 | 0 | 8 | 0/8/0 | 0/8/0 | 0/8/0 | 0/8/0 |
| degraded PNG (0.5°, blur 0.6, JPEG 60) | 15 | 8 | 0 | 0 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 |

Per-page field cells are emitted / refused / wrong.

The capture has 21 unpaired rows (9 seen only in the framed view, 12 only in the
shifted one): 48 rows were seen by both readings, and the shifted reading decoded
60. None is a course row, and no gold field is emitted from it — its fragment rows
are noise, and a handful carry a stray credit (`1.0`) with no code, so none is a
course record. That page's rows sit about as close together as the decoder's row
positions jitter, so five pairs it used to accept are now left unpaired — one of
them had certified a credit `1.0` against a neighbouring row's `1`. The other three
real pages and the 104 held-out pages pair every row exactly as before, and those
three have no unpaired rows.

On the real page every field is read: the second row's credit, which a float32
reading truncated to `0.`, is `0.5` in the float64 reading. All 32 field values
are correct on the three rendered pages.

The capture is a 256-pixel-wide crop of the table, not a page-scale raster; the
capture path remains unvalidated.

## Verdict

The pixels-to-records claim now holds on the real **rendered** page in the narrow
sense that the model reads the page: every field of the fixture is emitted and
right. It does **not** yet hold as a guarantee:

- emitted held-out values are still sometimes wrong, codes worst of all;
- not every held-out row is read completely, so some rows still need another
  path;
- a wrong character can be confident, and some survived both readings, so the
  experiment's safety property — every value equal to the fixture or null — is not
  yet met;
- the only real pages measured are renders of the fixture. No camera capture,
  scan or second institution has been read.

This is a usable **experiment** and a promising prototype. It is not a usable
single-model reader for the product, and its metrics must not be quoted as the
product's accuracy. The honest summary is: it now reads the real page family, and
it still sometimes lies.

## History: the landscape model

The first checkpoint (288,887 parameters) was trained on 1,200 invented landscape
pages with comma-separated fields and six-digit upper-case codes. Its held-out
numbers were good inside that family: 418 rows with 327 codes on seen layouts
(fields 318 / 313 / 299 / 289, 49 wrong) and 420 rows with 377 codes on unseen
layouts (366 / 354 / 337 / 315, 78 wrong); at the 0.9 floor, 295 codes with 7
wrong and 326 with 6 wrong.

It was then run untrained-adapted on the real pages in this tree: the PDF raster,
its PNG, the 256×202 capture and a degraded render. With the training pipeline's
own framing it decoded 9, 9, 20 and 8 rows and emitted **no field at all** on any
of them — 0 / 0 / 0 emitted, all refused, no wrong values. The row text was
nonsense digits (`8840600761`, `0666`, `66`, `009600`). Two causes were visible:
the landscape frame shrank 9 pt portrait type to about 7 px, and the vocabulary
and validation could not represent the fixture's lower-case codes.

That measurement is what the rebuild above answers; the numbers in this section
describe a checkpoint that no longer exists.

## Deleted attempts, and why

This tree is not under version control, so the losing experiments were recorded
here before their code and artifacts were removed.

### Grid reader — deleted 2026-10-22

- 467,657 parameters; 1,200 pages, 24 epochs, 50.8 s.
- 1,004 of 1,200 rows read at the 0.9 floor with **0 wrong**; per field
  1,004 / 991 / 989 / 995.
- **Why deleted:** the model was handed a fixed 12×40 character grid, so it was
  never a pixels-to-records reader. Numerically the best of the three, but invalid
  as evidence for the one-model path.
- Removed: `pace/page_model.py`, `scripts/train_page_model.py`,
  `tests/test_page_model.py`, `artifacts/page-model/`.

### Sequence reader — deleted 2026-10-22

- 235,504 parameters; 1,200 pages, 45 epochs, 398.6 s.
- Honest row counts: 413/418 and 405/420 rows decoded (a learned presence score
  gates empty slots), fixed from an earlier 444 and 457.
- Coverage collapsed: 129 and 74 codes at the 0.5 floor with 22 and 28 wrong;
  4 and 6 codes at the 0.9 floor with 0 wrong.
- **Why deleted:** the spotter reads 327 and 377 codes at the same floor. The
  sequence path fixed its phantom rows but lost more coverage than it gained.
- Removed: `pace/page_sequence.py`, `scripts/train_page_sequence.py`,
  `tests/test_page_sequence.py`, `artifacts/page-sequence/`.

## Public entry

`detect_and_read` in `pace/page_spotter.py` is now the only way this experiment
reads a page: page images in, one record per page out. Each record is
`{"rows": [...], "unpaired": [...]}`. A row carries the four fields (`None` where
unread), `unread_cells`, the raw `row_text` and whether both readings saw it;
`unpaired` lists the rows only one reading saw, as text with the view that saw
them, and no value is emitted from them. The module owns the second view, the
mapping of its row positions back into the first frame, and the agreement merge;
the script no longer reimplements any of it. Pairing is mutual and unique rather
than nearest: a row is compared with another only when each is the other's single
candidate within 4 px, so two rows closer together than that cannot confirm each
other's fields. `views=1` reproduces the single
reading the tables compare against, and the evaluation now counts the rows only
one reading saw (`unpaired_rows`, split by view) instead of dropping them.

The surface was broken: `PageSpotter.parameters` was a property, so
`detect_and_read` raised `TypeError` for the very class it documents, and every
number above had been produced by calling the network and `decode_page` directly.
It is a method now, matching torch's convention, and the device lookup accepts
either shape. Re-scoring the table through this entry is what exposed the next
defect: the same checkpoint and the same pages gave different fields on a GPU and a
CPU. Eight tests drive the entry: a real `PageSpotter`, a single view, a second
view that disagrees on one field, a row only one reading saw, a CPU/CUDA
comparison, the documented 3 px / 2 px / 1.015 warp of the pixels the network is
actually handed, the real page's known rows and field values, and the capture's
known refusals and unpaired split. The last three exist because the first five
would every one of them pass with `second_view` returning the page unchanged:
three of them fed the network a stub that synthesised rows from page coordinates,
one only checked the shape of a record from an untrained network, and the CPU/CUDA
comparison reads the same view on both devices — so nothing in the suite held the
second reading to its own pixels. Of the three that close it, two load the real
checkpoint and read real pages; the third records the batch the entry hands its
network and checks that it is the documented warp of the page.

### Same page, same fields, whatever the device

The decoder is deterministic given its inputs — it is numpy float32 over the logits
— so the difference entered in the forward pass. Measured on one page: cuDNN's
default TF32 mode put a GPU and a CPU 1.6e-2 apart in logits; with TF32 off they
were 2.9e-5 apart. That second number matters because the decisions sit close to
their boundaries: across the 108 pages the tightest presence peak is 1.8e-4 above
the 0.35 floor and the tightest pair of character classes is 3.8e-5 apart — the
latter at the device difference itself. So a few borderline decisions crossed: the
GPU read 1,845 / 415 / 7 held-out values and 7 of the 8 real rows, the CPU 6 of
the 8.

The forward now runs in float64 on a copy of the network (`reading_copy`), on
whichever device the network sits. Only an object with no `double` at all — the
reader's test doubles — is used un-widened; a network that has one and fails while
widening raises, because falling back would put the device dependence back
silently. A CPU and a GPU then agree to 7e-14 in logits,
and changing the CPU thread count moves them by 3e-14 — nine orders below that
3.8e-5 tie. Re-scored through `detect_and_read`, all 108 pages (the four real ones
and the 104 held-out) produce identical rows, unpaired rows and fields on both,
and a CPU and a CUDA re-score produce identical held-out and real-page blocks of
`metrics.json`. The real pages are now read completely, including the credit that
float32 truncated to `0.`.

That moves the record once, to the float64 reading in the tables above: the
held-out wrong count is unchanged and the real pages are read completely. Both
real-row results were float32 artifacts, the GPU's and the CPU's respectively; the
256×202 refusal is the same in all three.

The two before-fix baselines came from runs whose temporary output was deleted;
they exist only as the numbers in this section, not as artifacts, and the current
float64 path cannot regenerate them. Everything else in the tables is
stored in `metrics.json` — both current columns and the single-view baseline —
except Step 1's 2,001 / 275 / 45, which needs the duplicate rule switched off.

## Limits

Synthetic documents drawn in the real family, not real captures; titles and
headings are located but not transcribed; the row grouping, the row pairing, the
column split and the field validation are hand-written, not learned; one real rendered page (two
rasters and one degraded variant of it) is the only out-of-sample page measured;
the 256×202 capture is refused entirely; the second reading costs 135 extra
refusals and 41 fully-read rows. Refusal is the safety property this experiment
argues for, and it is close but does not yet hold: the misses described above show
why. The float64 reading also costs about three
times the float32 forward (0.66 s against 0.21 s per view on this CPU); that is
the price of the fields not depending on the device.
