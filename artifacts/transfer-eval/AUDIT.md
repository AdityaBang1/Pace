# Four questions the plan asked, measured

Every number here comes from a script in `scripts/` and a file in
`artifacts/transfer-eval/`. Nothing is estimated.

## 1. Does the 127-token window limit quality? (§16, §54)

Same training recipe, same fold, only `--max-state` differs. Scored on the
school-holdout fold 0 (544 rows from four schools the model never saw):

| window | accuracy | macro-F1 | ECE | relative cost |
|-------:|---------:|---------:|----:|---|
| 127 | 0.4026 | 0.2305 | 0.429 | 1.0× |
| 256 | 0.3768 | **0.2394** | 0.460 | ~1.6× |
| 384 | **0.4154** | 0.2264 | 0.453 | ~2.1× |
| 512 | 0.3787 | 0.2278 | 0.486 | ~2.8× |

**Answer: no.** The differences are inside the noise of 544 rows (accuracy swings
±0.04 with no monotone trend), while latency and memory grow with the window. The
127-token model stays. Longer inputs need segmentation, not a longer window — see
§4 below.

## 2. How much does OCR damage cost? (§15)

`scripts/measure_ocr_robustness.py`, `jev-field-v2`, 1,638 unseen-syllabus rows,
corruptions applied per row with a fixed seed:

| corruption | accuracy | macro-F1 | accuracy drop |
|---|---:|---:|---:|
| clean | 0.6148 | 0.4002 | — |
| lookalike characters (`l`→`1`, `O`→`0`, `rn`→`m`, 6%) | 0.5940 | 0.3806 | −2.1 |
| joined words (25% of spaces lost) | 0.5733 | 0.3475 | **−4.2** |
| duplicated lines (30%) | 0.6093 | 0.3946 | −0.6 |
| punctuation swaps (40%) | 0.6105 | 0.3993 | −0.4 |
| flattened layout (newlines → spaces) | 0.6148 | 0.4002 | 0.0 |
| all combined | 0.5549 | 0.3890 | **−6.0** |

Character confusions cost little; **lost word boundaries cost the most**, which
points at tokenisation robustness rather than at the classifier. Flattening is
free for a 127-token value reader — and catastrophic for sectioning, see below.
These corruptions were not in training: they are an evaluation, not a fix.

## 3. What happens to a real long document? (§17)

**Found and fixed in this pass.** `detect_document_sections` is line-anchored, so a
document whose line structure was lost yields zero sections and the reader saw its
first 127 tokens: a 396,667-character syllabus returned `other` at 99.8%. Reading
the *before* column is the point of the table:

| document | chars | sections before | sections after | coverage after | method | time |
|---|---:|---:|---:|---:|---|---:|
| UCLA SOCIOL 1 | 396,667 | **0** | **679** | 0.42 | window fallback | 18.3 s |
| UIUC CS 125 | 64,639 | 21 | 21 | 0.90 | heading sections | 0.4 s |

`pace/long_document.py` keeps the heading path first and falls back to
paragraph boundaries, then to sentence-aligned windows with 80-character overlap,
only when the heading scan finds nothing. Each piece records its method and its
character span, so the application can show which unit a field came from and
distinguish a real section from a window. Tests in `tests/test_long_document.py`
cover the headed case, the flattened case, text with no sentence breaks, and the
no-silent-drop rule.

The honest remaining caveat: on the flattened 397 k document only **42%** of
windows receive a non-`other` label (the rest is navigation, contact blocks and
boilerplate), and headings/body agreement measured earlier on headed sections was
0.22 — the heading is catalog-like, the body is syllabus prose from an unseen
school. Section-level output must therefore be presented as per-section
proposals, never as a document-level conclusion.

## 4. Can the selected model be deployed? (§35, §36, §37)

Measured on this desktop CPU (RTX 4070 machine, CPU path), `jev-field-v2`:

| quantity | value |
|---|---|
| parameters | 15,331,563 |
| fp32 weights | 61.3 MB |
| single-item latency (batch 1) | **14.7 ms** |
| throughput (batch 64, short values) | **97 items/s** |
| menu | 42 described options, cached once |

ONNX export is **blocked, and the blocker is in the model**: tracing
`nn.TransformerEncoder` bakes the example's batch and sequence length into the
attention reshapes (`requested shape {14,256,80}` for a 99×64×320 input), scripting
the module fails on `prim::is_nested`, and ONNX Runtime's INT8 quantizer rejects
the traced graph during shape inference (`Inferred shape and existing shape differ
in dimension 0: (128) vs (320)`). So:

- **fp32 PyTorch on CPU works and is fast enough for an on-device reader.**
- **INT8 was not measured**, because no valid graph could be produced yet. Do not
  quote a quantized size or latency for this checkpoint.
- Fixing it means replacing `nn.TransformerEncoder` with an explicit attention
  block whose reshapes use `-1` (the state-dict names would need to be preserved),
  then re-running `scripts/export_jev_onnx.py`. That is a contained task, not a
  research one.

**Correction (quantization is not blocked).** The ONNX road is blocked, but torch's
own dynamic INT8 conversion of the checkpoint works and was measured
(`scripts/measure_quantization.py`, `artifacts/jev-field-v2/quantization.json`, 600
real rows):

| quantity | fp32 | dynamic int8 |
|---|---|---|
| weights | 61.4 MB | **36.6 MB** (1.68×) |
| latency | 2.62 ms/row | **1.98 ms/row** (1.32×) |
| argmax agreement | — | **1.000** (600/600 rows) |
| probability MAE | — | 3.2e-6 (max 9.4e-5) |
| accuracy on those rows | 1.000 | 1.000 |

One torch detail worth keeping: dynamic quantized `nn.Linear` stores a function in
`weight`, so `nn.TransformerEncoder`'s sparsity fast-path probe raises
`AttributeError: 'function' object has no attribute 'device'` unless
`torch.backends.mha.set_fastpath_enabled(False)` is set first. Both the script and
the parity test do this.

## The biggest measurement error in the earlier audit

Every training row is `"<column>: <value>"` and the column usually names the
field. Nothing in the earlier audit noticed that the model could therefore read
the prefix instead of the value. The same 7,571 held-out rows, scored twice
(`scripts/measure_header_reliance.py`):

| checkpoint | published rows | value only (`bare_text`) | gap |
|---|---:|---:|---:|
| `jev-field-v2` (audited as 0.9982) | 0.9982 acc / 0.9711 F1 | 0.8991 / 0.7000 | 0.099 acc, 0.271 F1 |
| `jev-field-v4` (new) | 0.9983 / 0.9734 | **0.9473 / 0.8172** | 0.051 acc, 0.156 F1 |

So roughly a tenth of the headline accuracy and a quarter of the macro-F1 came
from the header, which is precisely the part of an institution's conventions that
changes between institutions. Fixing it (`scripts/build_header_dropout.py`, half
the header-bearing training rows lose their header) improved *both* views and
moved the benchmark that matters:

| benchmark | v2 | v4 |
|---|---:|---:|
| syllabus document-shift (534 new documents) | 0.648 / 0.425 / ECE 0.293 | **0.839 / 0.659 / ECE 0.133** |
| SIS cells (354) | 0.847 / 0.492 / 0.122 | **0.915** / 0.476 / **0.065** |
| bare values (7,571) | 0.899 / 0.700 | **0.947 / 0.817** |

## What this leaves open

| question | status |
|---|---|
| Is 127 tokens limiting? | **answered: no** |
| Is OCR tolerance adequate? | **partial**: −4.2 points when spaces are lost, −6.0 combined |
| Is long-document reading ready? | **no**: no fallback segmentation, 22% heading/body agreement |
| Is INT8 deployable? | **yes at the weights level**: 36.6 MB, 1.98 ms/row, identical decisions; an ONNX file for phone runtimes is still blocked |
| Is fp32 on-device reading viable? | **yes**: 14.7 ms/item, 61 MB, measured |
| Was the shipped accuracy reading the value or the column? | **partly the column**: 0.099 accuracy / 0.271 macro-F1 of v2's headline came from the header; v4 halves that dependence (`header_reliance.json`) |
| Does it read a publisher it has never seen? | **catalogue cells yes, syllabus prose no**: leave-one-publisher-out means 0.9865 (MIT), 0.9715 (USC), 0.9288 (Michigan), 0.6702 (Gothenburg, seen labels), 0.3887 (14-school syllabi, seen labels). The v4 recipe makes this *worse* (mean seen-label 0.793 → 0.658): the header is a shortcut inside a known corpus and the only handle on an unknown publisher |
| Does the compact model beat a full encoder off-distribution? | **no**: on the same 1,638 unseen-school rows bge-small scores 0.651 acc / 0.413 macro-F1 against our 0.491 / 0.388 (`artifacts/field-crossinstitution`). In-distribution the two are tied to the fourth decimal. The size advantage is real; the generalisation advantage is not there yet. |
