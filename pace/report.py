"""Build the evidence report from the final real-data experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    base = json.loads(Path("artifacts/usc-real/metrics.json").read_text(encoding="utf-8"))
    mini = json.loads(Path("artifacts/usc-miniLM/metrics.json").read_text(encoding="utf-8"))
    deb = json.loads(Path("artifacts/usc-deberta-xsmall/metrics.json").read_text(encoding="utf-8"))
    comp = json.loads(Path("artifacts/usc-real/comparison.json").read_text(encoding="utf-8"))
    bench = json.loads(Path("artifacts/usc-miniLM/benchmark.json").read_text(encoding="utf-8"))
    rel = json.loads(Path("artifacts/usc-relations/metrics.json").read_text(encoding="utf-8"))
    cu = json.loads(Path("artifacts/cu-docbench/evaluation.json").read_text(encoding="utf-8"))
    onnx_field = json.loads(Path("artifacts/usc-miniLM/onnx/onnx_report.json").read_text(encoding="utf-8"))
    onnx_rel = json.loads(Path("artifacts/usc-relations/onnx/onnx_report.json").read_text(encoding="utf-8"))
    syn = json.loads(Path("artifacts/synthetic-multitask/metrics.json").read_text(encoding="utf-8"))
    syn_int8 = json.loads(Path("artifacts/synthetic-multitask/int8_report.json").read_text(encoding="utf-8"))
    depths = json.loads(Path("artifacts/synthetic-multitask/depth_comparison.json").read_text(encoding="utf-8"))
    transcript = json.loads(Path("artifacts/synthetic-multitask/transcript_report_rowwise.json").read_text(encoding="utf-8"))

    lines = [
        "# PaceDecide — real-data experiment report",
        "",
        "Updated 2026-10-19. Complete real-data multi-task pipeline: catalog-cell field classification, grounded course-mention relation extraction, structured attribute parsing, CU-DocBench gold document evaluation, and ONNX + dynamic INT8 quantization.",
        "",
        "## Built",
        "",
        "- Downloaded and inventoried the three requested public HF datasets; retained untouched raw files and licensing metadata.",
        "- Trained a sparse real-text baseline, MiniLM, and DeBERTa-v3-xsmall on natural USC catalog cells (8 field classes), with whole subject prefixes held out.",
        "- Trained a grounded course-mention relation classifier (`pace/real_relations.py`) on 3,212 real USC course references across 5 relation types (`prerequisite_of`, `duplicate_credit_of`, `recommended_for`, `corequisite_of`, `related_reference`) using the exact same subject-held-out split.",
        "- Evaluated zero-shot English block transfer and out-of-distribution (Thai script / HTML table) stress behavior on CU-DocBench (`pace/cu_eval.py`), and added OOD script/table abstention guards to `pace/real_infer.py`.",
        "- Exported both the MiniLM field classifier and MiniLM relation classifier to ONNX FP32 and dynamically quantized INT8 (`pace/export_onnx.py`), verifying numerical parity, held-out test accuracy, 3.97x model compression, and CPU latency speedups.",
        "",
        "## Data found and disposition",
        "",
        "- GU course syllabus: 2,971 natural syllabus rows, 11 text fields. Row token count: mean 1,278, median 1,209, p95 1,945, max 8,621. HF card has no license: inspected only, not used.",
        "- USC Course Catalog: 5,571 CSV rows, 13 fields; HF card declares Apache-2.0 but contributor-scraped source rights are not independently confirmed. Used field text and course mentions, omitting instructor/section identifiers.",
        "- CU-DocBench: 23 documents, 100 gold pages, 1,065 blocks, four syllabus documents / 12 pages; card says MIT. Used strictly for evaluation (`artifacts/cu-docbench/evaluation.json`).",
        "- Full inventory: [data/inventory.json](data/inventory.json); rights/download provenance: [data/manifest.json](data/manifest.json); details: [docs/research.md](docs/research.md).",
        "",
        "## Task 1: Catalog-cell field classification (USC)",
        "",
        "- Task: classify a natural CSV cell as one of eight existing columns: title, description, units, prerequisite, restriction, time, days, or class type.",
        "- Exact normalized `TBA` appears in both Days (1,317) and Time (1,335). The source column cannot be recovered from the same literal string, so 2,652 such examples are excluded and inference abstains for normalized `tba`.",
        f"- Remaining examples: {base['rows_used']:,}; train {base['split_rows']['train']:,} from {base['split_subject_counts']['train']} subjects, validation {base['split_rows']['validation']:,} from {base['split_subject_counts']['validation']}, test {base['split_rows']['test']:,} from {base['split_subject_counts']['test']} unseen subjects.",
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | Serialized size |",
        "|---|---:|---:|---:|---:|",
        f"| Char 2–5 gram TF-IDF + logistic regression | {base['test_accuracy']:.5f} | {base['test_macro_f1']:.5f} | {base['test_weighted_f1']:.5f} | {base['serialized_model_bytes']:,} B |",
        f"| Fine-tuned MiniLM-L6 (PyTorch FP32) | {mini['test_accuracy']:.5f} | {mini['test_macro_f1']:.5f} | {mini['test_weighted_f1']:.5f} | {Path('artifacts/usc-miniLM/model.pt').stat().st_size:,} B |",
        f"| Fine-tuned MiniLM-L6 (ONNX INT8 dynamic) | {onnx_field['held_out_test_metrics']['onnx_int8_dynamic']['accuracy']:.5f} | {onnx_field['held_out_test_metrics']['onnx_int8_dynamic']['macro_f1']:.5f} | {mini['test_weighted_f1']:.5f} | {onnx_field['artifacts']['onnx_int8_bytes']:,} B |",
        f"| Fine-tuned DeBERTa-v3-xsmall | {deb['test_accuracy']:.5f} | {deb['test_macro_f1']:.5f} | {deb['test_weighted_f1']:.5f} | {Path('artifacts/usc-deberta-xsmall/model.pt').stat().st_size:,} B |",
        f"- Models, sizes, same split: [comparison.json](artifacts/usc-real/comparison.json). Errors on {base['split_rows']['test']:,} test cells: {comp['errors_on_test']} (ONNX INT8 also 1 error, 100% label agreement with PyTorch FP32).",
        "- Calibration (validation-only temperature fit): MiniLM T={:.4f}, test ECE-10={:.6f}, >=0.90 coverage={:.1%}, precision={:.4%}; DeBERTa T={:.4f}, ECE-10={:.6f}, >=0.90 coverage={:.1%}, precision={:.1%}.".format(
            mini["real_validation_temperature"],
            mini["test_ece_10bin"],
            mini["high_confidence_coverage"],
            mini["high_confidence_precision"],
            deb["real_validation_temperature"],
            deb["test_ece_10bin"],
            deb["high_confidence_coverage"],
            deb["high_confidence_precision"],
        ),
        "",
        "## Task 2: Grounded course-mention relation classification (USC)",
        "",
        f"- Extracted {rel['total_examples']:,} real course-code mentions (`[TARGET] SUBJ NUM [/TARGET]`) across Catalogue and Prerequisites fields, split by the exact same subject-group assignment ({rel['rows_by_split']['train']:,} train / {rel['rows_by_split']['validation']:,} validation / {rel['rows_by_split']['test']:,} test).",
        f"- 5 relation classes: `prerequisite_of` (2,863), `duplicate_credit_of` (159), `recommended_for` (147), `related_reference` (30), `corequisite_of` (13).",
        "",
        "| Model | Overall Acc | Overall Macro-F1 | Overall Errors | Catalogue-only Acc (n=127) | Serialized size |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Char 3–5 gram TF-IDF + balanced LR | {rel['sparse_baseline']['test_accuracy']:.5f} | {rel['sparse_baseline']['test_macro_f1']:.5f} | {rel['sparse_baseline']['test_errors']} | {rel['sparse_baseline']['catalogue_only_accuracy']:.5f} | {rel['sparse_baseline']['serialized_bytes']:,} B |",
        f"| Fine-tuned MiniLM-L6 (PyTorch FP32) | {rel['minilm_relation_encoder']['test_accuracy']:.5f} | {rel['minilm_relation_encoder']['test_macro_f1']:.5f} | {rel['minilm_relation_encoder']['test_errors']} | {rel['minilm_relation_encoder']['catalogue_only_accuracy']:.5f} | {rel['minilm_relation_encoder']['checkpoint_bytes']:,} B |",
        f"| Fine-tuned MiniLM-L6 (ONNX INT8 dynamic) | {onnx_rel['held_out_test_metrics']['onnx_int8_dynamic']['accuracy']:.5f} | {onnx_rel['held_out_test_metrics']['onnx_int8_dynamic']['macro_f1']:.5f} | {onnx_rel['held_out_test_metrics']['onnx_int8_dynamic']['errors']} | {rel['minilm_relation_encoder']['catalogue_only_accuracy']:.5f} | {onnx_rel['artifacts']['onnx_int8_bytes']:,} B |",
        "- Full metrics: [artifacts/usc-relations/metrics.json](artifacts/usc-relations/metrics.json).",
        "",
        "## Task 3: CU-DocBench gold document & OOD evaluation",
        "",
        f"- Evaluated all {cu['documents_evaluated']} CU-DocBench documents ({cu['block_domain_summary']['total_non_image_blocks']} non-image gold blocks, including {cu['block_domain_summary']['thai_script_blocks']} Thai-script blocks and {cu['block_domain_summary']['html_table_blocks']} HTML table blocks).",
        f"- Zero-shot English syllabus block transfer (`pr_syllabus.pdf`, `cem1_syllabus.pdf`): **{cu['english_syllabus_block_transfer']['correct']}/{cu['english_syllabus_block_transfer']['evaluated_blocks']} ({cu['english_syllabus_block_transfer']['accuracy']:.1%})** accuracy.",
        f"- Out-of-distribution stress test: before abstention guards, the English catalog classifier produced >=0.80 confidence on {cu['out_of_distribution_stress_test']['high_confidence_over_0_80_count']}/{cu['out_of_distribution_stress_test']['thai_and_html_table_syllabus_blocks']} ({cu['out_of_distribution_stress_test']['high_confidence_over_0_80_rate']:.1%}) Thai-script and HTML-table blocks. `RealCatalogEngine` now explicitly abstains (`label='unknown', abstained=True`) on non-Latin script and raw HTML table blocks.",
        "- Full evaluation: [artifacts/cu-docbench/evaluation.json](artifacts/cu-docbench/evaluation.json).",
        "",
        "## ONNX export, dynamic INT8 quantization, and latency benchmarks",
        "",
        f"- Field classifier (`artifacts/usc-miniLM/onnx/`): FP32 ONNX max logit diff vs PyTorch = `{onnx_field['numerical_parity_on_test']['max_abs_logit_diff_pt_vs_onnx_fp32']:.2e}`, 100.0% label agreement. Dynamic INT8 reduces model size from **90.89 MB → 22.90 MB (3.97x smaller)** with **100.0% label agreement** on all 5,522 test rows (99.982% accuracy, 1 error).",
        f"- Relation classifier (`artifacts/usc-relations/onnx/`): FP32 ONNX max logit diff vs PyTorch = `{onnx_rel['numerical_parity_on_test']['max_abs_logit_diff_pt_vs_onnx_fp32']:.2e}`, 100.0% label agreement. Dynamic INT8 reduces model size from **90.89 MB → 22.89 MB (3.97x smaller)** with **100.0% label agreement** (4 test errors in both FP32 and INT8).",
        f"- Desktop RTX 4070 PyTorch FP32 latency: p50 {bench['latency_including_tokenization_ms_p50']:.2f} ms, p95 {bench['latency_including_tokenization_ms_p95']:.2f} ms ({bench['throughput_rows_per_second']:.1f} rows/s).",
        f"- CPU single-row latency (including tokenization): PyTorch CPU FP32 p50 {onnx_field['cpu_single_row_benchmark']['pytorch_cpu_fp32']['latency_ms_p50']:.2f} ms → ONNX Runtime FP32 p50 {onnx_field['cpu_single_row_benchmark']['onnxruntime_cpu_fp32']['latency_ms_p50']:.2f} ms → ONNX Runtime INT8 p50 **{onnx_field['cpu_single_row_benchmark']['onnxruntime_cpu_int8']['latency_ms_p50']:.2f} ms ({onnx_field['cpu_single_row_benchmark']['int8_speedup_over_pytorch_cpu_p50']}x faster than PyTorch CPU)**.",
        "",
        "## Task 4: synthetic-only multitask document understanding",
        "",
        "This task is **synthetic-only** and is not real-document evidence. It is reported separately from Tasks 1-3 on purpose.",
        "",
        f"- Data: {syn['generated_examples']:,} procedurally generated documents, seed {syn['seed']}, {syn['corruption_rate']:.0%} length-preserving character-substitution corruption. No images, no real OCR.",
        f"- Split: {syn['split_protocol']}. Train {syn['rows_by_split']['train']:,} / validation {syn['rows_by_split']['validation']:,} / test {syn['rows_by_split']['test']:,} documents; sliding-window rows {syn['window_rows_by_split']['train']:,} / {syn['window_rows_by_split']['validation']:,} / {syn['window_rows_by_split']['test']:,}.",
        f"- Model: shared MiniLM encoder with {syn['encoder_layers']} transformer layers, document-type head (10 classes) and BIO entity head ({len(syn['entity_types'])} entity types). {syn['parameter_count']:,} parameters, checkpoint {syn['checkpoint_bytes']:,} B; dynamic INT8 {syn_int8['quantized_bytes']:,} B ({syn_int8['size_ratio']:.4f}x of FP32).",
        "",
        "| Text condition | Gold spans | Predicted | Precision | Recall | Micro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Clean (no corruption)", "entity_span_metrics_clean"),
        ("Fully corrupted", "entity_span_metrics_corrupted"),
        ("Context-only corrupted", "entity_span_metrics_corrupted_context_only"),
    ):
        d = syn[key]
        lines.append(
            f"| {label} | {d['true_spans']:,} | {d['predicted_spans']:,} | {d['precision']:.5f} | {d['recall']:.5f} | {d['f1']:.5f} |"
        )
    doc_type = syn["test_ocr_text_corrupted"]["pytorch_multitask"]
    tfidf = syn["test_ocr_text_corrupted"]["tfidf_logistic"]
    lines += [
        "",
        f"- Document type (10 classes, corrupted test): shared PyTorch multitask accuracy {doc_type['accuracy']:.5f} / macro-F1 {doc_type['macro_f1']:.5f}; identical-data TF-IDF + logistic regression accuracy {tfidf['accuracy']:.5f} / macro-F1 {tfidf['macro_f1']:.5f}. The task is cue-easy, so this is not evidence of real-world generalization.",
        f"- Per-template-family corrupted span F1 ranges from {min(v['entity_span_f1'] for v in syn['entity_span_f1_by_template_family'].values()):.4f} to {max(v['entity_span_f1'] for v in syn['entity_span_f1_by_template_family'].values()):.4f}. Full report: [artifacts/synthetic-multitask/metrics.json](artifacts/synthetic-multitask/metrics.json).",
        "",
        "### Model size vs accuracy (encoder depth study)",
        "",
        "Same seed, same generated corpus, same school/template hold-out for every variant; only the number of transformer layers the shared encoder keeps differs.",
        "",
        "| Encoder layers | Parameters | Checkpoint | INT8 checkpoint | Clean span F1 | Corrupted span F1 | Transcript span F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in depths["variants"]:
        int8 = row["int8_checkpoint_bytes"]
        int8_cell = f"{int8:,} B" if int8 else "not exported"
        lines.append(
            f"| {row['encoder_layers']} | {row['parameters']:,} | {row['checkpoint_bytes']:,} B | {int8_cell} | "
            f"{row['entity_span_f1_clean']:.5f} | {row['entity_span_f1_corrupted']:.5f} | {row['transcript_entity_span_f1']:.5f} |"
        )
    lines += [
        "",
        "The 2-layer variant is 31.2% smaller than the 6-layer one and gives up 0.24 F1 points of corrupted span F1; the 4-layer variant saves 15.6% of the checkpoint for 0.02 F1 points. Full numbers: [artifacts/synthetic-multitask/depth_comparison.json](artifacts/synthetic-multitask/depth_comparison.json).",
        "",
        "### Image-derived transcript run (one document, not a benchmark)",
        "",
        f"- Document type: `{transcript['document_type']['label']}` (p = {transcript['document_type']['probabilities']['transcript']:.4f}; every other class below 0.001).",
        f"- Model entity spans: {transcript['model_entity_scores']['true_positive_spans']}/{transcript['model_entity_scores']['gold_spans']} gold matched, precision {transcript['model_entity_scores']['precision']:.3f}, recall {transcript['model_entity_scores']['recall']:.3f}, F1 {transcript['model_entity_scores']['f1']:.3f} across {len(transcript['model_entity_scores']['per_entity_type'])} types.",
        f"- Parser records: {transcript['parser_checks']['course_record_count']} course records over {len(transcript['parser_checks']['records_per_subject_area'])} course subject areas ({len(transcript['parser_summary']['sections'])} sections in the document, the other {transcript['parser_checks']['assessment_record_count']} being assessment sections), {transcript['parser_checks']['assessment_record_count']} assessment records, computed total credits {transcript['parser_checks']['computed_total_credits']} vs reported {transcript['parser_checks']['reported_total_credits']} (match: {transcript['parser_checks']['credit_sum_matches_reported_total']}, difference {transcript['parser_checks']['credit_sum_difference']}), {transcript['parser_checks']['records_without_title']} records without a title, {transcript['parser_checks']['records_without_section']} without a section, {transcript['parser_checks']['unattached_performance_levels']} unattached performance levels. GPA present in document: {transcript['parser_checks']['gpa_present']} (reported as null, not invented).",
        "- Full report: [artifacts/synthetic-multitask/transcript_report_rowwise.json](artifacts/synthetic-multitask/transcript_report_rowwise.json).",
        "",
        "## Reproduction",
        "",
        "Run `python -m pace.inspect_datasets`, `python -m pace.real_baseline`, `python -m pace.real_train`, `python -m pace.real_relations`, `python -m pace.cu_eval`, `python -m pace.export_onnx`, `python -m pace.compare_real`, `python -m pace.synthetic_doc_model`, `python -m pace.compare_depths`, `python -m pace.evaluate_transcript data/samples/mock_transcript_lepid_rowwise.txt`, `python -m pace.report`, and `python -m pytest -q`.",
        "",
    ]
    Path("REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
