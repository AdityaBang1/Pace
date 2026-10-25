"""Compare real catalog model results on the exact shared subject-held-out split."""
from __future__ import annotations
import json
from pathlib import Path


def _load_metrics(name):
    return json.loads((Path('artifacts') / name / 'metrics.json').read_text(encoding='utf-8'))


def main():
    baseline = _load_metrics('usc-real')
    mini = _load_metrics('usc-miniLM')
    deberta = _load_metrics('usc-deberta-xsmall')
    baseline_errors = sum(a != b for a, b in zip(baseline['test_true_labels'], baseline['test_predictions']))
    mini_errors = sum(row['errors'] for row in mini['class_diagnostics'].values())
    deberta_errors = sum(row['errors'] for row in deberta['class_diagnostics'].values())
    rows = []
    for label, metrics, artifact in [
        ('char TF-IDF logistic', baseline, Path('artifacts/usc-real/usc_char_tfidf.joblib')),
        ('MiniLM-L6 fine-tuned', mini, Path('artifacts/usc-miniLM/model.pt')),
        ('DeBERTa-v3-xsmall fine-tuned', deberta, Path('artifacts/usc-deberta-xsmall/model.pt')),
    ]:
        rows.append({
            'model': label,
            'test_accuracy': metrics['test_accuracy'],
            'test_macro_f1': metrics['test_macro_f1'],
            'test_weighted_f1': metrics['test_weighted_f1'],
            'parameters': metrics.get('parameter_count'),
            'serialized_bytes': artifact.stat().st_size,
            'gpu': metrics.get('gpu'),
        })
    payload = {
        'dataset': 'USC/USC-Course-Catalog',
        'task': 'natural catalog-cell field classification',
        'same_holdout_subjects': baseline['split_subject_counts']['test'],
        'shared_test_rows': baseline['split_rows']['test'],
        'same_split_assignment': (
            baseline['subject_split_assignment'] == mini['subject_split_assignment'] == deberta['subject_split_assignment']
            and baseline['split_rows'] == mini['rows_by_split'] == deberta['rows_by_split']
            and baseline['split_subject_counts'] == mini['subjects_by_split'] == deberta['subjects_by_split']
        ),
        'ambiguous_value_excluded': baseline.get('ambiguous_values', mini.get('ambiguous_values', {})),
        'errors_on_test': {
            'char TF-IDF logistic': baseline_errors,
            'MiniLM-L6 fine-tuned': mini_errors,
            'DeBERTa-v3-xsmall fine-tuned': deberta_errors,
        },
        'models': rows,
        'observed_failure': 'All 2,652 exact normalized TBA cells (1,317 Days and 1,335 Time) are excluded because text-only input does not identify the originating column. Inference abstains on this known ambiguous literal. Remaining observed errors are listed per model; scores measure structured CSV-cell field recognition only.',
        'excluded_candidates': 'TinyBERT has no explicit HF license metadata; not trained. MobileBERT has Apache-2.0 and remains a candidate, not trained. ModernBERT-base is Apache-2.0, 149M/8192 and was not needed for this short-cell experiment.',
    }
    out = Path('artifacts/usc-real/comparison.json')
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
