# jev-field-v3 — the district / Infinite Campus-format checkpoint

`artifacts/jev-field-v3` is a weight average, **0.7 · `jev-field-v2` + 0.3 ·
`jv5-field`**, with the temperature refit on the validation split (0.6283). It
exists because district exports (Infinite Campus, OneRoster) are a document form
the general reader handles badly: the general reader scores 0.847 accuracy /
0.492 macro-F1 / ECE 0.122 on the 354-cell SIS holdout, and this average scores
0.8955 / 0.5307 / 0.0657.

It is a **format specialist, not a general replacement**. On syllabus documents
from unseen schools it is worse than the general reader (0.624 / 0.433 / 0.321),
and since v4 it is no longer the best reader of values either.

## Note on this file

The checkpoint was deleted and rebuilt from its documented recipe during the v4
work; the rebuilt weights were verified to reproduce the numbers in
[`../transfer-eval/TRANSFER.md`](../transfer-eval/TRANSFER.md) exactly:

| benchmark | rows | accuracy | macro-F1 | ECE |
|---|---:|---:|---:|---:|
| field corpus test | 7,571 | 0.9980 | 0.9739 | 0.0014 |
| syllabus document-shift | 534 | 0.6236 | 0.4330 | 0.3209 |
| SIS / Infinite Campus holdout | 354 | 0.8955 | 0.5307 | 0.0657 |

The original `metrics.json` and card text were not preserved; the numbers above
were re-measured from the rebuilt `model.pt` with `python -m pace.jev_eval`.

## Rebuild

```bash
python scripts/soup_checkpoints.py --inputs artifacts/jev-field-v2 artifacts/jv5-field \
  --weights 0.7 0.3 --output artifacts/jev-field-v3
cp artifacts/jev-field-v2/tokenizer.json artifacts/jev-field-v3/tokenizer.json
python scripts/fit_temperature.py --checkpoint artifacts/jev-field-v3
```
