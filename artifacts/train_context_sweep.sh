set -e
for s in 256 384 512; do
  PYTHONPATH=. python -m pace.jev_train --stage choice \
    --encoder artifacts/jev-field-v2-pretrain \
    --output artifacts/ctx-$s \
    --tokenizer artifacts/jev-field-v2/tokenizer.json \
    --train data/processed/transfer/school-fold-0/train.jsonl \
    --validation data/processed/transfer/school-fold-0/validation.jsonl \
    --test data/processed/transfer/school-fold-0/test.jsonl \
    --epochs 2 --batch-size 128 --lr 8e-4 --patience 1 \
    --option-dropout 0.15 --sampler-power 0.5 --hidden 320 --ffn 1024 \
    --max-state $s > artifacts/ctx-$s.log 2>&1
done
