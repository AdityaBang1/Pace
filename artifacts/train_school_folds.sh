set -e
for i in 0 1 2; do
  PYTHONPATH=. python -m pace.jev_train --stage choice \
    --encoder artifacts/jev-field-v2-pretrain \
    --output artifacts/school-fold-$i \
    --tokenizer artifacts/jev-field-v2/tokenizer.json \
    --train data/processed/transfer/school-fold-$i/train.jsonl \
    --validation data/processed/transfer/school-fold-$i/validation.jsonl \
    --test data/processed/transfer/school-fold-$i/test.jsonl \
    --epochs 2 --batch-size 192 --lr 8e-4 --patience 1 \
    --option-dropout 0.15 --sampler-power 0.5 --hidden 320 --ffn 1024 \
    > artifacts/school-fold-$i.log 2>&1
done
