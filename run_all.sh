#!/bin/bash
# Run the three MRPC fine-tuning configurations sequentially.
# All runs use the full train and validation splits.
set -e
cd "$(dirname "$0")"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

echo "=== Config 1/3: ep=3 lr=2e-5 bs=16 ==="
python3 ex1.py --do_train --max_train_samples -1 --max_eval_samples -1 \
    --num_train_epochs 3 --lr 2e-5 --batch_size 16 2>&1 | tail -40

echo
echo "=== Config 2/3: ep=3 lr=5e-5 bs=32 ==="
python3 ex1.py --do_train --max_train_samples -1 --max_eval_samples -1 \
    --num_train_epochs 3 --lr 5e-5 --batch_size 32 2>&1 | tail -40

echo
echo "=== Config 3/3: ep=3 lr=1e-4 bs=32 ==="
python3 ex1.py --do_train --max_train_samples -1 --max_eval_samples -1 \
    --num_train_epochs 3 --lr 1e-4 --batch_size 32 2>&1 | tail -40

echo
echo "=== ALL DONE ==="
cat res.txt
