"""Compare the predictions of two MRPC checkpoints on the validation set
and print the examples where the best model is correct and the worst is wrong."""

import argparse
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def predict(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    raw = load_dataset("nyu-mll/glue", "mrpc")["validation"]

    def _tok(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True, max_length=tokenizer.model_max_length,
        )

    tok = raw.map(_tok, batched=True)
    keep = {"input_ids", "attention_mask", "token_type_ids"}
    tok = tok.remove_columns([c for c in tok.column_names if c not in keep])
    loader = DataLoader(tok, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return raw, np.asarray(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_model_path", required=True)
    p.add_argument("--worst_model_path", required=True)
    p.add_argument("--num_examples", type=int, default=20)
    args = p.parse_args()

    raw, best_preds = predict(args.best_model_path)
    _, worst_preds = predict(args.worst_model_path)
    labels = np.asarray(raw["label"])

    best_correct = best_preds == labels
    worst_wrong = worst_preds != labels
    interesting = np.where(best_correct & worst_wrong)[0]

    print(f"Validation size:           {len(labels)}")
    print(f"Best  accuracy:            {best_correct.mean():.4f}")
    print(f"Worst accuracy:            {(worst_preds == labels).mean():.4f}")
    print(f"Best correct & Worst wrong: {len(interesting)}")
    print()

    label_counts = Counter(int(labels[i]) for i in interesting)
    print(f"Gold label distribution among 'best-only-correct' examples: {dict(label_counts)}")
    print()

    n = min(args.num_examples, len(interesting))
    print(f"--- {n} sample examples ---")
    for k, idx in enumerate(interesting[:n]):
        ex = raw[int(idx)]
        print(f"\n[{k + 1}] gold = {ex['label']}, best = {best_preds[idx]}, worst = {worst_preds[idx]}")
        print(f"    s1: {ex['sentence1']}")
        print(f"    s2: {ex['sentence2']}")


if __name__ == "__main__":
    main()
