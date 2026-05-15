"""
Compare two checkpoints on MRPC validation and print examples
where the best model is correct and the worst model is wrong.
"""

import argparse

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def get_preds(path, raw):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = raw.map(
        lambda b: tok(b["sentence1"], b["sentence2"], truncation=True),
        batched=True,
    )
    ds = ds.remove_columns([c for c in ds.column_names
                            if c not in ("input_ids", "attention_mask", "token_type_ids")])
    loader = DataLoader(ds, batch_size=64, collate_fn=DataCollatorWithPadding(tok))

    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out.extend(torch.argmax(model(**batch).logits, dim=-1).cpu().tolist())
    return np.asarray(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_model_path", required=True)
    p.add_argument("--worst_model_path", required=True)
    p.add_argument("--num_examples", type=int, default=20)
    args = p.parse_args()

    raw = load_dataset("nyu-mll/glue", "mrpc")["validation"]
    labels = np.asarray(raw["label"])

    best = get_preds(args.best_model_path, raw)
    worst = get_preds(args.worst_model_path, raw)

    print(f"best  acc: {(best == labels).mean():.4f}")
    print(f"worst acc: {(worst == labels).mean():.4f}")

    idxs = np.where((best == labels) & (worst != labels))[0]
    print(f"# examples where best is correct and worst is wrong: {len(idxs)}")

    n = min(args.num_examples, len(idxs))
    for k, i in enumerate(idxs[:n]):
        ex = raw[int(i)]
        print(f"\n[{k+1}] gold={ex['label']} best={best[i]} worst={worst[i]}")
        print("  s1:", ex["sentence1"])
        print("  s2:", ex["sentence2"])


if __name__ == "__main__":
    main()
