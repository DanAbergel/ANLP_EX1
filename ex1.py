"""Fine-tune bert-base-uncased on MRPC for paraphrase detection."""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "mrpc"
RES_FILE = "res.txt"
PREDICTIONS_FILE = "predictions.txt"


class LossLogger(TrainerCallback):
    def __init__(self, out_path):
        self.out_path = out_path
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.records.append({"step": state.global_step, "loss": float(logs["loss"])})
            with open(self.out_path, "w") as f:
                json.dump(self.records, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


def load_splits(max_train, max_eval, max_predict):
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train_ds = raw["train"]
    eval_ds = raw["validation"]
    test_ds = raw["test"]
    if max_train != -1:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_eval != -1:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))
    if max_predict != -1:
        test_ds = test_ds.select(range(min(max_predict, len(test_ds))))
    return train_ds, eval_ds, test_ds


def tokenize_dataset(dataset, tokenizer):
    def _tok(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
    return dataset.map(_tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def run_name(epochs, lr, batch_size):
    return f"ep{epochs}_lr{lr}_bs{batch_size}"


def append_res(epochs, lr, batch_size, eval_acc):
    line = f"epoch_num: {epochs}, lr: {lr}, batch_size: {batch_size}, eval_acc: {eval_acc:.4f}\n"
    with open(RES_FILE, "a") as f:
        f.write(line)


def do_train(args):
    os.environ.setdefault("WANDB_PROJECT", "anlp-ex1-mrpc")
    if not os.environ.get("WANDB_API_KEY") and "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_ds, eval_ds, _ = load_splits(args.max_train_samples, args.max_eval_samples, -1)
    train_tok = tokenize_dataset(train_ds, tokenizer)
    eval_tok = tokenize_dataset(eval_ds, tokenizer)

    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "label"}
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols])
    eval_tok = eval_tok.remove_columns([c for c in eval_tok.column_names if c not in keep_cols])

    name = run_name(args.num_train_epochs, args.lr, args.batch_size)
    output_dir = os.path.join("runs", name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=["wandb"],
        run_name=name,
        seed=42,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    os.makedirs("runs", exist_ok=True)
    loss_logger = LossLogger(os.path.join("runs", f"{name}_losses.json"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_logger],
    )
    trainer.train()

    metrics = trainer.evaluate()
    eval_acc = metrics["eval_accuracy"]
    print(f"[{name}] validation accuracy = {eval_acc:.4f}")
    append_res(args.num_train_epochs, args.lr, args.batch_size, eval_acc)

    final_dir = os.path.join("runs", name, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


def do_predict(args):
    if args.model_path is None:
        raise ValueError("--do_predict requires --model_path")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _, _, test_ds = load_splits(-1, -1, args.max_predict_samples)
    sentences1 = list(test_ds["sentence1"])
    sentences2 = list(test_ds["sentence2"])

    test_tok = tokenize_dataset(test_ds, tokenizer)
    keep_cols = {"input_ids", "attention_mask", "token_type_ids"}
    test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in keep_cols])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_tok, batch_size=64, collate_fn=collator)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)

    with open(PREDICTIONS_FILE, "w") as f:
        for s1, s2, p in zip(sentences1, sentences2, all_preds):
            s1 = s1.replace("\n", " ").strip()
            s2 = s2.replace("\n", " ").strip()
            f.write(f"{s1}###{s2}###{p}\n")

    print(f"Wrote {len(all_preds)} predictions to {PREDICTIONS_FILE}")


def main():
    args = parse_args()
    if not (args.do_train or args.do_predict):
        raise SystemExit("Must pass --do_train and/or --do_predict")
    if args.do_train:
        do_train(args)
    if args.do_predict:
        do_predict(args)


if __name__ == "__main__":
    main()
