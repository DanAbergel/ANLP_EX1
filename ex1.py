import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def get_args():
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


def tokenize(ds, tok):
    return ds.map(
        lambda b: tok(b["sentence1"], b["sentence2"], truncation=True),
        batched=True,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def main():
    args = get_args()

    if not (args.do_train or args.do_predict):
        raise SystemExit("nothing to do, pass --do_train and/or --do_predict")

    # use offline mode if user is not logged into wandb (avoids crashing)
    os.environ.setdefault("WANDB_PROJECT", "anlp-ex1-mrpc")
    if not os.environ.get("WANDB_API_KEY") and "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    if args.do_train:
        run_name = f"ep{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
        out_dir = os.path.join("runs", run_name)

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

        raw = load_dataset("nyu-mll/glue", "mrpc")
        train_ds = raw["train"]
        eval_ds = raw["validation"]
        if args.max_train_samples != -1:
            train_ds = train_ds.select(range(args.max_train_samples))
        if args.max_eval_samples != -1:
            eval_ds = eval_ds.select(range(args.max_eval_samples))

        train_ds = tokenize(train_ds, tok)
        eval_ds = tokenize(eval_ds, tok)

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=64,
            logging_steps=1,         # log loss every step (asked by the spec)
            save_strategy="no",      # we only need the final model
            eval_strategy="no",
            report_to=["wandb"],
            run_name=run_name,
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tok,
            data_collator=DataCollatorWithPadding(tok),
            compute_metrics=compute_metrics,
        )
        trainer.train()

        metrics = trainer.evaluate()
        acc = metrics["eval_accuracy"]
        print(f"validation accuracy = {acc:.4f}")

        with open("res.txt", "a") as f:
            f.write(
                f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, "
                f"batch_size: {args.batch_size}, eval_acc: {acc:.4f}\n"
            )

        # save the final model so we can reuse it for --do_predict later
        trainer.save_model(os.path.join(out_dir, "final"))
        tok.save_pretrained(os.path.join(out_dir, "final"))

    if args.do_predict:
        if args.model_path is None:
            raise ValueError("--do_predict needs --model_path")

        tok = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.eval()  # important: turn off dropout for inference

        test_ds = load_dataset("nyu-mll/glue", "mrpc")["test"]
        if args.max_predict_samples != -1:
            test_ds = test_ds.select(range(args.max_predict_samples))

        s1 = list(test_ds["sentence1"])
        s2 = list(test_ds["sentence2"])
        test_tok = tokenize(test_ds, tok)

        # use Trainer.predict so we get batched inference + dynamic padding for free
        pred_args = TrainingArguments(
            output_dir="runs/_pred",
            per_device_eval_batch_size=64,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=pred_args,
            processing_class=tok,
            data_collator=DataCollatorWithPadding(tok),
        )
        logits = trainer.predict(test_tok).predictions
        preds = np.argmax(logits, axis=-1)

        with open("predictions.txt", "w") as f:
            for a, b, p in zip(s1, s2, preds):
                f.write(f"{a.strip()}###{b.strip()}###{int(p)}\n")
        print(f"wrote {len(preds)} predictions to predictions.txt")


if __name__ == "__main__":
    main()
