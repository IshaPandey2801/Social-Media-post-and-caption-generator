# fine_tune_rewriter.py  (low-memory variant)
import argparse
from pathlib import Path
import logging
import os
import torch

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fine_tune(
    train_file,
    valid_file,
    model_name="t5-small",
    output_dir="outputs/t5_rewriter",
    batch_size=1,
    num_epochs=1,
    learning_rate=5e-5,
    fp16=False,
    gradient_accumulation_steps=1,
    save_total_limit=2,
):
    # limit CPU threads for lower memory/overhead
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {}
    if train_file:
        data_files["train"] = train_file
    if valid_file:
        data_files["validation"] = valid_file

    ds = load_dataset("json", data_files=data_files)
    logger.info(ds)

    # tokenizer and model (small)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # VERY small lengths to save memory
    max_input_length = 32
    max_target_length = 32

    def preprocess(batch):
        inputs = batch["input"]
        targets = batch["output"]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names if "train" in ds else ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length")

    # Try to load rouge but continue if not available
    compute_metrics = None
    if "validation" in tokenized:
        try:
            rouge = evaluate.load("rouge")
            def compute_metrics_fn(eval_pred):
                preds, labels = eval_pred
                if isinstance(preds, tuple):
                    preds = preds[0]
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                res = rouge.compute(predictions=decoded_preds, references=decoded_labels)
                return {k: float(v) for k, v in res.items()}
            compute_metrics = compute_metrics_fn
        except Exception:
            logger.warning("Rouge metric not available — continuing without compute_metrics.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=False,   # generate is memory heavy; disable for training
        logging_steps=20,
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_total_limit=save_total_limit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=False,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"] if "train" in tokenized else None,
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    logger.info("Training complete. Model saved to %s", str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="output/train.jsonl")
    parser.add_argument("--valid", type=str, default="output/valid.jsonl")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--output_dir", type=str, default="outputs/t5_rewriter")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--save_limit", type=int, default=2)
    args = parser.parse_args()

    fine_tune(
        train_file=args.train,
        valid_file=args.valid,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        gradient_accumulation_steps=args.grad_accum,
        save_total_limit=args.save_limit,
    )
