import os
import argparse
import mlflow
from datasets import load_dataset
import safetensors
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from model.model import REXConfig, REX
from safetensors import safe_open as safetensors_open
from safetensors.torch import load_file, load
import torch
import torch.nn as nn

def load_and_tokenize_datasets(
    dataset_file_path: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 128,
    train_val_ratio: float = 0.95,
):
    raw_dataset = load_dataset('json', data_files=dataset_file_path, split='train')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    split_dataset = raw_dataset.train_test_split(
        train_size=train_val_ratio,
        shuffle=True,
        seed=42
    )
    split_dataset['validation'] = split_dataset.pop('test')

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)

    tokenized_datasets = split_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length < max_length:
            return {}
        total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i:i+max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    return lm_datasets, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REX model.")
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_val_ratio", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    mlflow.set_experiment("REX Pre-training")

    datasets, tokenizer = load_and_tokenize_datasets(
        dataset_file_path=args.dataset_file_path,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        train_val_ratio=args.train_val_ratio
    )

    config = REXConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_length,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        n_embd=768,
        dropout=0.1,
    )

    model = REX(config=config)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=5000,
        metric_for_best_model="loss",
        bf16=True,
        learning_rate=3e-4,
        report_to=["mlflow"],
        run_name="REX_Pretraining_Run",
        torch_compile=True,                      
        torch_compile_mode="reduce-overhead",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
    )



    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(f"{args.output_dir}/saved")
