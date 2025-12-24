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
from itertools import chain


"""
This script was used to pre trained REX from scratch using a custom dataset.
The dataset was a JSONL file with a "text" field. For each line like this:
{"text": "This is a sample text."}
The training set was a mix between a subset of C4 and Wikipedia dumps.
The model was trained for 350M tokens (Mistral 7B tokenizer) for two epochs.
Also for more details about the training setup, check the README.md file.
"""


def load_and_tokenize_datasets(
    dataset_name_or_path: str, # Changed to support Hub datasets
    tokenizer_name: str = "gpt2",
    max_length: int = 1024, # Increased for 300M model standard
    train_val_ratio: float = 0.95,
):
    # 1. Load the dataset (Supports local JSON or HF Hub)
    if dataset_name_or_path.endswith(".json"):
        raw_dataset = load_dataset('json', data_files=dataset_name_or_path, split='train')
    else:
        # Optimized for the FineWeb-Edu dataset we discussed
        raw_dataset = load_dataset(dataset_name_or_path, name="sample-10BT", split="train")

    # 2. Use the FAST tokenizer (Rust-based)
    # This is the single biggest speedup for tokenization
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # GPT-2 fix: It usually doesn't have an unk_token, so we use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Create Split
    split_dataset = raw_dataset.train_test_split(
        train_size=train_val_ratio,
        shuffle=True,
        seed=42
    )
    
    # 4. Define CPU count for parallel processing
    # Leave 1-2 cores free so the OS doesn't freeze
    num_proc = max(1, os.cpu_count() - 2)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    # 5. Tokenize in Parallel
    tokenized_datasets = split_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc, # <--- PARALLELIZATION
        remove_columns=[
            "text", "id", "dump", "url", "file_path", 
            "language", "language_score", "token_count", 
            "score", "int_score"
        ],
        desc="Tokenizing dataset"
    )

    # 6. Optimized Grouping (The "Packing" Step)
    def group_texts(examples):
        # Concatenate all texts. 
        # 'itertools.chain' is much faster than sum(list, [])
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder, we want tight blocks
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
            
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # 7. Group in Parallel
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000, # Process larger chunks at once
        num_proc=num_proc, # <--- PARALLELIZATION
        desc="Grouping texts"
    )

    return lm_datasets, tokenizer


"""
For training I used Argparse instead of Hydra because of I python version issues.
The model was trained on GCP using a single L4 GPU with 24GB of VRAM.
The training used mixed precision (bf16) and torch compile to optimize the training speed.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REX model.")
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_val_ratio", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    mlflow.set_experiment("REX Pre-training")

    datasets, tokenizer = load_and_tokenize_datasets(
        dataset_name_or_path=args.dataset_file_path,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        train_val_ratio=args.train_val_ratio
    )

    config = REXConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_length,
        n_layers=18,
        n_heads=16,
        n_kv_heads=4,
        n_embd=1024,
        dropout=0.1,
    )

    model = REX(config=config)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,                  
        dataloader_num_workers=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_torch_fused",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=1000,
        metric_for_best_model="loss",
        bf16=True,
        learning_rate=3e-4,
        report_to=["mlflow"],
        run_name="REX_Pretraining_Run",
        torch_compile=True,                      
        torch_compile_mode="default",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
    )



    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(f"{args.output_dir}/saved")
