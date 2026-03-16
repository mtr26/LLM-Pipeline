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
Need to update this can't lie
We're using a H100 here on Modal for 1.8B tokens.
For training I used Argparse instead of Hydra because of I python version issues.
The model was trained on GCP using a single L4 GPU with 24GB of VRAM.
The training used mixed precision (bf16) and torch compile to optimize the training speed.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REX model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_val_ratio", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    mlflow.set_experiment("REX Pre-training")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = load_dataset("Maynx/RexContinousPreTraining", split="train")
    print(f"Loaded {len(dataset)} sequences.")
    dataset = dataset.train_test_split(test_size=1-args.train_val_ratio)

    model = REX.from_pretrained(
        args.model_path,
        device_map=None,
        low_cpu_mem_usage=False
    )

    for block in model.blocks:
        block.attention.generate_sin_cos_pos_emb(model.config.max_len)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,                  
        dataloader_num_workers=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_torch_fused",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        save_total_limit=2,
        weight_decay=0.05,
        warmup_ratio=0.02,                   # Gentle warmup to avoid shocking the weights
        lr_scheduler_type="cosine",          # Smooth decay for mathematical convergence
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        bf16=True,
        learning_rate=1.5e-4,
        report_to=["mlflow"],
        run_name="REX_Pretraining_Run",
        torch_compile=True,                      
        torch_compile_mode="default",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )



    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(f"{args.output_dir}/saved")
