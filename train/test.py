import torch
from model.model import REXConfig, REX, generate_texts
from transformers import AutoTokenizer, Trainer, TrainingArguments
import safetensors
import os
from datasets import load_dataset, DatasetDict

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id


config = REXConfig(
    vocab_size=tokenizer.vocab_size,
    max_len=1024,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    n_embd=768,
    dropout=0.1,
)

sf_path = "models"
model = REX(config=config)

model.save_pretrained(sf_path)

print(sum(p.numel() for p in model.parameters()) / 1e6)

model = REX.from_pretrained(sf_path)


