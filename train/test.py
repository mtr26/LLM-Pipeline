import torch
from model.model import REXConfig, REX, generate_texts, generate_texts_no_kv
from transformers import AutoTokenizer, Trainer, TrainingArguments
import safetensors
import os
from datasets import load_dataset, DatasetDict

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

model = REX.from_pretrained("models")


while True:
    prompt = input("Enter a prompt: ")
    generated_texts = generate_texts_no_kv(
        model,
        tokenizer,
        [prompt],
        max_length=200
    )
    print(f"Generated text: {generated_texts[0]}\n")
