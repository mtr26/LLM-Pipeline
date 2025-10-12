import torch
import math
from model.model import REX, REXConfig
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import argparse


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
    parser = argparse.ArgumentParser(description="Debug REX model with a small batch.")
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    # === CONFIG ===
    dataset_file_path = args.dataset_file_path
    tokenizer_name = args.tokenizer_name
    max_length = args.max_length
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # === LOAD DATA ===
    datasets, tokenizer = load_and_tokenize_datasets(
        dataset_file_path=dataset_file_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )

    # Take a few samples from the training set
    train_data = datasets["train"].select(range(4))
    train_data = datasets["train"].select(range(4))

# Convert to plain torch tensors manually
    input_ids = torch.tensor(train_data["input_ids"], dtype=torch.long)
    labels = torch.tensor(train_data["labels"], dtype=torch.long)

    # Handle padding (since you use unk as pad)
    labels[labels == tokenizer.pad_token_id] = -100

    print(f"Batch shape: {input_ids.shape}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Max token ID: {input_ids.max().item()}")
    print(f"Min token ID: {input_ids.min().item()}")


    # === INIT MODEL ===
    config = REXConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=max_length,
        n_layers=2,        # smaller for debugging
        n_heads=4,
        n_kv_heads=2,
        n_embd=128,
        dropout=0.1,
    )
    model = REX(config).to(device)
    model.eval()


    # === DEBUG FORWARD PASS ===
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), labels=labels.to(device))

    logits = outputs.logits
    loss = outputs.loss

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Count how many tokens are actually contributing to loss
    mask = labels != -100
    valid_tokens = mask.sum().item()
    total_tokens = labels.numel()

    print(f"Valid (non-pad) tokens: {valid_tokens}/{total_tokens} "
        f"({valid_tokens / total_tokens:.2%})")

    print(f"Perplexity ≈ {math.exp(loss.item()):.2f}")
