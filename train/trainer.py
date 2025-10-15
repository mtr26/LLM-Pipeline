import os
import argparse
import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from model.model import REXConfig, REX
from safetensors import safe_open as safetensors_open
from safetensors.torch import load as safetensors_load
import torch


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
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,
        n_embd=1024,
        dropout=0.1,
    )

    if args.model_path:
        print(f"Loading REX weights from {args.model_path}")
        config = REXConfig.from_pretrained(args.model_path)
        model = REX(config)  # build model with real tensors

        # --- Find checkpoint weights ---
        sf_path = os.path.join(args.model_path, "model.safetensors")

        if os.path.exists(sf_path):
            print("Loading weights from .safetensors file...")
            state_dict = safetensors_load(sf_path, device="cpu")
        else:
            # --- Handle sharded checkpoints ---
            sharded_paths = sorted(
                [os.path.join(args.model_path, f) for f in os.listdir(args.model_path) if f.startswith("pytorch_model-")]
            )
            if len(sharded_paths) > 0:
                print(f"Loading {len(sharded_paths)} sharded weight files...")
                state_dict = {}
                for path in sharded_paths:
                    chunk = torch.load(path, map_location="cpu")
                    state_dict.update(chunk)
            else:
                raise FileNotFoundError(f"No weight files found in {args.model_path}")

        # --- Load weights ---
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        print("Any meta params?", any(p.is_meta for p in model.parameters()))
        lr = 5e-6
    else:
        model = REX(config=config)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    print("Any meta params?", any(p.is_meta for p in model.parameters()))
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        bf16=True,
        learning_rate=5e-5 if args.model_path is None else lr,
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
        #data_collator=data_collator
    )



    trainer.train()
    trainer.save_model(args.output_dir)
