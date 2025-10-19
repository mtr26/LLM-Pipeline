import os
import argparse
import copy
import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model.model import REX
import torch


def prepare_finetuning_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    test_size: float = 0.10,
):
    # Load Dolly 15k dataset
    dataset = load_dataset(dataset_name, split="train")

    def preprocess_and_mask(example):
        # --- Build text template ---
        instruction = f"### Instruction:\n{example.get('instruction', '').strip()}"
        context = f"\n\n### Input:\n{example.get('context', '').strip()}" if example.get("context") else ""
        response = f"\n\n### Response:\n{example.get('response', '').strip()}{tokenizer.eos_token}"

        # Full sequence: instruction + (optional) context + response
        full_text = instruction + context + response

        # --- Tokenize ---
        tokenized_full = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        # Prompt without the answer (so we can mask its loss)
        prompt_only = instruction + context + "\n\n### Response:\n"
        tokenized_prompt = tokenizer(prompt_only, max_length=max_length, truncation=True)

        # --- Mask labels ---
        labels = copy.deepcopy(tokenized_full["input_ids"])
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len  # Ignore loss before response

        tokenized_full["labels"] = labels
        return tokenized_full

    # Preprocess dataset
    processed_dataset = dataset.map(
        preprocess_and_mask,
        remove_columns=dataset.column_names,
        desc="Tokenizing and masking Dolly 15k"
    )

    # Split into train/test
    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=42)
    return split_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune REX model on Dolly 15k.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./dolly15k_rex_finetuned")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    mlflow.set_experiment("REX Fine-tuning")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model = REX.from_pretrained(args.model_path)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    datasets = prepare_finetuning_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_steps=50,
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=bf16,
        fp16=not bf16,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name="REX_Dolly15k_FineTuning",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
