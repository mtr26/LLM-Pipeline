import os
import argparse
import copy
import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model.model import REX
import torch


def prepare_finetuning_dataset_slimorca(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    test_size: float = 0.10,
):
    # Load SlimOrca dataset
    dataset = load_dataset(dataset_name, split="train")

    def preprocess_and_mask(example):
        # --- FIX: Parse the 'conversations' list ---
        system_prompt = ""
        question = ""
        response_text = ""
        for turn in example["conversations"]:
            if turn.get("from") == "system":
                system_prompt = turn.get("value", "")
            elif turn.get("from") == "human":
                question = turn.get("value", "")
            elif turn.get("from") == "gpt":
                response_text = turn.get("value", "")
        
        # --- Build text template ---
        instruction = f"### Instruction:\n{question.strip()}"
        context = f"\n\n### Input:\n{system_prompt.strip()}" if system_prompt else ""
        response = f"\n\n### Response:\n{response_text.strip()}{tokenizer.eos_token}"

        full_text = instruction + context + response
        prompt_only = instruction + context + "\n\n### Response:\n"

        tokenized_full = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False
        )
        tokenized_prompt = tokenizer(
            prompt_only,
            max_length=max_length,
            truncation=True,
            padding=False
        )

        labels = copy.deepcopy(tokenized_full["input_ids"])
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        tokenized_full["labels"] = labels

        return tokenized_full

    # Preprocess the dataset
    processed_dataset = dataset.map(
        preprocess_and_mask,
        remove_columns=dataset.column_names,
        desc="Tokenizing and masking SlimOrca"
    )

    # Train/test split
    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=42)
    return split_dataset


def custom_data_collator(features):
    """
    Custom collator that pads input_ids with pad_token_id and labels with -100.
    """
    # Find the longest sequence in the batch
    max_length = max(len(f["input_ids"]) for f in features)
        
    input_ids_padded = []
    labels_padded = []
    attention_mask = []

    for f in features:
        # How many padding tokens are needed for this sequence
        pad_length = max_length - len(f["input_ids"])
        # Pad 'input_ids' with the tokenizer's pad token ID
        input_ids_padded.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_length)

        # Pad 'labels' with -100 so they are ignored by the loss function
        labels_padded.append(f["labels"] + [-100] * pad_length)

        # Create the attention mask (1 for real tokens, 0 for padding)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_length)

    # Convert lists to tensors for the model
    return {
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "labels": torch.tensor(labels_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune REX model on SlimOrca.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Open-Orca/SlimOrca")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./slimorca_rex_finetuned")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    args = parser.parse_args()

    mlflow.set_experiment("REX Fine-tuning on SlimOrca")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model = REX.from_pretrained(args.model_path)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    datasets = prepare_finetuning_dataset_slimorca(
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
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
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
        run_name="REX_Continued_FineTuning",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
