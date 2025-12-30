import os
import argparse
import copy
import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model.model import REX
import torch
from trl import SFTTrainer, SFTConfig

def format_ultrachat(example):
    messages = example["messages"]
    prompt = ""
    completion = None
    for m in messages:
        if m["role"] == "assistant":
            completion = m["content"]
            break
        else:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"

    if completion is None:
        return None
    completion = completion + tokenizer.eos_token
    return {
        "prompt": prompt,
        "completion": completion
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune REX model on Dolly 15k.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./dolly15k_rex_finetuned")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    mlflow.set_experiment("REX Fine-tuning")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )


    model = REX.from_pretrained(
        args.model_path,
        device_map=None,
        low_cpu_mem_usage=False
    )

    for block in model.blocks:
        block.attention.generate_sin_cos_pos_emb(model.config.max_len)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    dataset = load_dataset(args.dataset_name, split="train_sft[:60000]")
    dataset = dataset.train_test_split(test_size=0.05)

    dataset = dataset.map(
        format_ultrachat,
        num_proc=os.cpu_count()
    ).filter(lambda x: x is not None)

    # safe guard usually only Ampere or newer GPUs support bf16 (no T4 or P100)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir="./out",
        max_length=args.max_length,           
        packing=False,                  
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=500,
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="mlflow",
        run_name="REX_SFT_Run",
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
