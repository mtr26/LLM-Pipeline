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
    n_layers=16,
    n_heads=16,
    n_kv_heads=4,
    n_embd=1024,
    dropout=0.1,
)
sf_path = "models/model.safetensors"

model = REX(config=config)


training_args = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        learning_rate=5e-5,
        report_to=["mlflow"],
        run_name="REX_Pretraining_Run"

)

dataset_dict = load_dataset("wikitext", "wikitext-2-raw-v1")

if not isinstance(dataset_dict, DatasetDict):
    raise TypeError("Expected load_dataset to return a DatasetDict")

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        #data_collator=data_collator
    )

trainer.save_model("test_save_model", safe_serialization=False)
"""
state_dict = safetensors.torch.load_file(sf_path, device="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)


emb = model.fc_out.weight
print(emb.mean(), emb.std())

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

while True:
    prompt = input("Enter a prompt (or 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break

    generated_texts = generate_texts(model, tokenizer, prompt, max_length=5, top_k=None)
    print(generated_texts)
"""