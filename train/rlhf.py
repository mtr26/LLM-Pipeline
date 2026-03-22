from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from trl import DPOTrainer, DPOConfig
import torch
import argparse
from model.model import REX

def format_chatml(user):
    return f"""<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"""

def format_answer(answer):
    return f"""{answer}\n<|im_end|>"""

def preprocess_uf(sample):
    # UF usually provides prompts as text, or you map the 'messages' list.
    # We inject your strict system identity here to enforce it during DPO.
    user_prompt = sample["prompt"] 
    
    prompt = format_chatml(user_prompt)
    
    # Extract the chosen and rejected text from the UF schema
    chosen_text = sample["chosen"][1]["content"] 
    rejected_text = sample["rejected"][1]["content"]
    
    return {
        "prompt": prompt,
        "chosen": format_answer(chosen_text),
        "rejected": format_answer(rejected_text),
    }

# Map and filter out prompts that are too long
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune REX model on an arbitrary dataset")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    
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

    special_tokens = {
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>"
        ]
    }

    tokenizer.add_special_tokens(special_tokens)


    model = REX.from_pretrained(
        args.model_path,
        device_map=None,
        low_cpu_mem_usage=False
    )

    for block in model.blocks:
        block.attention.generate_sin_cos_pos_emb(model.config.max_len)

    dataset = load_dataset(args.dataset_name, split="train_prefs")

    dataset = dataset.map(preprocess_uf)

    # =========================
    # 4. DPO config (Optimized)
    # =========================
    config = DPOConfig(
        beta=0.1,                  # The KL penalty throttle
        per_device_train_batch_size=args.batch_size, # Scale up for H100
        learning_rate=args.learning_rate,        # Excellent choice
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        max_length=args.max_length,           # Expanded for UF
        max_prompt_length=args.max_length / 2,    # Expanded for UF
        bf16=True,                 # Hardware acceleration
    )

    # =========================
    # 5. Trainer
    # =========================
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL will implicitly copy the active model
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # =========================
    # 6. Train & Save
    # =========================
    trainer.train()
    trainer.save_model("./rex-dpo-final")
    tokenizer.save_pretrained("./rex-dpo-final")