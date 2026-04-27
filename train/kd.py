import os
import argparse
import copy
import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from model.model import REX
import torch
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig



class RexKDTrainer(SFTTrainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Lock down the teacher model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides the standard HF Trainer compute_loss to inject Teacher Forcing.
        The **kwargs ensures compatibility with the newest Transformers updates.
        """
        labels = inputs.get("labels")

        valid_keys = ["input_ids", "attention_mask", "labels"]
        inputs = {k: v for k, v in inputs.items() if k in valid_keys}

        # 1. Forward Pass: Student (Rex)
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # If Rex mimics HF correctly, it already computed standard Cross-Entropy loss
        # on the tokens where label != -100
        ce_loss = outputs.loss 

        # 2. Forward Pass: Teacher (Mistral)
        with torch.no_grad():
            # Ensure teacher inputs are on the same device as the teacher model
            teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        # 3. Sequence Alignment & Masking
        # Shift logits and labels for next-token prediction
        shift_logits_student = student_logits[..., :-1, :].contiguous()
        shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Isolate ONLY the response tokens (ignore the prompt)
        active_mask = shift_labels != -100
        active_student_logits = shift_logits_student[active_mask]
        active_teacher_logits = shift_logits_teacher[active_mask]

        # 4. Knowledge Distillation Loss (KL Divergence)
        # We divide by temperature to "soften" the distribution
        student_log_probs = F.log_softmax(active_student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(active_teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Scale back up by T^2 to match the gradient magnitude of CE loss
        kl_loss = kl_loss * (self.temperature ** 2)

        # 5. Combine Losses
        loss = (self.alpha * ce_loss) + ((1.0 - self.alpha) * kl_loss)

        # Log the split metrics so you can watch the tug-of-war live
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "teacher_student_balance": (ce_loss / (kl_loss + 1e-8)).item()
            })

        return (loss, outputs) if return_outputs else loss


def format_clean_chatml(example):
    formatted_text = ""
    messages = example.get("messages", [])

    if not messages:
        return {"text": ""} # Return empty string instead of None for Dataset mapping safety

    for message in messages:
        role = message["role"]
        content = message["content"]
        
        # This elegantly handles 'system', 'user', and 'assistant' dynamically
        if role in ["system", "user", "assistant"]:
            formatted_text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

    example["text"] = formatted_text + tokenizer.eos_token
    return example

def format_recast_chatml(example):
    prompt = example.get("winner_prompt", "").strip()
    response = example.get("response_of_winner_prompt", "").strip()

    # Skip bad samples safely
    if not prompt or not response:
        return {"text": ""}

    formatted_text = ""

    # Optional system prompt (VERY recommended for REX)
    formatted_text += "<|im_start|>system\nYou are REX. Follow instructions exactly.\n<|im_end|>\n"

    # User message
    formatted_text += f"<|im_start|>user\n{prompt}\n<|im_end|>\n"

    # Assistant message
    formatted_text += f"<|im_start|>assistant\n{response}\n<|im_end|>\n"

    return {"text": formatted_text + tokenizer.eos_token}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune REX model on an arbitrary dataset")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./rex_finetuned")
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

    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Maynx/Rex-Mistral-KD-Teacher",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" # Keeps inference fast
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    model.fc_out = torch.nn.Linear(model.config.n_embd, len(tokenizer), bias=False)
    model.fc_out.weight.data.copy_(model.embedding.weight.data)

    model.config.max_len = args.max_length
    for block in model.blocks:
        block.attention.generate_sin_cos_pos_emb(model.config.max_len)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=0.05)
    

    dataset = dataset.map(
        format_clean_chatml,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names,
    ).filter(lambda x: x is not None)

    # safe guard usually only Ampere or newer GPUs support bf16 (no T4 or P100)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir="./out",
        max_length=args.max_length,           
        packing=False,                  
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=1000,
        save_strategy="no",
        eval_strategy="epoch",
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="mlflow",
        run_name="REX_SFT_Run",
        dataset_text_field="text"
    )

    trainer = RexKDTrainer(
        teacher_model=teacher_model,
        alpha=0.5,       # Balance: 0.5 means equal weight to Ground Truth and Teacher
        temperature=2.0,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
