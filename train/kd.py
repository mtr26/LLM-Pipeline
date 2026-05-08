import os
import argparse
import copy
import time
import mlflow
from typing import Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from model.model import REX
import torch
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig

from kd_config import SparseKDConfig, SparseKDScheduler


class RexSparseKDTrainer(SFTTrainer):
    def __init__(
        self,
        teacher_model,
        kd_config: SparseKDConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Lock down the teacher model
        self.teacher_device = self.args.device
        self.teacher_model = teacher_model.to(self.teacher_device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        if hasattr(self.teacher_model, "config"):
            self.teacher_model.config.use_cache = False
        
        # KD configuration and scheduler
        self.kd_config = kd_config
        self.kd_scheduler = SparseKDScheduler(kd_config)
        
        # Profiling
        self.teacher_time_ms = 0.0
        self.kd_loss_sum = 0.0
        self.ce_loss_sum = 0.0
        self.kd_steps_count = 0
        self.actual_teacher_steps = 0
        self.skipped_teacher_steps = 0
        self.kd_tokens_processed = 0
        self.total_compute_loss_calls = 0

    def _get_crop_bounds(self, seq_len: int, global_step: int) -> Tuple[int, int]:
        """Return a contiguous KD crop range [start, end)."""
        kd_len = min(self.kd_config.kd_seq_len, seq_len)
        if kd_len >= seq_len:
            return 0, seq_len

        max_offset = seq_len - kd_len
        if self.kd_config.crop_strategy == "start":
            start = 0
        elif self.kd_config.crop_strategy == "end":
            start = max_offset
        elif self.kd_config.crop_strategy == "center":
            start = max_offset // 2
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(global_step) + 1337)
            start = int(torch.randint(0, max_offset + 1, (1,), generator=generator).item())

        return start, start + kd_len

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Sparse online KD with lightweight MiniLLM features.
        
        - Teacher forward only runs on KD steps
        - Supports reduced-context KD
        - Includes top-k, entropy-aware, token subsampling, sequence weighting
        """
        self.total_compute_loss_calls += 1
        labels = inputs.get("labels")
        valid_keys = ["input_ids", "attention_mask", "labels"]
        inputs = {k: v for k, v in inputs.items() if k in valid_keys}

        # 1. Forward Pass: Student (always runs)
        outputs = model(**inputs)
        student_logits = outputs.logits
        ce_loss = outputs.loss

        # 2. Determine if teacher should run on this step
        should_run_teacher = self.kd_scheduler.should_run_teacher(self.state.global_step)

        if should_run_teacher:
            # Time the teacher forward pass
            if self.teacher_device.type == "cuda":
                torch.cuda.synchronize(self.teacher_device)
            t_start = time.time()
            
            # Apply reduced-context cropping if needed
            teacher_inputs, crop_start, crop_end = self._prepare_teacher_inputs(inputs)
            student_kd_logits = student_logits[:, crop_start:crop_end, :]
            kd_labels = labels[:, crop_start:crop_end] if labels is not None else None
            
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=teacher_inputs["input_ids"],
                    attention_mask=teacher_inputs.get("attention_mask"),
                )
                teacher_logits = teacher_outputs.logits
            if self.teacher_device.type == "cuda":
                torch.cuda.synchronize(self.teacher_device)
            
            teacher_time_ms = (time.time() - t_start) * 1000.0
            self.teacher_time_ms += teacher_time_ms
            
            # 3. Compute KD loss
            kd_loss, kd_tokens_processed = self._compute_kd_loss(
                student_kd_logits,
                teacher_logits,
                kd_labels,
            )
            
            # 4. Combine losses
            if kd_tokens_processed > 0:
                loss = (self.kd_config.alpha * ce_loss) + ((1.0 - self.kd_config.alpha) * kd_loss)
            else:
                loss = ce_loss
            
            # Record for logging
            self.ce_loss_sum += ce_loss.item()
            self.kd_loss_sum += kd_loss.item()
            self.kd_steps_count += 1
            self.actual_teacher_steps += 1
            self.kd_tokens_processed += kd_tokens_processed
        else:
            # Non-KD steps: just use CE loss
            loss = ce_loss
            kd_loss = torch.tensor(0.0, device=loss.device)
            teacher_time_ms = 0.0
            kd_tokens_processed = 0
            self.skipped_teacher_steps += 1

        # 5. Log metrics
        self._log_kd_metrics(ce_loss, kd_loss, should_run_teacher, teacher_time_ms, kd_tokens_processed)

        return (loss, outputs) if return_outputs else loss

    def _prepare_teacher_inputs(self, inputs: dict) -> Tuple[dict, int, int]:
        """Prepare contiguous teacher inputs and return crop bounds."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        seq_len = input_ids.shape[1]
        crop_start, crop_end = self._get_crop_bounds(seq_len, self.state.global_step)

        teacher_inputs = {
            "input_ids": input_ids[:, crop_start:crop_end].to(self.teacher_device),
        }
        if attention_mask is not None:
            teacher_inputs["attention_mask"] = attention_mask[:, crop_start:crop_end].to(self.teacher_device)

        return teacher_inputs, crop_start, crop_end

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute KD loss with lightweight MiniLLM features.
        """
        if labels is None:
            return torch.tensor(0.0, device=student_logits.device), 0

        # Shift for next-token prediction
        shift_logits_student = student_logits[..., :-1, :].contiguous()
        shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Get active mask (only response tokens)
        active_mask = shift_labels != -100
        if not active_mask.any().item():
            return torch.tensor(0.0, device=student_logits.device), 0

        active_student_logits = shift_logits_student[active_mask]
        active_teacher_logits = shift_logits_teacher[active_mask]

        # A. Apply top-k KD if configured
        if self.kd_config.kd_topk is not None:
            active_student_logits, active_teacher_logits = self.kd_scheduler.apply_topk_kd(
                active_student_logits, active_teacher_logits
            )

        # B. Compute entropy weights if configured
        if self.kd_config.entropy_weighting is not None:
            entropy_weights = self.kd_scheduler.compute_entropy_weights(
                active_teacher_logits,
                temperature=self.kd_config.temperature
            )
        else:
            entropy_weights = None

        # C. Apply token subsampling if configured
        if self.kd_config.kd_token_subsample_ratio < 1.0:
            token_mask = self.kd_scheduler.get_token_mask(
                active_student_logits.shape[0],
                active_student_logits.device
            )
            active_student_logits = active_student_logits[token_mask]
            active_teacher_logits = active_teacher_logits[token_mask]
            if entropy_weights is not None:
                entropy_weights = entropy_weights[token_mask]

        if active_student_logits.numel() == 0:
            return torch.tensor(0.0, device=student_logits.device), 0

        # Compute KL divergence
        student_log_probs = F.log_softmax(
            active_student_logits / self.kd_config.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            active_teacher_logits / self.kd_config.temperature, dim=-1
        )

        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'
        )

        # Apply entropy weighting
        kl_per_token = kl_div.sum(dim=-1)
        if entropy_weights is not None:
            kl_per_token = kl_per_token * entropy_weights

        kl_loss = kl_per_token.mean()

        # Scale by temperature^2
        kl_loss = kl_loss * (self.kd_config.temperature ** 2)

        return kl_loss, int(active_student_logits.shape[0])

    def _log_kd_metrics(self, ce_loss: torch.Tensor, kd_loss: torch.Tensor, teacher_ran: bool, teacher_time_ms: float, kd_tokens_processed: int):
        """Log KD-specific metrics."""
        self.kd_scheduler.record_step(
            self.state.global_step,
            teacher_ran,
            teacher_time_ms=teacher_time_ms,
            kd_tokens_processed=kd_tokens_processed,
        )
        
        if self.state.global_step % self.kd_config.kd_logging_steps == 0:
            logs = {
                "ce_loss": ce_loss.item(),
                "kd_enabled": int(teacher_ran),
                "kd_freq": self.kd_scheduler._get_kd_frequency(self.state.global_step),
                "actual_teacher_steps": self.actual_teacher_steps,
                "skipped_teacher_steps": self.skipped_teacher_steps,
                "average_teacher_forward_ms": (
                    self.teacher_time_ms / self.actual_teacher_steps if self.actual_teacher_steps > 0 else 0.0
                ),
                "effective_kd_ratio": (
                    self.actual_teacher_steps / max(self.total_compute_loss_calls, 1)
                ),
                "kd_tokens_processed": self.kd_tokens_processed,
                "effective_tokens_per_second": (
                    self.kd_tokens_processed / (self.teacher_time_ms / 1000.0)
                    if self.teacher_time_ms > 0 else 0.0
                ),
            }
            
            if teacher_ran and self.kd_steps_count > 0:
                logs["kd_loss"] = kd_loss.item()
                logs["avg_kd_loss"] = self.kd_loss_sum / self.kd_steps_count
                logs["avg_ce_loss"] = self.ce_loss_sum / self.kd_steps_count
            
            if self.kd_config.kd_profile_enabled:
                stats = self.kd_scheduler.get_statistics()
                if stats:
                    logs.update({
                        f"kd_percent_teacher_steps": stats.get("percent_steps_with_teacher", 0),
                        "actual_teacher_steps": stats.get("actual_teacher_steps", self.actual_teacher_steps),
                        "skipped_teacher_steps": stats.get("skipped_teacher_steps", self.skipped_teacher_steps),
                        "average_teacher_forward_ms": stats.get("average_teacher_forward_ms", 0.0),
                        "effective_kd_ratio": stats.get("effective_kd_ratio", 0.0),
                        "kd_tokens_processed": stats.get("kd_tokens_processed", self.kd_tokens_processed),
                        "effective_tokens_per_second": stats.get("effective_tokens_per_second", 0.0),
                    })
                    if stats.get("average_teacher_forward_ms", 0.0) > 0:
                        logs["teacher_time_ms_per_kd_step"] = stats.get("average_teacher_forward_ms", 0.0)
            
            self.log(logs)


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
    parser = argparse.ArgumentParser(description="Sparse Online KD for REX model")
    
    # Model and dataset
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./rex_kd_finetuned")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Sparse KD scheduling
    parser.add_argument("--kd_every_n_steps", type=int, default=1,
                        help="Run KD every N steps (1=every step, 2=every other, etc.)")
    parser.add_argument("--kd_schedule", type=str, default=None,
                        help="Dynamic KD schedule as comma-separated pairs, e.g., '0:2,20000:4,60000:8'")
    
    # Reduced-context KD
    parser.add_argument("--train_seq_len", type=int, default=1024,
                        help="Training sequence length")
    parser.add_argument("--kd_seq_len", type=int, default=1024,
                        help="KD sequence length (if < train_seq_len, teacher sees cropped context)")
    parser.add_argument("--crop_strategy", type=str, default="start",
                        choices=["start", "end", "center", "random"],
                        help="How to crop sequences for teacher")
    
    # KD loss weighting
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Balance weight: alpha*CE + (1-alpha)*KD")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Softening temperature for logits")
    
    # Lightweight MiniLLM features
    parser.add_argument("--kd_topk", type=int, default=None,
                        help="Only distill top-k logits per position")
    parser.add_argument("--entropy_weighting", type=str, default=None,
                        choices=["high_entropy", "low_entropy"],
                        help="Scale KD by teacher entropy")
    parser.add_argument("--kd_token_subsample_ratio", type=float, default=1.0,
                        help="Subsample fraction of tokens for KD (0.5 = every other token)")
    parser.add_argument("--sequence_weighting", type=str, default=None,
                        choices=["ce_magnitude", "entropy", "random"],
                        help="Per-sequence KD weighting strategy")
    
    # Logging
    parser.add_argument("--kd_logging_steps", type=int, default=100,
                        help="Log KD metrics every N steps")
    parser.add_argument("--kd_profile_enabled", action="store_true", default=True,
                        help="Enable KD profiling and teacher timing")
    
    args = parser.parse_args()

    # Parse dynamic schedule if provided
    kd_schedule = None
    if args.kd_schedule:
        kd_schedule = []
        for pair in args.kd_schedule.split(","):
            step, freq = pair.split(":")
            kd_schedule.append((int(step), int(freq)))

    # Create KD config
    kd_config = SparseKDConfig(
        kd_every_n_steps=args.kd_every_n_steps,
        kd_schedule=kd_schedule,
        train_seq_len=args.train_seq_len,
        kd_seq_len=args.kd_seq_len,
        crop_strategy=args.crop_strategy,
        alpha=args.alpha,
        temperature=args.temperature,
        kd_topk=args.kd_topk,
        entropy_weighting=args.entropy_weighting,
        kd_token_subsample_ratio=args.kd_token_subsample_ratio,
        sequence_weighting=args.sequence_weighting,
        kd_logging_steps=args.kd_logging_steps,
        kd_profile_enabled=args.kd_profile_enabled,
    )

    mlflow.set_experiment("REX Sparse KD Training")

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
        device_map=None,
        attn_implementation="flash_attention_2"
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    model.fc_out = torch.nn.Linear(model.config.n_embd, len(tokenizer), bias=False)
    model.fc_out.weight.data.copy_(model.embedding.weight.data)

    model.config.max_len = args.max_length
    for block in model.blocks:
        block.attention.generate_sin_cos_pos_emb(model.config.max_len)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"KD Config: kd_every_n_steps={args.kd_every_n_steps}, kd_seq_len={args.kd_seq_len}, alpha={args.alpha}")

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=0.05)

    dataset = dataset.map(
        format_clean_chatml,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names,
    ).filter(lambda x: x is not None)

    # Safe guard: check BF16 support
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir="./out",
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        eval_strategy="epoch",
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="constant_with_warmup",
        report_to="mlflow",
        run_name="REX_Sparse_KD_Run",
        dataset_text_field="text"
    )

    trainer = RexSparseKDTrainer(
        teacher_model=teacher_model,
        kd_config=kd_config,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    
    # Log final KD statistics
    final_stats = trainer.kd_scheduler.get_statistics()
    if final_stats:
        print(f"\nFinal KD Statistics: {final_stats}")
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
