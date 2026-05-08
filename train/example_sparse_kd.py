"""
Example: Using Sparse Online KD for REX Training

This script demonstrates programmatic usage of the sparse KD trainer
without relying on command-line arguments.

Usage:
    python train/example_sparse_kd.py
"""

import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig

from model.model import REX
from kd_config import SparseKDConfig
from kd import RexSparseKDTrainer, format_clean_chatml


def example_basic_sparse_kd():
    """
    Example 1: Basic sparse KD (every 2 steps)
    Ideal for: Quick prototyping, small GPUs
    """
    print("=" * 80)
    print("Example 1: Basic Sparse KD (every 2 steps)")
    print("=" * 80)
    
    # Create KD config
    kd_config = SparseKDConfig(
        kd_every_n_steps=2,  # Run KD every 2 steps
        alpha=0.5,           # Equal weight to CE and KD
        temperature=2.0,
        kd_logging_steps=10,
    )
    
    print(f"KD Config: {kd_config}")
    print(f"Expected teacher utilization: ~50%")
    print(f"Expected speedup: ~1.8-2.0x with minimal quality loss\n")


def example_reduced_context_kd():
    """
    Example 2: Long-context training with reduced KD context
    Ideal for: Long-context data, expensive attention
    """
    print("=" * 80)
    print("Example 2: Long-Context KD (4096 train, 2048 teacher)")
    print("=" * 80)
    
    kd_config = SparseKDConfig(
        train_seq_len=4096,          # Student trains on full 4096
        kd_seq_len=2048,             # Teacher only sees last 2048
        crop_strategy="end",         # Crop end of sequence (recent context)
        kd_every_n_steps=1,
        alpha=0.5,
        temperature=2.0,
    )
    
    print(f"KD Config: {kd_config}")
    print(f"Student: trains on 4096 tokens (full sequence)")
    print(f"Teacher: processes 2048 tokens (attention is ~4x faster)")
    print(f"Expected teacher speedup: ~4x per KD step")
    print(f"Expected total speedup: ~3-4x over baseline\n")


def example_dynamic_schedule():
    """
    Example 3: Dynamic KD schedule
    Ideal for: Long training runs, adaptive learning
    """
    print("=" * 80)
    print("Example 3: Dynamic KD Schedule")
    print("=" * 80)
    
    kd_config = SparseKDConfig(
        kd_schedule=[
            (0, 2),        # Steps 0-20k: KD every 2 steps
            (20000, 4),    # Steps 20k-60k: KD every 4 steps
            (60000, 8),    # Steps 60k+: KD every 8 steps
        ],
        alpha=0.5,
        temperature=2.0,
    )
    
    print(f"KD Schedule: {kd_config.kd_schedule}")
    print(f"Step 0: KD frequency = 2 (strong early learning)")
    print(f"Step 20k: KD frequency = 4 (moderate)")
    print(f"Step 60k: KD frequency = 8 (cost reduction phase)")
    print(f"Expected teacher time: ~25% of baseline\n")


def example_lightweight_minillm():
    """
    Example 4: Lightweight MiniLLM-inspired optimizations
    Ideal for: Extreme efficiency, small GPU budgets
    """
    print("=" * 80)
    print("Example 4: Lightweight MiniLLM Features")
    print("=" * 80)
    
    kd_config = SparseKDConfig(
        kd_every_n_steps=2,                    # Sparse scheduling
        kd_topk=64,                            # Only top-64 logits
        entropy_weighting="high_entropy",      # Higher weight on uncertain tokens
        kd_token_subsample_ratio=0.5,          # Every 2nd token
        sequence_weighting="ce_magnitude",     # Weight sequences by difficulty
        alpha=0.5,
        temperature=2.0,
    )
    
    print(f"KD Config: {kd_config}")
    print(f"Features:")
    print(f"  - Sparse KD: every 2 steps (~50% teacher compute)")
    print(f"  - Top-k KD: only distill top-64/50k logits (~95% fewer logit comparisons)")
    print(f"  - Entropy weighting: focus on uncertain predictions")
    print(f"  - Token subsampling: KD every 2nd token (~50% fewer KD computations)")
    print(f"  - Sequence weighting: harder sequences get more KD")
    print(f"Expected combined speedup: ~20-30x over full KD")
    print(f"Expected quality: ~90-95% of full KD quality\n")


def example_production_setup():
    """
    Example 5: Production configuration for real training
    Shows how to create trainer and start training
    """
    print("=" * 80)
    print("Example 5: Production Setup (demonstrates trainer creation)")
    print("=" * 80)
    
    # These would be your real model/data paths
    model_path = "Maynx/Rex-Instruct-v0.1"
    dataset_name = "your_org/instruction_dataset"
    tokenizer_name = "gpt2"
    
    # Production KD config
    kd_config = SparseKDConfig(
        train_seq_len=2048,
        kd_seq_len=1024,              # 50% teacher speedup
        crop_strategy="random",
        kd_every_n_steps=4,           # 75% teacher compute savings
        kd_topk=32,                   # Top-32 only
        entropy_weighting="high_entropy",
        kd_token_subsample_ratio=0.5,
        alpha=0.6,                    # Slightly higher CE weight
        temperature=3.0,              # Softer targets
        kd_logging_steps=100,
        kd_profile_enabled=True,
    )
    
    print(f"Production KD Config:")
    print(f"  - Training: 2048 tokens/sample")
    print(f"  - Teacher KD: 1024 tokens (50% reduction)")
    print(f"  - Sparse: every 4 steps")
    print(f"  - Top-k: 32 logits")
    print(f"  - Token subsample: 50%")
    print(f"  - Alpha: 0.6 (CE-heavy)")
    print(f"\nExpected efficiency:")
    print(f"  - Teacher compute: ~10% of full KD baseline")
    print(f"  - Total training time: ~2-3x faster than full KD")
    print(f"  - Quality: ~95% of full KD quality")
    print(f"\nTo run this setup:")
    print(f"""
python train/kd.py \\
  --model_path "{model_path}" \\
  --dataset_name "{dataset_name}" \\
  --tokenizer_name "{tokenizer_name}" \\
  --train_seq_len 2048 \\
  --kd_seq_len 1024 \\
  --crop_strategy "random" \\
  --kd_every_n_steps 4 \\
  --kd_topk 32 \\
  --entropy_weighting "high_entropy" \\
  --kd_token_subsample_ratio 0.5 \\
  --alpha 0.6 \\
  --temperature 3.0 \\
  --batch_size 8 \\
  --learning_rate 2e-5 \\
  --max_length 2048
    """)


def example_monitoring():
    """
    Example 6: What to look for in logs
    """
    print("=" * 80)
    print("Example 6: Monitoring Sparse KD Training")
    print("=" * 80)
    
    print("""
Key Metrics to Monitor:

1. CE Loss
   - Should decrease steadily
   - If diverging: your model might be too small for task
   - If plateauing: reduce learning rate or increase training data

2. KD Loss (when it runs)
   - Should be smaller than CE loss initially
   - Ratio ce_loss/kd_loss tells you balance
   - If KD >> CE: teacher might be too hard/easy, adjust temperature

3. KD Frequency
   - `kd_enabled`: toggles between 0 and 1
   - `kd_freq`: current KD frequency (changes with schedule)
   - `kd_percent_teacher_steps`: actual % steps with teacher

4. Teacher Profiling
   - `teacher_time_ms_per_kd_step`: teacher forward time
   - Compare with student forward (~80ms typically)
   - If teacher time >> 10x student: check device placement

5. Loss Curves
   - Plot ce_loss, kd_loss, and avg_kd_loss together
   - Look for divergence between ce_loss and kd_loss
   - Stable: both smooth curves
   - Problematic: kd_loss spikes or diverges

Example MLflow Logging (via argparse):

python train/kd.py \\
  --model_path "Maynx/Rex-Instruct-v0.1" \\
  --dataset_name "your_dataset" \\
  --kd_every_n_steps 2 \\
  --kd_logging_steps 50

Then view with:
  mlflow ui --backend-store-uri ./mlruns

In MLflow UI:
  - Look at "Plots" tab: compare ce_loss vs kd_loss
  - Look at "Metrics" tab: check kd_percent_teacher_steps (~50%)
  - Look at "Logs" tab: see all metrics over time
    """)


def main():
    print("\n" + "=" * 80)
    print("Sparse Online KD for REX: Examples and Guidelines")
    print("=" * 80 + "\n")
    
    # Run all examples
    example_basic_sparse_kd()
    example_reduced_context_kd()
    example_dynamic_schedule()
    example_lightweight_minillm()
    example_production_setup()
    example_monitoring()
    
    print("\n" + "=" * 80)
    print("For detailed documentation, see: train/SPARSE_KD_GUIDE.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
