"""
Sparse Online Knowledge Distillation Configuration and Scheduling.

Supports:
- Sparse KD frequency (skip teacher forwards on non-KD steps)
- Dynamic KD schedules (change frequency over training phases)
- Reduced-context KD (teacher sees shorter sequences)
- Lightweight MiniLLM-inspired features (top-k, entropy-aware, subsampling, weighting)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch


@dataclass
class SparseKDConfig:
    """Configuration for sparse online KD scheduling."""
    
    # ============ Sparse KD Scheduling ============
    # Default: run KD every step (backward compatible)
    kd_every_n_steps: int = 1
    
    # Dynamic schedule: list of (step_threshold, kd_frequency) tuples
    # Example: [(0, 2), (20000, 4), (60000, 8)]
    # Means: steps 0-20k run KD every 2 steps, 20k-60k every 4, after 60k every 8
    kd_schedule: Optional[List[Tuple[int, int]]] = None
    
    # ============ Reduced-Context KD ============
    # Training sequence length (full)
    train_seq_len: int = 4096
    
    # KD sequence length (cropped for teacher)
    # If < train_seq_len, teacher only processes cropped region
    kd_seq_len: int = 4096
    
    # Crop offset strategy: 'start', 'end', 'random', 'center'
    # 'start': crop from beginning (first kd_seq_len tokens)
    # 'end': crop from end (last kd_seq_len tokens)
    # 'center': crop from middle
    # 'random': random offset (deterministic per step if seed set)
    crop_strategy: str = "start"
    
    # ============ KD Loss Weighting ============
    # Base KD weight: loss = alpha * ce_loss + (1-alpha) * kd_loss
    alpha: float = 0.5
    
    # Temperature for softening logits
    temperature: float = 2.0
    
    # ============ Lightweight MiniLLM Features ============
    
    # A. Top-k KD: only distill top-k logits
    # None = full vocab, 32 = only top 32 logits per position
    kd_topk: Optional[int] = None
    
    # B. Entropy-aware KD: scale KD weight by teacher entropy
    # None = constant weight
    # 'high_entropy': higher KD weight on uncertain (high-entropy) positions
    # 'low_entropy': higher KD weight on confident (low-entropy) positions
    entropy_weighting: Optional[str] = None
    
    # C. Token subsampling: KD only on subset of positions
    # 1.0 = all tokens, 0.5 = every other token, etc.
    kd_token_subsample_ratio: float = 1.0
    
    # D. Sequence weighting: scale KD contribution per sequence
    # None = constant, 'ce_magnitude' = by CE loss magnitude
    # 'entropy' = by average entropy, 'random' = random weights
    sequence_weighting: Optional[str] = None
    
    # ============ Logging & Profiling ============
    # Log KD metrics every N steps
    kd_logging_steps: int = 100
    
    # Enable detailed KD profiling (teacher time, etc.)
    kd_profile_enabled: bool = True
    
    # ============ Future Extensibility Hooks ============
    # Placeholder for MiniLLM rollout, speculative decoding, etc.
    future_features: dict = field(default_factory=dict)
    
    def validate(self):
        """Validate configuration sanity."""
        assert self.kd_every_n_steps >= 1, "kd_every_n_steps must be >= 1"
        assert self.alpha >= 0 and self.alpha <= 1, "alpha must be in [0, 1]"
        assert self.temperature > 0, "temperature must be positive"
        assert 0 < self.kd_token_subsample_ratio <= 1.0, "subsample_ratio must be in (0, 1]"
        assert self.kd_seq_len <= self.train_seq_len, "kd_seq_len must be <= train_seq_len"
        assert self.crop_strategy in ['start', 'end', 'center', 'random'], \
            f"crop_strategy must be 'start', 'end', 'center', or 'random', got {self.crop_strategy}"
        if self.entropy_weighting is not None:
            assert self.entropy_weighting in ['high_entropy', 'low_entropy'], \
                f"entropy_weighting must be 'high_entropy' or 'low_entropy', got {self.entropy_weighting}"
        if self.sequence_weighting is not None:
            assert self.sequence_weighting in ['ce_magnitude', 'entropy', 'random'], \
                f"sequence_weighting must be 'ce_magnitude', 'entropy', or 'random', got {self.sequence_weighting}"


class SparseKDScheduler:
    """
    Manages sparse KD scheduling during training.
    Determines when to run teacher forward passes.
    """
    
    def __init__(self, config: SparseKDConfig):
        self.config = config
        config.validate()
        
        self._step_count = 0
        self._teacher_forward_count = 0
        self._teacher_forward_ms = 0.0
        self._kd_tokens_processed = 0
    
    def should_run_teacher(self, global_step: int) -> bool:
        """
        Determine if teacher forward should run at this step.
        
        Args:
            global_step: Current training step
            
        Returns:
            bool: True if teacher forward should run
        """
        kd_freq = self._get_kd_frequency(global_step)
        return global_step % kd_freq == 0
    
    def _get_kd_frequency(self, global_step: int) -> int:
        """
        Get KD frequency for a given step.
        Supports dynamic schedules.
        """
        if self.config.kd_schedule is None:
            return self.config.kd_every_n_steps
        
        # Find the appropriate frequency from schedule
        freq = self.config.kd_every_n_steps
        for step_threshold, kd_freq in self.config.kd_schedule:
            if global_step >= step_threshold:
                freq = kd_freq
            else:
                break
        
        return freq
    
    def get_crop_indices(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get cropping indices for reduced-context KD.
        
        Args:
            seq_len: Full sequence length
            batch_size: Batch size
            device: Device to place tensors on
            
        Returns:
            Tensor of shape (batch_size, kd_seq_len) with indices to select
        """
        if self.config.kd_seq_len >= seq_len:
            # No cropping needed
            return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        kd_len = self.config.kd_seq_len
        
        if self.config.crop_strategy == 'start':
            # Crop from beginning
            indices = torch.arange(kd_len, device=device)
            return indices.unsqueeze(0).expand(batch_size, -1)
        
        elif self.config.crop_strategy == 'end':
            # Crop from end
            start_idx = seq_len - kd_len
            indices = torch.arange(start_idx, seq_len, device=device)
            return indices.unsqueeze(0).expand(batch_size, -1)
        
        elif self.config.crop_strategy == 'center':
            # Crop from center
            start_idx = (seq_len - kd_len) // 2
            indices = torch.arange(start_idx, start_idx + kd_len, device=device)
            return indices.unsqueeze(0).expand(batch_size, -1)
        
        elif self.config.crop_strategy == 'random':
            # Random contiguous offset (shared across the batch)
            max_offset = seq_len - kd_len
            offset = torch.randint(0, max_offset + 1, (1,), device=device).item()
            indices = torch.arange(offset, offset + kd_len, device=device)
            return indices.unsqueeze(0).expand(batch_size, -1)
    
    def get_token_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get token subsampling mask.
        
        Args:
            seq_len: Sequence length
            device: Device to place tensors on
            
        Returns:
            Boolean tensor of shape (seq_len,) indicating which tokens to use for KD
        """
        if self.config.kd_token_subsample_ratio >= 1.0:
            return torch.ones(seq_len, dtype=torch.bool, device=device)
        
        # Create deterministic mask based on ratio
        mask_size = max(1, int(seq_len * self.config.kd_token_subsample_ratio))
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        # Sample indices uniformly
        step_size = seq_len / mask_size
        indices = torch.arange(mask_size, device=device)
        sampled_indices = (indices * step_size).long()
        sampled_indices = torch.clamp(sampled_indices, 0, seq_len - 1)
        
        mask[sampled_indices] = True
        return mask
    
    def compute_entropy_weights(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute entropy-based weights for KD.
        
        Args:
            logits: Tensor of shape (B*L, V) where B*L is batch*seq_len, V is vocab
            temperature: Temperature for softmax
            
        Returns:
            Tensor of shape (B*L,) with entropy weights
        """
        if self.config.entropy_weighting is None:
            return torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
        
        # Compute probability distribution
        probs = torch.softmax(logits / temperature, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        # Avoid log(0) with small epsilon
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=logits.dtype, device=logits.device))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        if self.config.entropy_weighting == 'high_entropy':
            # Higher weight on uncertain (high-entropy) predictions
            return normalized_entropy
        else:  # 'low_entropy'
            # Higher weight on confident (low-entropy) predictions
            return 1.0 - normalized_entropy
    
    def compute_sequence_weights(self, ce_losses: torch.Tensor, strategy: Optional[str] = None) -> torch.Tensor:
        """
        Compute per-sequence weights for KD.
        
        Args:
            ce_losses: CE loss per sample, shape (B,)
            strategy: Weighting strategy or use config
            
        Returns:
            Tensor of shape (B,) with per-sequence weights
        """
        strategy = strategy or self.config.sequence_weighting
        
        if strategy is None:
            return torch.ones_like(ce_losses)
        
        if strategy == 'ce_magnitude':
            # Weight by CE loss magnitude
            # Normalize to [0.5, 1.5] to avoid extreme weights
            mean_loss = ce_losses.mean()
            std_loss = ce_losses.std() + 1e-8
            normalized = (ce_losses - mean_loss) / std_loss
            return 0.5 + 0.5 * torch.tanh(normalized)
        
        elif strategy == 'entropy':
            # Weight by per-sample entropy (average over tokens)
            # This would need token-level entropies as input
            # For now, use uniform
            return torch.ones_like(ce_losses)
        
        elif strategy == 'random':
            # Random weights for exploration (debug purposes)
            return torch.rand_like(ce_losses)
        
        return torch.ones_like(ce_losses)
    
    def apply_topk_kd(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask logits to keep only top-k.
        
        Args:
            student_logits: Shape (*, V)
            teacher_logits: Shape (*, V)
            
        Returns:
            Tuple of masked (student_logits, teacher_logits)
        """
        if self.config.kd_topk is None:
            return student_logits, teacher_logits
        
        topk = min(self.config.kd_topk, teacher_logits.shape[-1])
        teacher_topk_values, teacher_topk_indices = torch.topk(teacher_logits, topk, dim=-1)
        student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)

        return student_topk_logits, teacher_topk_values
    
    def record_step(self, global_step: int, teacher_forward_ran: bool, teacher_time_ms: float = 0.0, kd_tokens_processed: int = 0):
        """Record step for profiling."""
        self._step_count += 1
        if teacher_forward_ran:
            self._teacher_forward_count += 1
            self._teacher_forward_ms += teacher_time_ms
            self._kd_tokens_processed += kd_tokens_processed
    
    def get_statistics(self) -> dict:
        """Get profiling statistics."""
        if self._step_count == 0:
            return {}
        
        percent_with_teacher = (self._teacher_forward_count / self._step_count) * 100
        skipped_teacher_steps = self._step_count - self._teacher_forward_count
        average_teacher_forward_ms = (
            self._teacher_forward_ms / self._teacher_forward_count
            if self._teacher_forward_count > 0 else 0.0
        )
        effective_kd_ratio = self._teacher_forward_count / self._step_count
        effective_tokens_per_second = (
            self._kd_tokens_processed / (self._teacher_forward_ms / 1000.0)
            if self._teacher_forward_ms > 0 else 0.0
        )
        return {
            "total_steps": self._step_count,
            "steps_with_teacher": self._teacher_forward_count,
            "actual_teacher_steps": self._teacher_forward_count,
            "skipped_teacher_steps": skipped_teacher_steps,
            "percent_steps_with_teacher": percent_with_teacher,
            "average_teacher_forward_ms": average_teacher_forward_ms,
            "effective_kd_ratio": effective_kd_ratio,
            "kd_tokens_processed": self._kd_tokens_processed,
            "effective_tokens_per_second": effective_tokens_per_second,
        }
