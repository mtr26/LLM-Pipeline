
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from typing import Optional, Tuple, List

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class REXConfig(PretrainedConfig):
    model_type = "REX"
    def __init__(
        self,
        vocab_size: int = 50257,
        max_len: int = 1024,
        n_layers: int = 12,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        n_embd: int = 768,
        dropout: float = 0.1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_embd = n_embd
        self.dropout = dropout
        super().__init__(**kwargs)


def scaled_dot_product_attention_grouped_flash(
        queries: torch.Tensor,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        scale: float, 
        is_causal: bool = False,
        dropout_p: float = 0.0,
        mask: torch.Tensor = None
        ) -> torch.Tensor:
    """
    Compute scaled dot-product attention with grouped queries.
    
    Args:
        queries (torch.Tensor): Query tensor of shape (B, T_q, C).
        keys (torch.Tensor): Key tensor of shape (B, T_k, C).
        values (torch.Tensor): Value tensor of shape (B, T_v, C).
        scale (float): Scaling factor for the dot product.
        
    Returns:
        torch.Tensor: Output tensor after applying attention.
    """
    q = queries.permute(0, 2, 1, 3)
    k = keys.permute(0, 2, 1, 3)
    v = values.permute(0, 2, 1, 3)

    bq, hq, nq, dq = q.shape
    bk, hk, nk, dk = k.shape
    bv, hv, nv, dv = v.shape

    repeat = hq // hk
    k = k.repeat_interleave(repeat, dim=1)  # (B, hq, Tk, d)
    v = v.repeat_interleave(repeat, dim=1)  # (B, hq, Tv, d)

    with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )
    out = out.permute(0, 2, 1, 3)

    return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, offset: int = 0):
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device) + offset
        positions = positions.unsqueeze(0).expand(B, T)
        return x + self.pos_embedding(positions)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class GroupedQueryAttention(nn.Module):
    """
    Borrowed from my own REX implementation, this is why it also supports cross attention and RoPE.
    Originally based on the "Return of the Encoder: Maximizing Parameter Efficiency for SLMs" paper.
    For those interested, I made a little change using RSNorm instead of LayerNorm.
    """
    def __init__(self, 
                dim: int, 
                k_dim: int, 
                kv_heads: int, 
                query_heads: int, 
                max_length: int, 
                dropout: int = 0.1, 
                is_causal: bool = False, 
                apply_rotary: bool = True
                ):
        super().__init__()
        assert dim % query_heads == 0, "dim must be divisible by query_heads"
        self.dim = dim
        self.kv_heads = kv_heads
        self.query_heads = query_heads
        self.is_causal = is_causal
        self.max_length = max_length
        kv_dim = (dim // query_heads) * kv_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(k_dim, kv_dim)
        self.v_proj = nn.Linear(k_dim, kv_dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim // query_heads

        self.apply_rotary = apply_rotary
        self.scale = self.head_dim**-0.5

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
    # Transpose to [B, H, L, D] for RoPE rotation
        q = q.permute(0, 2, 1, 3)  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3)  # [B, H_kv, L, D]
    
        # Apply rotary embeddings
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
    
        # Transpose back to original layout [B, L, H, D]
        q_embed = q_embed.permute(0, 2, 1, 3)
        k_embed = k_embed.permute(0, 2, 1, 3)
    
        return q_embed, k_embed

    @torch._dynamo.disable()
    def generate_sin_cos_pos_emb(self, seq_len, device, rope_theta=10000, rope_factor=8.0):
        base, rope_factor, dim, max_seq_len = (
            rope_theta,
            rope_factor,
            self.head_dim,
            self.max_length
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        if rope_factor > 1.0:
            seq_len_eff = max(seq_len, max_seq_len)
            base_adjustment = ((rope_factor * seq_len_eff / max_seq_len) - (rope_factor - 1)) ** (dim / (dim - 2))
            adjusted_base = base * base_adjustment
            inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, dim, 2, device=device).float() / dim))

        position_ids = torch.arange(seq_len, device=device, dtype=torch.float)
        if not self.is_causal:
            position_ids = position_ids - ((seq_len - 1) // 2)
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return cos_emb, sin_emb

    def forward(
            self,
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            mask: torch.Tensor = None,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
        ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        bq, nq, dq = q.shape
        bk, nk, dk = k.shape
        bv, nv, dv = v.shape
        
        q = q.view(bq, nq, self.query_heads, dq // self.query_heads)
        k = k.view(bk, nk, self.kv_heads, dk // self.kv_heads)
        v = v.view(bv, nv, self.kv_heads, dv // self.kv_heads)

        if self.apply_rotary:
            cos_emb, sin_emb = self.generate_sin_cos_pos_emb(nq, device=q.device)
            cos_emb = cos_emb.to(q.device)
            sin_emb = sin_emb.to(q.device)
            q, k = self.apply_rotary_pos_emb(q, k, cos_emb, sin_emb)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=1)  # (B, T_past+T, kv_heads, head_dim)
            v = torch.cat([pv, v], dim=1)

        new_kv = (k, v) if use_cache else None

        out = scaled_dot_product_attention_grouped_flash(q, k, v, self.scale, self.is_causal, mask=mask)
        out = out.reshape(out.size(0), out.size(1), out.size(2) * out.size(3))  # Flatten the heads
        out = self.out_proj(out)
        return out, new_kv


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout:int = 0.1):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.w2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, 4 * n_embd, bias=False)

    def forward(self, x : torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config: REXConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(
            dim=config.n_embd, k_dim=config.n_embd, 
            kv_heads=config.n_kv_heads,
            query_heads=config.n_heads, max_length=config.max_len, 
            dropout=config.dropout, is_causal=True, apply_rotary=True
        )
        self.ff = MLP(config.n_embd, config.dropout)
        self.ln1, self.ln2 = RMSNorm(config.n_embd), RMSNorm(config.n_embd)
    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attention(self.ln1(x), self.ln1(x), self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv

class REX(PreTrainedModel):
    config_class = REXConfig
    def __init__(self, config: REXConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.n_embd)
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)

    def forward(
            self, 
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
        ) -> CausalLMOutputWithPast:
        
        x = self.embedding(input_ids)

        new_past_key_values = [] if use_cache else None
        
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i]
            if self.training:
                x, new_kv = checkpoint.checkpoint(block, x, past_kv, use_cache, use_reentrant=False)
            else:
                x, new_kv = block(x, past_kv, use_cache)
            if use_cache:
                new_past_key_values.append(new_kv)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values if use_cache else None,
        )

@torch.no_grad()
def generate_texts(model, tokenizer, prompts, max_length=50, temperature=1.0, top_k=None):
    model.eval()
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(next(model.parameters()).device)
    
    past_key_values = None
    for _ in range(max_length):
        logits, past_key_values = model(input_ids, use_cache=True, past_key_values=past_key_values)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(1, indices, values)
            probs = F.softmax(probs, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)
        input_ids = next_token  # only feed last token in next step

    return tokenizer.batch_decode(torch.cat([tokenizer(prompts).input_ids, input_ids], dim=1), skip_special_tokens=True)



config = REXConfig(
    vocab_size=50257,
    max_len=1024,
    n_layers=2,
    n_heads=8,
    n_kv_heads=2,
    n_embd=128,
    dropout=0.1
)


