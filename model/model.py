import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


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


class FlashAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, dropout: int = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scaling = embed_dim ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, past_k=None, past_v=None):
        B, T_in, D = x.shape
        H, head_dim = self.num_heads, D // self.num_heads

        # project & split
        qkv = self.qkv_proj(x).view(B, T_in, H, 3, head_dim)
        q, k, v = qkv[...,0,:], qkv[...,1,:], qkv[...,2,:]    # each (B, T_in, H, head_dim)
        q, k, v = [t.permute(0,2,1,3) for t in (q,k,v)]       # now (B, H, T_in, head_dim)


        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        T_k = k.size(2)    # total key/value length
        T_q = q.size(2)    # query length

        if x.device.type == 'cuda':
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout.p,
                    is_causal=True,
                    scale=self.scaling
                )
        else:
            mask = torch.triu(
                torch.full((T_q, T_k), -1e9, device=x.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0) 

            scores = torch.matmul(q, k.transpose(-2,-1)) * self.scaling 
            scores = scores + mask
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        T_out = attn.size(2) 
        attn = attn.permute(0,2,1,3).reshape(B, T_out, D)

        out = self.out_proj(self.dropout(attn))
        return out, (k, v)
    

    
class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout:int = 0.1):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.w2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, 4 * n_embd, bias=False)

    def forward(self, x : torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class Block(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, dropout : int =0.1):
        super(Block, self).__init__()
        self.attention = FlashAttention(n_heads, n_embd, dropout) 
        self.ff = MLP(n_embd, dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x: torch.Tensor, cache=None):
        attn_out, present = self.attention(
            self.ln1(x),
            past_k=cache[0] if cache else None,
            past_v=cache[1] if cache else None
            )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, present
    

class Transformer(nn.Module):
    def __init__(self,
                n_layers: int, 
                n_heads: int, 
                n_embd: int, 
                vocab_size: int, 
                max_len:int = 5000, 
                dropout:int = 0.1,
                kv_cacheing: bool = False):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = PositionalEncoding(n_embd, max_len)
        self.blocks = nn.Sequential(*[Block(n_heads, n_embd, dropout) for _ in range(n_layers)])
        self.ln_f = RMSNorm(n_embd)
        self.fc_out = nn.Linear(n_embd, vocab_size)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.max_length = max_len
        self.kv_cacheing = kv_cacheing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights according to module type."""
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)

    def forward(self, x : torch.Tensor):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            if self.training:
                # explicit use_reentrant flag silences the warning too
                x, _ = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x, _ = block(x) 
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids : torch.Tensor, max_new_tokens : int):
        """
        Generate new tokens given an input sequence, using KV-caching.
        """
        B, T = input_ids.shape
        device = input_ids.device

        caches = [ (None, None) for _ in range(self.n_layers) ]

        x = self.embedding(input_ids)
        x = self.positional_encoding(x, offset=0)
        for i, block in enumerate(self.blocks):
            if self.kv_cacheing:
                x, present = block(x, cache=caches[i])
                caches[i] = present  
            else:
                x, _ = block(x)

        generated = input_ids
        for step in range(max_new_tokens):
            last_token = generated[:, -1:].to(device)
            offset     = generated.shape[1] - 1

            x = self.embedding(last_token)
            x = self.positional_encoding(x, offset=offset)

            for i, block in enumerate(self.blocks):
                if self.kv_cacheing:
                    x, present = block(x, cache=caches[i])
                    caches[i] = present
                else:
                    x, _ = block(x)

            x = self.ln_f(x)
            logits = self.fc_out(x)
            next_token = torch.argmax(logits[:, -1:], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.squeeze(0)], dim=1)

        return generated