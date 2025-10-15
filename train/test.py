import torch
from model.model import REXConfig, REX

config = REXConfig(
    vocab_size=32768,
    max_len=1024,
    n_layers=16,
    n_heads=16,
    n_kv_heads=4,
    n_embd=1024,
    dropout=0.1,
)


model = REX.from_pretrained("rex-test-model", config=config)

