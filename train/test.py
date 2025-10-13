import torch
from model.model import REXConfig, REX

config = REXConfig(
    vocab_size=50200,
    max_len=1024,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    n_embd=768,
    dropout=0.1,
)

model = REX(config)

dummy_input = torch.randint(0, config.vocab_size, (1, 128))  # Batch size of 1, sequence length of 128
outputs = model(dummy_input)


model.save_pretrained("./rex-model")


model.from_pretrained("./rex-model")

outputs2 = model(dummy_input)

print((outputs.logits != outputs2.logits).sum())