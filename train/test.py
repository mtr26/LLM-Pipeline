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

model.save_pretrained("./rex-model")
