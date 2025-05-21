
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import argparse
from trainer import Trainer
import hydra
from omegaconf import DictConfig
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# TODO: File another way to import the model
# Same problem as in the trainer.py file
CURRENT_DIR = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.model import Transformer



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# Classic dataloading functions
def chunkify(token_list, block_size, stride=None):
    if stride is None:
        stride = block_size
    blocks = []
    for i in range(0, len(token_list) - block_size, stride):
        x = token_list[i : i + block_size]
        y = token_list[i + 1 : i + block_size + 1]
        blocks.append((x, y))
    return blocks

class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        x, y = self.blocks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# === Loading Dataset ===
def load_dataset(file_path: str, train_ratio: int, val_ratio: int, max_length: int) -> tuple[Dataset, Dataset]:
    """
    Load the dataset from a text file and split it into training and validation sets.
    """
    with open(f"{CURRENT_DIR}/{file_path}", 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer.encode(text)
    N = len(tokens)
    train_end = int(train_ratio * N)
    val_end = train_end + int(val_ratio * N)
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    train_blocks = chunkify(train_tokens, max_length)
    val_blocks = chunkify(val_tokens,   max_length)
    print(f"Train blocks: {len(train_blocks)}, Val blocks: {len(val_blocks)}")

    train_ds = BlockDataset(train_blocks)
    val_ds   = BlockDataset(val_blocks)
    return train_ds, val_ds

# === Main Function ===
@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_name = cfg.training.run_name

    # === Model ===
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg.model.n_embd,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_len=cfg.model.max_length
    )

    # === Load Dataset ===
    train_ds, val_ds = load_dataset(
        file_path="train/input.txt",
        train_ratio=cfg.training.train_ratio,
        val_ratio=cfg.training.val_ratio,
        max_length=cfg.model.max_length
    )

    # === Training ===
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        learning_rate=cfg.training.lr,
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        mixed_precision=cfg.training.mixed_precision,
        T_max=cfg.training.epochs,
        max_grad_norm=cfg.training.max_grad_norm,
    )

    trainer.run(
        num_epochs=cfg.training.epochs, 
        run_name=run_name, 
        curent_dir=CURRENT_DIR
    
    )
    trainer.save_model(
        run_name=run_name, 
        path=f"{CURRENT_DIR}/models"
    )

if __name__ == "__main__":
    main()


            
