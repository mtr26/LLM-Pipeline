
import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import tqdm
from transformers import GPT2Tokenizer
from torch.amp import autocast, GradScaler
import math
import os
import sys

# TODO: File another way to import the model
# This is a temporary soloution to import the model, since it is not really a good practice.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.model import Transformer

class Trainer:
    """
    A class to train and validate a Transformer model using PyTorch.
    """
    def __init__(self, 
                model: Transformer,
                train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset,
                tokenizer: GPT2Tokenizer,
                batch_size: int = 32, 
                learning_rate: float = 1e-4,
                mixed_precision: bool = False,
                T_max: int = 10,
                max_grad_norm: float = 1.0,
                ):   
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.scaler = GradScaler() if mixed_precision else None

    def train_one_epoch(self) -> float:
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_train_loss = 0.0

        scaler = self.scaler if self.mixed_precision else None

        for input_ids, targets in self.train_loader:
            input_ids, targets = input_ids.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast(self.device.type):
                    logits = self.model(input_ids)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
            else:
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.mixed_precision:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_train_loss += loss.item()

        return total_train_loss / len(self.train_loader)
    

    def validate(self) -> float:
        """
        Validate the model on the validation set.
        """
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, targets in self.val_loader:  
                input_ids, targets = input_ids.to(self.device), targets.to(self.device)
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                total_val_loss += loss.item()
        avg_val = total_val_loss / len(self.val_loader)
        return avg_val
    
    def save_model(self, run_name: str, path: str = ''):
        """
        Save the model and log it to MLflow.
        """
        torch.save(self.model, f"{path}/model_{run_name}.pth")
        print(f"Model saved to {run_name}.pth")
        

    def log_metrics(self, epoch : int, train_loss : float, val_loss : float):
        """
        Log training and validation metrics to MLflow.
        """
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("epoch", epoch)
        mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)
        mlflow.log_metric("train_perplexity", math.exp(train_loss), step=epoch)
        mlflow.log_metric("val_perplexity", math.exp(val_loss), step=epoch)
        if self.device.type == "cuda":
            mlflow.log_metric("GPU Memory Allocated", torch.cuda.memory_allocated(self.device) / 1024 ** 2, step=epoch)
            mlflow.log_metric("GPU Memory Cached", torch.cuda.memory_reserved(self.device) / 1024 ** 2, step=epoch)


    def run(self, num_epochs: int = 1, run_name: str = "BasicLM", curent_dir: str = '') -> Transformer:
        """
        Run the training and validation process.
        """
        print(f"Training {self.model.__class__.__name__} for {num_epochs} epochs...")
        print(f"Run name: {run_name}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Max Grad Norm: {self.max_grad_norm}")
        print(f"Device: {self.device}")
        eval_interval = max(1, num_epochs // 10)
        mlflow.set_tracking_uri(f"file:///{curent_dir}/mlruns") 
        mlflow.set_experiment(run_name)
        with mlflow.start_run():
            mlflow.run_name = run_name
            mlflow.log_param("num_epoch", num_epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("n_layers", self.model.n_layers)
            mlflow.log_param("n_heads", self.model.n_heads)
            mlflow.log_param("n_embd", self.model.n_embd)
            mlflow.log_param("max_length", self.model.max_length)
            for epoch in tqdm.tqdm(range(num_epochs)):        
                train_loss = self.train_one_epoch()
                val_loss  = self.validate()
                self.log_metrics(epoch, train_loss, val_loss)
                if (epoch + 1) % eval_interval == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            print("Training complete.")
        return self.model
        
    


    