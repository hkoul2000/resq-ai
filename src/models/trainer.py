"""Training pipeline for ResQNet flood prediction models.

Supports:
- Single model training
- Deep ensemble training (M independent models)
- MLflow tracking
- Mixed precision training
- Early stopping
- Checkpoint saving/loading
- Smoke mode for quick testing
"""
import os
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import mlflow

from src.models.resqnet import ResQNet

logger = logging.getLogger(__name__)

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Get optimizer based on config."""
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    
    # Filter out parameters that do not require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer_name = config.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = config.get("momentum", 0.9)
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def warmup_cosine_schedule(optimizer: optim.Optimizer, warmup_steps: int, total_steps: int) -> optim.lr_scheduler.LambdaLR:
    """Create a schedule with a learning rate that decreases following the values of the cosine function."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], num_training_steps: int) -> Optional[optim.lr_scheduler.LRScheduler]:
    """Get learning rate scheduler."""
    scheduler_name = config.get("scheduler", "cosine").lower()
    warmup_epochs = config.get("warmup_epochs", 0)
    epochs = config.get("epochs", 10)
    
    warmup_steps = int((warmup_epochs / epochs) * num_training_steps) if epochs > 0 else 0
    
    if scheduler_name == "cosine":
        return warmup_cosine_schedule(optimizer, warmup_steps, num_training_steps)
    elif scheduler_name == "step":
        step_size = int((config.get("step_epochs", 30) / epochs) * num_training_steps)
        gamma = config.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None

class BCEWithDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute basic classification/segmentation metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    tp = intersection
    fp = preds.sum().item() - tp
    fn = targets.sum().item() - tp
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: Union[str, torch.device]):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.smoke = config.get("project", {}).get("smoke", False)
        
        self.optimizer = get_optimizer(self.model, config.get("training", {}))
        
        # Loss function
        loss_name = config.get("training", {}).get("loss", "bce_dice")
        if loss_name == "bce_dice":
            dice_weight = config.get("training", {}).get("dice_weight", 0.5)
            self.criterion = BCEWithDiceLoss(bce_weight=1.0-dice_weight, dice_weight=dice_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.mixed_precision = config.get("training", {}).get("mixed_precision", False) and str(device) != "cpu"
        self.scaler = GradScaler(device.type) if self.mixed_precision else None
        self.grad_clip = config.get("training", {}).get("gradient_clip", 1.0)
        
        self.checkpoint_dir = Path(config.get("project", {}).get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = config.get("training", {}).get("early_stopping_patience", 10)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int, scheduler: Optional[optim.lr_scheduler.LRScheduler] = None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = self.to_device(batch)
            targets = batch.get("label", batch.get("mask"))
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.mixed_precision:
                with autocast(device_type=self.device.type):
                    outputs = self.model(batch)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    loss = self.criterion(logits, targets)
                
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = self.criterion(logits, targets)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
            if scheduler is not None:
                scheduler.step()
                
            total_loss += loss.item()
            
            if self.smoke and batch_idx >= 1:
                break
                
        return {"train_loss": total_loss / (batch_idx + 1 if self.smoke else num_batches)}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_metrics = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self.to_device(batch)
                targets = batch.get("label", batch.get("mask"))
                
                if self.mixed_precision:
                    with autocast(device_type=self.device.type):
                        outputs = self.model(batch)
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                        loss = self.criterion(logits, targets)
                else:
                    outputs = self.model(batch)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    loss = self.criterion(logits, targets)
                    
                total_loss += loss.item()
                
                metrics = compute_metrics(logits, targets)
                for k, v in metrics.items():
                    all_metrics[k] += v
                    
                if self.smoke and batch_idx >= 1:
                    break
                    
        num_batches = batch_idx + 1
        res = {"val_loss": total_loss / num_batches}
        for k in all_metrics:
            res[f"val_{k}"] = all_metrics[k] / num_batches
            
        return res

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        epochs = self.config.get("training", {}).get("epochs", 100)
        if self.smoke:
            epochs = min(epochs, 2)
            
        num_training_steps = epochs * len(train_loader)
        scheduler = get_scheduler(self.optimizer, self.config.get("training", {}), num_training_steps)
        
        best_metrics = {}
        
        mlflow_enabled = self.config.get("mlflow", {}).get("enabled", True)
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch, scheduler)
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch+1}/{epochs} - " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            
            if mlflow_enabled:
                mlflow.log_metrics(metrics, step=epoch)
                
            val_loss = metrics["val_loss"]
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_metrics = metrics.copy()
                self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Load best model for final return
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            self.load_checkpoint(best_path)
            
        return best_metrics

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class EnsembleTrainer:
    """Trains M independent models with different seeds."""
    def __init__(self, config: Dict[str, Any], device: Union[str, torch.device]):
        self.config = config
        self.device = device
        self.ensemble_size = config.get("uq", {}).get("ensemble_size", 5)
        self.checkpoint_dir = Path(config.get("project", {}).get("checkpoint_dir", "checkpoints")) / "ensemble"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_ensemble(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, float]]:
        ensemble_results = []
        
        for i in range(self.ensemble_size):
            logger.info(f"Training ensemble member {i+1}/{self.ensemble_size}")
            
            # Set seed for this member
            seed = self.config.get("project", {}).get("seed", 42) + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            # Create model
            model = ResQNet(self.config).to(self.device)
            
            trainer = Trainer(model, self.config, self.device)
            
            # Override checkpoint dir for this member
            member_dir = self.checkpoint_dir / f"member_{i}"
            member_dir.mkdir(exist_ok=True)
            trainer.checkpoint_dir = member_dir
            
            metrics = trainer.fit(train_loader, val_loader)
            ensemble_results.append(metrics)
            
        return ensemble_results
