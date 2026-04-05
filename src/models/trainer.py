"""
Model training utilities for ADDS
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_init import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    from utils import get_logger, config
except ImportError:
    from src.utils import get_logger, config

logger = get_logger(__name__)


class ADDSTrainer:
    """
    Generic trainer for ADDS models
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device ('cuda' or 'cpu')
            scheduler: Learning rate scheduler
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info(f"✓ Trainer initialized on device: {device}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self._forward_pass(batch)
            targets = batch['target']
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss and metrics dictionary
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._batch_to_device(batch)
                
                outputs = self._forward_pass(batch)
                targets = batch['target']
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets_list = np.array(targets_list)
        
        metrics = self._calculate_metrics(predictions, targets_list)
        
        return avg_loss, metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
        
        Returns:
            Training history
        """
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"✓ Saved best model to {save_path}")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        logger.info("✓ Training completed")
        return history
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def _forward_pass(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass (to be customized based on model)
        Override this method for custom models
        """
        # Default: assumes batch has 'input' key
        if 'input' in batch:
            return self.model(batch['input'])
        else:
            raise NotImplementedError("Override _forward_pass for custom batch structure")
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': np.sqrt(mse)
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"✓ Loaded model from {path}")


def create_trainer(
    model: nn.Module,
    learning_rate: float = None,
    weight_decay: float = 0.0,
    device: str = None
) -> ADDSTrainer:
    """
    Create trainer with default settings from config
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate (defaults from config)
        weight_decay: Weight decay
        device: Device
    
    Returns:
        Configured trainer
    """
    if learning_rate is None:
        learning_rate = config.get('training.learning_rate', 0.001)
    
    criterion = nn.MSELoss()  # Default for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    trainer = ADDSTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    
    return trainer
