# Training utilities for Neural Machine Translation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
import os
from typing import Dict, Optional, Tuple
from datetime import datetime
import json

class NMTTrainer:
    """
    Trainer class for Neural Machine Translation model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: torch.device,
                 learning_rate: float = 0.0001, warmup_steps: int = 4000,
                 vocab_size: int = 32000, label_smoothing: float = 0.1,
                 checkpoint_dir: str = "checkpoints"):
        """
        Args:
            model: Transformer model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            device: Device to train on (cuda/cpu)
            learning_rate: Peak learning rate for scheduler (0.0001 for student project)
            warmup_steps: Learning rate warmup steps (4000 typical)
            vocab_size: Vocabulary size for loss calculation (32000)
            label_smoothing: Label smoothing factor (0.1 typical)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.vocab_size = vocab_size
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(
            num_classes=vocab_size,
            smoothing=label_smoothing,
            ignore_index=0  # Ignore PAD token
        )
        
        # Optimizer - Adam with specific parameters for Transformer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),  # Standard for Transformer
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Learning rate scheduler - Warmup + Cosine decay
        self.scheduler = TransformerLRScheduler(
            optimizer=self.optimizer,
            d_model=model.encoder.embedding.token_embedding.d_model,
            warmup_steps=warmup_steps
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Batch size: {train_loader.batch_size}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step
        
        Args:
            batch: Batch of training data
            
        Returns:
            float: Training loss for this step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        src_tokens = batch['src_tokens'].to(self.device)
        tgt_input = batch['tgt_input'].to(self.device)
        tgt_output = batch['tgt_output'].to(self.device)
        
        # Forward pass
        logits = self.model(src_tokens, tgt_input)
        
        # Calculate loss
        # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = tgt_output.view(-1)
        
        loss = self.criterion(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        
        return loss.item()
    
    def validate(self) -> float:
        """
        Run validation loop
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                src_tokens = batch['src_tokens'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Forward pass
                logits = self.model(src_tokens, tgt_input)
                
                # Calculate loss
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = tgt_output.view(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple[float, float]: (average_train_loss, validation_loss)
        """
        start_time = time.time()
        total_train_loss = 0.0
        num_batches = len(self.train_loader)
        
        print(f"Epoch {self.epoch + 1} - Training...")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Training step
            train_loss = self.train_step(batch)
            total_train_loss += train_loss
            
            # Log progress every 100 steps
            if (batch_idx + 1) % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                
                print(f"  Step {self.step:6d} | "
                      f"Batch {batch_idx + 1:4d}/{num_batches} | "
                      f"Loss: {train_loss:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {elapsed:.1f}s")
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches
        
        # Validation
        print("  Validating...")
        val_loss = self.validate()
        
        # Update tracking
        self.train_losses.append(avg_train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        epoch_time = time.time() - start_time
        
        print(f"  Epoch {self.epoch + 1} complete:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss: {val_loss:.4f}")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    LR: {self.learning_rates[-1]:.6f}")
        
        return avg_train_loss, val_loss
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"    ✅ New best model saved: {best_path}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"  Resuming from epoch {self.epoch + 1}, step {self.step}")
    
    def train(self, num_epochs: int, save_every: int = 1, 
             early_stopping_patience: int = 5) -> None:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train one epoch
            train_loss, val_loss = self.train_epoch()
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f'checkpoint_epoch_{epoch + 1}.pt'
                )
                self.save_checkpoint(checkpoint_path, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.epoch + 1,
            'total_steps': self.step
        }
        
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {history_path}")


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss
    Helps with overfitting and calibration in NMT models
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, 
                 ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, num_classes] predicted logits
            targets: [N] target class indices
            
        Returns:
            torch.Tensor: Smoothed cross entropy loss
        """
        # Create mask for padding tokens
        mask = (targets != self.ignore_index)
        
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Apply mask
        logits = logits[mask]
        targets = targets[mask]
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        # (1 - smoothing) for correct class, smoothing / (num_classes - 1) for others
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate loss
        loss = -torch.sum(true_dist * log_probs, dim=-1).mean()
        
        return loss


class TransformerLRScheduler:
    """
    Learning rate scheduler for Transformer model
    Implements warmup + inverse square root decay
    """
    
    def __init__(self, optimizer: optim.Optimizer, d_model: int, 
                 warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        """Update learning rate"""
        self.step_num += 1
        
        # Calculate learning rate
        lr = self._calculate_lr()
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _calculate_lr(self) -> float:
        """Calculate learning rate based on step"""
        step = max(self.step_num, 1)  # Avoid division by zero
        
        # Transformer learning rate schedule
        lr = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        
        return lr
    
    def get_last_lr(self) -> list:
        """Get current learning rate"""
        return [self._calculate_lr()]
    
    def state_dict(self) -> dict:
        """Get scheduler state"""
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'step_num': self.step_num
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state"""
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.step_num = state_dict['step_num']


if __name__ == "__main__":
    # Test label smoothing loss
    print("Testing Label Smoothing Cross Entropy...")
    
    vocab_size = 1000
    batch_size = 32
    seq_len = 20
    
    criterion = LabelSmoothingCrossEntropy(vocab_size, smoothing=0.1, ignore_index=0)
    
    # Random logits and targets
    logits = torch.randn(batch_size * seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # Add some padding tokens
    targets[targets < 50] = 0  # Make some tokens padding
    
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Test learning rate scheduler
    print("\nTesting Learning Rate Scheduler...")
    
    model = torch.nn.Linear(512, vocab_size)
    optimizer = optim.Adam(model.parameters())
    scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=1000)
    
    print("Learning rate progression:")
    for step in range(0, 2000, 200):
        scheduler.step_num = step
        lr = scheduler._calculate_lr()
        print(f"Step {step:4d}: LR = {lr:.6f}")