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
from contextlib import nullcontext

class NMTTrainer:
    """
    Trainer class for Neural Machine Translation model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: torch.device,
                 learning_rate: float = 0.0001, warmup_steps: int = 4000,
                 vocab_size: int = 32000, label_smoothing: float = 0.1,
                 checkpoint_dir: str = "checkpoints",
                 save_every_steps: int = 1000, progress_every_steps: int = 50,
                 use_amp: bool = False):
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
            save_every_steps: Save checkpoint every N training steps (1000 default)
            progress_every_steps: Report progress every N training steps (50 default)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # Opt-in CUDA AMP; CPU and existing training behavior remain unchanged.
        self.use_amp = bool(use_amp and device.type == 'cuda')
        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.vocab_size = vocab_size
        self.checkpoint_dir = checkpoint_dir
        self.save_every_steps = save_every_steps
        self.progress_every_steps = progress_every_steps
        
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
        self.learning_rates_after_optimizer_steps = []
        self.warmup_steps = warmup_steps
        self._resume_epoch = None
        
        # Timing for progress reporting
        self.training_start_time = None
        self.epoch_start_time = None
        self.step_times = []
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Periodic checkpoints every: {save_every_steps} steps")
        print(f"  Progress reports every: {progress_every_steps} steps")
        
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
        
        # Autocast only the forward/loss; keep gradient clipping in FP32.
        context = torch.autocast('cuda', dtype=torch.float16) if self.use_amp else nullcontext()
        with context:
            logits = self.model(src_tokens, tgt_input)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                  tgt_output.reshape(-1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # GradScaler can skip an optimizer step on overflow.
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() >= scale_before:
            self.scheduler.step()
        actual_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates_after_optimizer_steps.append(actual_lr)
        
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
                
                context = torch.autocast('cuda', dtype=torch.float16) if self.use_amp else nullcontext()
                with context:
                    logits = self.model(src_tokens, tgt_input)
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                          tgt_output.reshape(-1))

                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train_epoch(self, speed_test_mode: bool = False, speed_test_samples: int = 1000) -> Tuple[float, float]:
        """
        Train for one epoch with periodic checkpointing and detailed progress reporting
        
        Args:
            speed_test_mode: If True, train only on speed_test_samples for performance measurement
            speed_test_samples: Number of samples to use in speed test mode
            
        Returns:
            Tuple[float, float]: (average_train_loss, validation_loss)
        """
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        
        if self.training_start_time is None:
            self.training_start_time = self.epoch_start_time
            
        total_train_loss = 0.0
        num_batches = len(self.train_loader)
        
        if speed_test_mode:
            # Calculate how many batches we need for speed_test_samples
            samples_per_batch = self.train_loader.batch_size
            max_batches = min(num_batches, (speed_test_samples + samples_per_batch - 1) // samples_per_batch)
            print(f"SPEED TEST MODE: Training on ~{max_batches * samples_per_batch} samples ({max_batches} batches)")
            num_batches = max_batches
        
        print(f"Epoch {self.epoch + 1} - Training...")
        
        batch_times = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            if speed_test_mode and batch_idx >= num_batches:
                break
                
            batch_start_time = time.time()
            
            # Training step
            train_loss = self.train_step(batch)
            total_train_loss += train_loss
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Progress reporting every N steps
            if self.step % self.progress_every_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                total_elapsed = time.time() - self.training_start_time
                epoch_elapsed = time.time() - self.epoch_start_time
                
                # Calculate processing speed
                recent_times = batch_times[-10:] if len(batch_times) >= 10 else batch_times
                avg_batch_time = sum(recent_times) / len(recent_times) if recent_times else 0
                samples_per_sec = self.train_loader.batch_size / avg_batch_time if avg_batch_time > 0 else 0
                
                print(f"  Step {self.step:6d} | "
                      f"Batch {batch_idx + 1:4d}/{num_batches} | "
                      f"Loss: {train_loss:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Speed: {samples_per_sec:.1f} samples/sec | "
                      f"Epoch time: {epoch_elapsed:.1f}s | "
                      f"Total time: {total_elapsed:.1f}s")
            
            # Periodic checkpointing during epoch
            if not speed_test_mode and self.step % self.save_every_steps == 0:
                checkpoint_path = self.save_periodic_checkpoint()
                print(f"  Periodic checkpoint saved: {os.path.basename(checkpoint_path)}")
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches
        
        # Speed test mode: calculate and report performance metrics
        if speed_test_mode:
            total_batch_time = sum(batch_times)
            avg_batch_time = total_batch_time / len(batch_times)
            total_samples = num_batches * self.train_loader.batch_size
            samples_per_sec = total_samples / total_batch_time
            
            print(f"  SPEED TEST RESULTS:")
            print(f"     Samples processed: {total_samples}")
            print(f"     Total time: {total_batch_time:.1f}s")
            print(f"     Avg batch time: {avg_batch_time:.2f}s")
            print(f"     Processing speed: {samples_per_sec:.1f} samples/sec")
            
            # Estimate full training time
            full_dataset_size = len(self.train_loader.dataset)
            estimated_epoch_time = (full_dataset_size / samples_per_sec) / 3600  # in hours
            print(f"     Estimated time per full epoch: {estimated_epoch_time:.1f} hours")
            
            # Skip validation in speed test mode
            return avg_train_loss, 0.0
        
        # Validation (not in speed test mode)
        print("  Validating...")
        val_start_time = time.time()
        val_loss = self.validate()
        val_time = time.time() - val_start_time
        
        # Update tracking
        self.train_losses.append(avg_train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        epoch_time = time.time() - self.epoch_start_time
        
        print(f"  Epoch {self.epoch + 1} complete:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss: {val_loss:.4f}")
        print(f"    Epoch Time: {epoch_time:.1f}s")
        print(f"    Val Time: {val_time:.1f}s")
        print(f"    LR: {self.learning_rates[-1]:.6f}")
        
        # Reset epoch start time for next epoch
        self.epoch_start_time = None
        
        return avg_train_loss, val_loss
    
    def save_periodic_checkpoint(self, suffix: str = None) -> str:
        """
        Save periodic checkpoint during training (safe overwrite strategy)
        
        Args:
            suffix: Optional suffix for checkpoint filename
            
        Returns:
            str: Path to saved checkpoint
        """
        if suffix:
            checkpoint_name = f'checkpoint_step_{self.step}_{suffix}.pt'
        else:
            checkpoint_name = f'checkpoint_step_{self.step}.pt'
            
        filepath = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint = self._create_checkpoint_payload(epoch_completed=False)
        
        # Save to temporary file first, then rename (atomic operation)
        temp_filepath = filepath + '.tmp'
        torch.save(checkpoint, temp_filepath)
        
        # On Windows, need to handle existing file
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        self._cleanup_old_checkpoints()
        return filepath

    def _create_checkpoint_payload(self, epoch_completed: bool = True) -> dict:
        """Create the complete state needed to resume training."""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'learning_rates_after_optimizer_steps': self.learning_rates_after_optimizer_steps,
            'warmup_steps': self.warmup_steps,
            'architecture': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else None,
            'timestamp': datetime.now().isoformat(),
            'epoch_completed': epoch_completed,
            'best_val_loss': self.best_val_loss,
            'amp_scaler_state_dict': self.scaler.state_dict(),
        }

    def _cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Keep the newest periodic checkpoints and remove older ones."""
        import glob
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_step_*.pt')
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for old_checkpoint in checkpoints[keep_last:]:
            os.remove(old_checkpoint)

    def save_checkpoint(self, filepath: str, is_best: bool = False, epoch_completed: bool = True) -> None:
        """Save one complete checkpoint and optionally update best_model.pt."""
        checkpoint = self._create_checkpoint_payload(epoch_completed=epoch_completed)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"     New best model saved: {best_path}")
    
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
        if 'amp_scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.learning_rates_after_optimizer_steps = checkpoint.get('learning_rates_after_optimizer_steps', [])
        self.warmup_steps = checkpoint.get('warmup_steps', self.warmup_steps)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        completed = checkpoint.get('epoch_completed', True)
        self._resume_epoch = checkpoint['epoch'] + 1 if completed else checkpoint['epoch']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"  Resuming from epoch {self._resume_epoch + 1}, step {self.step}")
    
    def speed_test(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Run speed test on a subset of training data to measure performance
        
        Args:
            num_samples: Number of training samples to use for speed test
            
        Returns:
            Dict with speed test metrics
        """
        print(f"Running speed test with {num_samples} samples...")
        
        # Save original state
        original_epoch = self.epoch
        original_step = self.step
        
        # Run speed test epoch
        self.epoch = 0  # Reset for clean metrics
        self.step = 0
        
        train_loss, _ = self.train_epoch(speed_test_mode=True, speed_test_samples=num_samples)
        
        # Calculate metrics from the test
        samples_per_batch = self.train_loader.batch_size
        batches_processed = min(len(self.train_loader), (num_samples + samples_per_batch - 1) // samples_per_batch)
        actual_samples = batches_processed * samples_per_batch
        
        # Restore original state
        self.epoch = original_epoch
        self.step = original_step
        
        return {
            'samples_processed': actual_samples,
            'batches_processed': batches_processed,
            'train_loss': train_loss
        }

    def train(self, num_epochs: int, save_every: int = 1, 
             early_stopping_patience: int = 5, speed_test_first: bool = False, 
             speed_test_samples: int = 1000) -> None:
        """
        Main training loop with optional speed test
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            speed_test_first: Run speed test before full training
            speed_test_samples: Number of samples for speed test
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Periodic checkpoints every {self.save_every_steps} steps")
        print(f"Progress reports every {self.progress_every_steps} steps")
        
        # Optional speed test
        if speed_test_first:
            self.speed_test(speed_test_samples)
            print("\nSpeed test complete. Starting full training...")
            print("-" * 60)
        
        patience_counter = 0
        
        start_epoch = self._resume_epoch if self._resume_epoch is not None else self.epoch
        if self._resume_epoch is not None:
            print(f"Continuing training from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, num_epochs):
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
        # Equivalent to the original smoothing (including off-target classes),
        # without allocating a full [tokens, vocabulary] target matrix.
        mask = (targets != self.ignore_index)
        logits = logits[mask]
        targets = targets[mask]
        if targets.numel() == 0:
            return logits.sum() * 0.0

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        other = (-log_probs.sum(dim=-1) - nll) / (self.num_classes - 1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * other).mean()


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