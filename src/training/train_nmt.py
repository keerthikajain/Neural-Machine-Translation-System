# Main training script for Neural Machine Translation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import json

# Import our modules
from src.model import create_transformer_model
from src.data.dataset import create_data_loaders, create_validation_files
from src.training.trainer import NMTTrainer

def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def create_training_config(args):
    """Create training configuration dictionary"""
    config = {
        'model': {
            'vocab_size': args.vocab_size,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'd_ff': args.d_ff,
            'max_seq_length': args.max_length,
            'dropout': args.dropout
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'warmup_steps': args.warmup_steps,
            'num_epochs': args.num_epochs,
            'label_smoothing': args.label_smoothing,
            'max_length': args.max_length
        },
        'data': {
            'train_english': args.train_english,
            'train_hindi': args.train_hindi,
            'val_english': args.val_english, 
            'val_hindi': args.val_hindi,
            'tokenizer_path': args.tokenizer_path
        },
        'system': {
            'device': str(args.device),
            'checkpoint_dir': args.checkpoint_dir,
            'timestamp': datetime.now().isoformat()
        }
    }
    return config

def main():
    parser = argparse.ArgumentParser(description='Train Neural Machine Translation model')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')  
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Peak learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    
    # Data paths
    parser.add_argument('--train_english', type=str, default='data/processed/clean_train_en.txt',
                       help='Training English file')
    parser.add_argument('--train_hindi', type=str, default='data/processed/clean_train_hi.txt',  
                       help='Training Hindi file')
    parser.add_argument('--val_english', type=str, default='data/processed/val_en.txt',
                       help='Validation English file')
    parser.add_argument('--val_hindi', type=str, default='data/processed/val_hi.txt',
                       help='Validation Hindi file')  
    parser.add_argument('--tokenizer_path', type=str, default='data/processed/tokenizer/sp_model.model',
                       help='SentencePiece tokenizer path')
    
    # System parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=0, help='Data loader workers')
    
    args = parser.parse_args()
    
    # Setup device
    args.device = setup_device()
    
    print("=" * 80)
    print("NEURAL MACHINE TRANSLATION TRAINING")
    print("=" * 80)
    
    # Create validation files if they don't exist
    if not os.path.exists(args.val_english) or not os.path.exists(args.val_hindi):
        print("Creating validation files from original dataset...")
        create_validation_files()
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader = create_data_loaders(
        train_english_file=args.train_english,
        train_hindi_file=args.train_hindi,
        val_english_file=args.val_english,
        val_hindi_file=args.val_hindi,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_transformer_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    total_params, trainable_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = NMTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        vocab_size=args.vocab_size,
        label_smoothing=args.label_smoothing,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save training configuration
    config = create_training_config(args)
    config_path = os.path.join(args.checkpoint_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved: {config_path}")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\nStarting training...")
    print("-" * 50)
    
    try:
        trainer.train(
            num_epochs=args.num_epochs,
            save_every=1,  # Save every epoch
            early_stopping_patience=5
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        interrupt_path = os.path.join(args.checkpoint_dir, 'interrupted_checkpoint.pt')
        trainer.save_checkpoint(interrupt_path)
        print(f"Checkpoint saved: {interrupt_path}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        # Save current state for debugging
        error_path = os.path.join(args.checkpoint_dir, 'error_checkpoint.pt')
        trainer.save_checkpoint(error_path)
        print(f"Error checkpoint saved: {error_path}")
        raise
    
    print("\nTraining script completed!")

if __name__ == "__main__":
    main()