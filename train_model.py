# Simple training launcher for NMT model
import os
import sys
import subprocess
from datetime import datetime

def main():
    """
    Launch NMT training with student-friendly defaults
    """
    print("=" * 60)
    print("SAARTHI NMT MODEL TRAINING")
    print("English → Hindi Neural Machine Translation")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        'data/processed/clean_train_en.txt',
        'data/processed/clean_train_hi.txt', 
        'data/processed/tokenizer/sp_model.model'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run data preprocessing and tokenizer training first.")
        return
    
    # Student-friendly configuration
    config = {
        # Model size - manageable for student hardware
        'vocab_size': 32000,
        'd_model': 512, 
        'n_heads': 8,
        'n_layers': 6,
        'max_length': 256,  # Shorter sequences for faster training
        
        # Training - conservative settings for learning
        'batch_size': 16,   # Smaller batch size for limited GPU memory
        'num_epochs': 5,    # Fewer epochs for initial training
        'learning_rate': 0.0001,
        'warmup_steps': 2000,  # Reduced warmup
        
        # System
        'checkpoint_dir': f'checkpoints/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("Training Configuration:")
    print(f"  Model: {config['d_model']}-dim, {config['n_layers']} layers, {config['n_heads']} heads")
    print(f"  Vocabulary: {config['vocab_size']} tokens")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Max length: {config['max_length']}")
    print(f"  Checkpoint dir: {config['checkpoint_dir']}")
    print()
    
    # Ask for confirmation
    response = input("Start training? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Build command
    cmd = [
        sys.executable, 'src/training/train_nmt.py',
        '--vocab_size', str(config['vocab_size']),
        '--d_model', str(config['d_model']),
        '--n_heads', str(config['n_heads']),
        '--n_layers', str(config['n_layers']),
        '--max_length', str(config['max_length']),
        '--batch_size', str(config['batch_size']),
        '--num_epochs', str(config['num_epochs']),
        '--learning_rate', str(config['learning_rate']),
        '--warmup_steps', str(config['warmup_steps']),
        '--checkpoint_dir', config['checkpoint_dir']
    ]
    
    print("Starting training...")
    print("Command:", ' '.join(cmd))
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error running training: {e}")

if __name__ == "__main__":
    main()