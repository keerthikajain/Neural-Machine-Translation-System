# Simple training launcher for NMT model
import os
import sys
import subprocess
from datetime import datetime

def main():
    """
    Launch NMT training with student-friendly defaults and new optimizations
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
    
    # Original student-friendly configuration (preserved)
    config = {
        # Model size - manageable for student hardware  
        'vocab_size': 32000,
        'd_model': 512, 
        'n_heads': 8,
        'n_layers': 6,
        'max_length': 256,  # Original setting
        
        # Training - conservative settings for learning
        'batch_size': 16,   # Original setting
        'num_epochs': 5,    # Fewer epochs for initial training
        'learning_rate': 0.0001,
        'warmup_steps': 2000,  # Reduced warmup
        
        # NEW: CPU optimization settings
        'save_every_steps': 1000,    # Checkpoint every 1000 steps
        'progress_every_steps': 50,  # Progress every 50 steps
        
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
    print(f"  NEW: Checkpoints every {config['save_every_steps']} steps")
    print(f"  NEW: Progress reports every {config['progress_every_steps']} steps")
    print()
    
    print("Available options:")
    print("1. Start full training (with periodic checkpoints)")
    print("2. Run speed test first, then ask about training") 
    print("3. Exit")
    
    while True:
        choice = input("Choose option [1/2/3]: ").strip()
        
        if choice == '3':
            print("Cancelled.")
            return
        elif choice in ['1', '2']:
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Build base command
    base_cmd = [
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
        '--save_every_steps', str(config['save_every_steps']),
        '--progress_every_steps', str(config['progress_every_steps']),
        '--checkpoint_dir', config['checkpoint_dir']
    ]
    
    if choice == '2':
        # Add speed test only flag
        cmd = base_cmd + ['--speed_test_only', '--speed_test_samples', '1000']
        print("Running speed test only, will exit after showing results...")
    else:
        cmd = base_cmd
        print("Starting full training...")
    
    print("Command:", ' '.join(cmd))
    print()
    print("IMPORTANT: Training will save checkpoints every 1000 steps.")
    print("You can interrupt training (Ctrl+C) and resume later.")
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