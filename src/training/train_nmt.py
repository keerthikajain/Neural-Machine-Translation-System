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
            'dropout': args.dropout,
            'architecture': args.architecture,
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

def analyze_sequence_lengths(train_english_file: str, train_hindi_file: str, tokenizer_path: str):
    """Analyze sequence lengths in training data to understand truncation impact"""
    import sentencepiece as spm
    
    print("Analyzing sequence lengths in training data...")
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    en_lengths = []
    hi_lengths = []
    
    # Sample 10,000 examples for analysis (to avoid loading entire dataset)
    sample_size = 10000
    
    with open(train_english_file, 'r', encoding='utf-8') as ef, \
         open(train_hindi_file, 'r', encoding='utf-8') as hf:
        
        for i, (en_line, hi_line) in enumerate(zip(ef, hf)):
            if i >= sample_size:
                break
                
            # Tokenize and get lengths
            en_tokens = sp.encode_as_ids(en_line.strip())
            hi_tokens = sp.encode_as_ids(hi_line.strip())
            
            en_lengths.append(len(en_tokens))
            hi_lengths.append(len(hi_tokens))
    
    import numpy as np
    
    print(f"Analyzed {len(en_lengths)} sample pairs:")
    
    for lang, lengths in [("English", en_lengths), ("Hindi", hi_lengths)]:
        lengths = np.array(lengths)
        print(f"  {lang} sequences:")
        print(f"    Mean length: {lengths.mean():.1f}")
        print(f"    Median length: {np.median(lengths):.1f}")
        print(f"    Max length: {lengths.max()}")
        print(f"    95th percentile: {np.percentile(lengths, 95):.1f}")
        print(f"    99th percentile: {np.percentile(lengths, 99):.1f}")
        
        # Calculate truncation impact
        truncated_128 = (lengths > 128).sum()
        truncated_256 = (lengths > 256).sum()
        
        print(f"    Would be truncated at 128: {truncated_128} ({truncated_128/len(lengths)*100:.1f}%)")
        print(f"    Would be truncated at 256: {truncated_256} ({truncated_256/len(lengths)*100:.1f}%)")
    
    # Combined analysis (max of both sequences)
    max_lengths = np.maximum(en_lengths, hi_lengths)
    truncated_128 = (max_lengths > 128).sum()
    truncated_256 = (max_lengths > 256).sum()
    
    print(f"  Combined (max of pair):")
    print(f"    Pairs truncated at 128: {truncated_128} ({truncated_128/len(max_lengths)*100:.1f}%)")
    print(f"    Pairs truncated at 256: {truncated_256} ({truncated_256/len(max_lengths)*100:.1f}%)")

def create_benchmark_dataset(train_english_file: str, train_hindi_file: str, 
                            tokenizer_path: str, sample_size: int = 200):
    """
    Create a small benchmark dataset by reading only the first N samples
    This avoids loading the full 1.4M dataset for benchmarking
    """
    import sentencepiece as spm
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    # Read only the first sample_size lines
    en_samples = []
    hi_samples = []
    
    with open(train_english_file, 'r', encoding='utf-8') as ef, \
         open(train_hindi_file, 'r', encoding='utf-8') as hf:
        
        for i, (en_line, hi_line) in enumerate(zip(ef, hf)):
            if i >= sample_size:
                break
            en_samples.append(en_line.strip())
            hi_samples.append(hi_line.strip())
    
    return en_samples, hi_samples

def create_benchmark_data_loaders(en_samples: list, hi_samples: list, 
                                  tokenizer_path: str, batch_size: int, max_length: int):
    """
    Create data loaders from pre-loaded sample lists (no file I/O)
    """
    from src.data.dataset import NMTDataset
    from torch.utils.data import DataLoader
    import sentencepiece as spm
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    # Create dataset from samples (no file loading)
    dataset = NMTDataset(
        english_sentences=en_samples,
        hindi_sentences=hi_samples,
        tokenizer=sp,
        max_length=max_length
    )
    
    # Create data loader with SAME settings as normal training (including collate_fn)
    from src.data.dataset import collate_fn
    import torch
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep consistent for benchmarking
        collate_fn=collate_fn,  # CRITICAL: Use existing collate function for padding
        num_workers=0,  # Single thread for consistency
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return data_loader

def run_controlled_benchmark(base_args, model):
    """
    Run controlled 4-test benchmark comparing batch sizes and sequence lengths
    Uses lightweight dataset loading to avoid loading 1.4M samples 4 times
    """
    print("\n" + "=" * 60)
    print("CONTROLLED BENCHMARK MODE - 4 TEST COMPARISON")
    print("=" * 60)
    
    # Use smaller sample size for faster benchmarking
    benchmark_samples = 200  # Reduced from 1000
    print(f"Loading {benchmark_samples} samples for benchmark (same for all tests)...")
    
    # Load samples ONCE (not 1.4M samples, just benchmark_samples)
    en_samples, hi_samples = create_benchmark_dataset(
        base_args.train_english, 
        base_args.train_hindi, 
        base_args.tokenizer_path, 
        benchmark_samples
    )
    
    print(f"Loaded {len(en_samples)} sample pairs for benchmarking")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters (same for all tests)")
    print()
    
    # Test configurations - only one variable changes per test
    test_configs = [
        {
            'name': 'Baseline',
            'batch_size': 16,
            'max_length': 256,
            'description': 'batch_size=16, max_length=256 (baseline)'
        },
        {
            'name': 'Batch Comparison', 
            'batch_size': 8,
            'max_length': 256,
            'description': 'batch_size=8, max_length=256 (only batch size changed)'
        },
        {
            'name': 'Sequence Comparison',
            'batch_size': 16,
            'max_length': 128, 
            'description': 'batch_size=16, max_length=128 (only max length changed)'
        },
        {
            'name': 'Combined Optimized',
            'batch_size': 8,
            'max_length': 128,
            'description': 'batch_size=8, max_length=128 (both optimizations)'
        }
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"[TEST {i}/4] {config['name']}")
        print(f"Configuration: {config['description']}")
        print("-" * 50)
        
        # Create data loader for this specific configuration (NO full dataset loading)
        train_loader = create_benchmark_data_loaders(
            en_samples, hi_samples,  # Use pre-loaded samples
            base_args.tokenizer_path,
            config['batch_size'],
            config['max_length']
        )
        
        # Create minimal validation loader (just for trainer initialization)
        val_loader = create_benchmark_data_loaders(
            en_samples[:20], hi_samples[:20],  # Just 20 samples for validation
            base_args.tokenizer_path,
            config['batch_size'],
            config['max_length']
        )
        
        # Verify batch count matches expectation
        expected_batches = len(en_samples) // config['batch_size']
        actual_batches = len(train_loader)
        print(f"Batches: {actual_batches} (expected ~{expected_batches}, batch_size={config['batch_size']})")
        print(f"Samples: {len(train_loader.dataset)} (benchmark subset)")
        
        # Create trainer for this configuration
        trainer = NMTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=base_args.device,
            learning_rate=base_args.learning_rate,
            warmup_steps=base_args.warmup_steps,
            vocab_size=base_args.vocab_size,
            label_smoothing=base_args.label_smoothing,
            checkpoint_dir=f"benchmark_test_{i}",
            save_every_steps=base_args.save_every_steps,
            progress_every_steps=base_args.progress_every_steps
        )
        
        # Run speed test on the benchmark dataset
        import time
        test_start = time.time()
        
        # Run training loop on benchmark data (like speed_test but with full dataset)
        trainer.epoch = 0
        trainer.step = 0
        
        print(f"Running benchmark training on {len(train_loader.dataset)} samples...")
        train_loss, _ = trainer.train_epoch(speed_test_mode=True, speed_test_samples=len(train_loader.dataset))
        
        test_time = time.time() - test_start
        
        # Calculate performance metrics
        samples_processed = len(train_loader.dataset)
        measured_speed = samples_processed / test_time if test_time > 0 else 0
        
        # Estimate full epoch time (1.4M samples)
        full_dataset_size = 1412400  # Known size
        estimated_epoch_hours = (full_dataset_size / measured_speed) / 3600 if measured_speed > 0 else 999
        
        result = {
            'name': config['name'],
            'batch_size': config['batch_size'],
            'max_length': config['max_length'],
            'samples_processed': samples_processed,
            'test_time_sec': test_time,
            'measured_speed_samples_sec': measured_speed,
            'estimated_epoch_hours': estimated_epoch_hours,
            'actual_batches': actual_batches
        }
        results.append(result)
        
        print(f"RESULT: {measured_speed:.1f} samples/sec, {test_time:.1f}s total")
        print(f"Estimated full epoch: {estimated_epoch_hours:.1f} hours")
        print()
    
    # Summary comparison
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Test':<20} {'Batch':<6} {'MaxLen':<7} {'Speed':<12} {'Time':<8} {'Est Epoch':<10} {'Batches':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} {result['batch_size']:<6} {result['max_length']:<7} "
              f"{result['measured_speed_samples_sec']:<12.1f} {result['test_time_sec']:<8.1f} "
              f"{result['estimated_epoch_hours']:<10.1f} {result['actual_batches']:<8}")
    
    # Analysis
    baseline = results[0]
    print(f"\nAnalysis (vs Baseline {baseline['measured_speed_samples_sec']:.1f} samples/sec):")
    for result in results[1:]:
        speedup = result['measured_speed_samples_sec'] / baseline['measured_speed_samples_sec'] if baseline['measured_speed_samples_sec'] > 0 else 0
        time_improvement = baseline['estimated_epoch_hours'] / result['estimated_epoch_hours'] if result['estimated_epoch_hours'] > 0 else 0
        print(f"  {result['name']}: {speedup:.2f}x speed, {time_improvement:.2f}x time improvement")
    
    print(f"\nBenchmark used {benchmark_samples} samples per test (same subset for all)")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters (same for all tests)")
    print(f"Estimates based on measured performance, not promises")
    return results

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
    
    # NEW: Performance and checkpointing parameters
    parser.add_argument('--save_every_steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--progress_every_steps', type=int, default=50, help='Report progress every N steps')
    parser.add_argument('--speed_test_first', action='store_true', help='Run speed test before training')
    parser.add_argument('--speed_test_only', action='store_true', help='Run speed test only, then exit')
    parser.add_argument('--benchmark_mode', action='store_true', help='Run controlled benchmark (4 tests), then exit')
    parser.add_argument('--speed_test_samples', type=int, default=1000, help='Samples for speed test')
    parser.add_argument('--test_batch_size', type=int, default=None, help='Test different batch size (8 or 16)')
    parser.add_argument('--test_max_length', type=int, default=None, help='Test different max length (128)')
    
    
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
    
    # Analysis parameters
    parser.add_argument('--analyze_lengths', action='store_true', help='Analyze sequence lengths only')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='reference', choices=['reference', 'compact_a'],
                        help='Transformer architecture: reference 512x6 or Compact A 256x4')
    
    args = parser.parse_args()
    
    # Setup device
    args.device = setup_device()

    if args.architecture == 'compact_a':
        args.d_model = 256
        args.n_heads = 4
        args.n_layers = 4
        args.d_ff = 1024
        args.max_length = 128
        args.dropout = 0.1
    
    print("=" * 80)
    print("NEURAL MACHINE TRANSLATION TRAINING")
    print("=" * 80)
    
    # Run sequence length analysis if requested
    if args.analyze_lengths:
        analyze_sequence_lengths(args.train_english, args.train_hindi, args.tokenizer_path)
        return
    
    # Determine final configuration before creating any data loaders
    final_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
    final_max_length = args.test_max_length if args.test_max_length else args.max_length
    
    print(f"Final configuration: batch_size={final_batch_size}, max_length={final_max_length}")
    
    # Create validation files if they don't exist
    if not os.path.exists(args.val_english) or not os.path.exists(args.val_hindi):
        print("Creating validation files from original dataset...")
        create_validation_files()
    
    # Create data loaders ONCE with final configuration
    print("Loading datasets...")
    train_loader, val_loader = create_data_loaders(
        train_english_file=args.train_english,
        train_hindi_file=args.train_hindi,
        val_english_file=args.val_english,
        val_hindi_file=args.val_hindi,
        tokenizer_path=args.tokenizer_path,
        batch_size=final_batch_size,
        max_length=final_max_length,
        num_workers=args.num_workers
    )
    
    print(f"Training batches: {len(train_loader)} (batch_size={final_batch_size})")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    if args.architecture == 'compact_a':
        from src.model import create_compact_a_model
        compact_config = {
            'vocab_size': args.vocab_size,
            'd_model': 256,
            'n_heads': 4,
            'encoder_layers': 4,
            'decoder_layers': 4,
            'd_ff': 1024,
            'max_seq_length': 128,
            'dropout': 0.1,
        }
        model = create_compact_a_model(vocab_size=args.vocab_size)
        args.d_model = compact_config['d_model']
        args.n_heads = compact_config['n_heads']
        args.n_layers = compact_config['encoder_layers']
        args.d_ff = compact_config['d_ff']
        args.max_length = compact_config['max_seq_length']
        args.dropout = compact_config['dropout']
    else:
        model = create_transformer_model(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_length=args.max_length,
            dropout=args.dropout,
        )
    
    total_params, trainable_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    
    # Create trainer (data loaders already created with final configuration)
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
        checkpoint_dir=args.checkpoint_dir,
        save_every_steps=args.save_every_steps,
        progress_every_steps=args.progress_every_steps
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
    
    # Speed test only mode - exit after speed test
    if args.speed_test_only:
        print("\nSPEED TEST ONLY MODE")
        print("-" * 50)
        speed_results = trainer.speed_test(args.speed_test_samples)
        print(f"\nSpeed test completed!")
        print(f"Results: {speed_results['samples_processed']} samples processed")
        print(f"Performance measured and logged above")
        print(f"\nTo start full training, run without --speed_test_only flag")
        return
    
    # Benchmark mode - run controlled 4-test comparison, then exit
    if args.benchmark_mode:
        run_controlled_benchmark(args, model)
        print(f"\nBenchmark completed! No training started.")
        return
    
    # Start training
    print("\nStarting training...")
    print("-" * 50)
    
    try:
        trainer.train(
            num_epochs=args.num_epochs,
            save_every=1,  # Save every epoch
            early_stopping_patience=5,
            speed_test_first=args.speed_test_first,
            speed_test_samples=args.speed_test_samples
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        interrupt_path = os.path.join(args.checkpoint_dir, 'interrupted_checkpoint.pt')
        trainer.save_checkpoint(interrupt_path, epoch_completed=False)
        print(f"Checkpoint saved: {interrupt_path}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        # Save current state for debugging
        error_path = os.path.join(args.checkpoint_dir, 'error_checkpoint.pt')
        trainer.save_checkpoint(error_path, epoch_completed=False)
        print(f"Error checkpoint saved: {error_path}")
        raise
    
    print("\nTraining script completed!")

if __name__ == "__main__":
    main()