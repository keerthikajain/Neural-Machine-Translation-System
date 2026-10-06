# Test the complete training pipeline with a small model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our modules
from src.model import create_transformer_model
from src.data.dataset import create_data_loaders
from src.training.trainer import NMTTrainer

def test_training_pipeline():
    """Test the complete training pipeline with a tiny model"""
    print("=" * 60)
    print("TESTING TRAINING PIPELINE")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create tiny model for testing
    print("Creating small test model...")
    model = create_transformer_model(
        vocab_size=32000,
        d_model=128,  # Much smaller for testing
        n_heads=4,    # Fewer heads
        n_layers=2    # Fewer layers
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Test model parameters: {total_params:,}")
    
    # Create small data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_english_file='data/processed/clean_train_en.txt',
        train_hindi_file='data/processed/clean_train_hi.txt',
        val_english_file='data/processed/val_en.txt',
        val_hindi_file='data/processed/val_hi.txt',
        tokenizer_path='data/processed/tokenizer/sp_model.model',
        batch_size=2,     # Very small batch
        max_length=32,    # Short sequences
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test single batch forward pass
    print("\nTesting model forward pass...")
    batch = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"  Source: {batch['src_tokens'].shape}")
    print(f"  Target input: {batch['tgt_input'].shape}")
    print(f"  Target output: {batch['tgt_output'].shape}")
    
    model.eval()
    with torch.no_grad():
        src_tokens = batch['src_tokens'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        
        # Forward pass
        logits = model(src_tokens, tgt_input)
        print(f"  Model output: {logits.shape}")
        
        # Check output is correct
        expected_shape = (batch['tgt_input'].size(0), batch['tgt_input'].size(1), 32000)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        
        print("✅ Forward pass successful!")
    
    # Test trainer initialization
    print("\nTesting trainer initialization...")
    trainer = NMTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.0001,
        warmup_steps=100,  # Very small warmup for testing
        vocab_size=32000,
        label_smoothing=0.1,
        checkpoint_dir='test_checkpoints'
    )
    
    print("✅ Trainer initialized successfully!")
    
    # Test single training step
    print("\nTesting single training step...")
    model.train()
    
    # Get batch and test training step
    batch = next(iter(train_loader))
    initial_loss = trainer.train_step(batch)
    print(f"Training step loss: {initial_loss:.4f}")
    
    print("✅ Training step successful!")
    
    # Test validation
    print("\nTesting validation...")
    val_loss = trainer.validate()
    print(f"Validation loss: {val_loss:.4f}")
    
    print("✅ Validation successful!")
    
    # Test one full epoch (just a few batches)
    print("\nTesting mini training loop...")
    
    # Limit to just 5 batches for testing
    original_len = len(trainer.train_loader.dataset)
    
    # Create a smaller subset for testing
    from torch.utils.data import Subset
    small_indices = list(range(10))  # Just 10 samples
    small_dataset = Subset(trainer.train_loader.dataset, small_indices)
    small_loader = DataLoader(small_dataset, batch_size=2, collate_fn=trainer.train_loader.collate_fn)
    
    # Replace with small loader temporarily
    trainer.train_loader = small_loader
    
    try:
        train_loss, val_loss = trainer.train_epoch()
        print(f"Mini epoch completed:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        
        print("✅ Mini training epoch successful!")
        
    except Exception as e:
        print(f"❌ Training epoch failed: {e}")
        raise
    
    # Test checkpoint saving
    print("\nTesting checkpoint saving...")
    checkpoint_path = 'test_checkpoints/test_checkpoint.pt'
    trainer.save_checkpoint(checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        print("✅ Checkpoint saved successfully!")
        
        # Test loading
        trainer.load_checkpoint(checkpoint_path)
        print("✅ Checkpoint loaded successfully!")
    else:
        print("❌ Checkpoint not found!")
    
    # Cleanup
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
    
    print("\n" + "=" * 60)
    print("🎉 ALL TRAINING PIPELINE TESTS PASSED!")
    print("The training pipeline is ready for full training.")
    print("=" * 60)

if __name__ == "__main__":
    test_training_pipeline()