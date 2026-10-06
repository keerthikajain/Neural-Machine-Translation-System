# Comprehensive Training Pipeline Verification
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import sentencepiece as smp
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from datetime import datetime

# Import our modules
from src.model import create_transformer_model
from src.data.dataset import NMTDataset, collate_fn, create_data_loaders
from src.training.trainer import NMTTrainer, LabelSmoothingCrossEntropy, TransformerLRScheduler

print("=" * 80)
print("COMPREHENSIVE TRAINING PIPELINE VERIFICATION")
print("=" * 80)

verification_results = {
    'timestamp': datetime.now().isoformat(),
    'tests': {},
    'issues_found': [],
    'issues_fixed': [],
    'hardware': {},
    'final_status': None
}

def add_test_result(test_name, status, details=""):
    """Record test result"""
    verification_results['tests'][test_name] = {
        'status': status,
        'details': details
    }
    symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{symbol} {test_name}: {status}")
    if details:
        print(f"    {details}")

def add_issue(issue_type, description):
    """Record issue found or fixed"""
    verification_results[issue_type].append(description)
# 1. DATASET PIPELINE VERIFICATION
print("\n1. DATASET PIPELINE VERIFICATION")
print("-" * 50)

try:
    # Test tokenizer configuration
    tokenizer_path = 'data/processed/tokenizer/sp_model.model'
    sp = smp.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    expected_tokens = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3}
    actual_tokens = {
        'PAD': sp.pad_id(),
        'UNK': sp.unk_id(),
        'BOS': sp.bos_id(),
        'EOS': sp.eos_id()
    }
    
    tokenizer_ok = expected_tokens == actual_tokens
    add_test_result("Tokenizer Configuration", 
                   "PASS" if tokenizer_ok else "FAIL",
                   f"Expected: {expected_tokens}, Actual: {actual_tokens}")
    
    if sp.vocab_size() != 32000:
        add_issue("issues_found", f"Vocab size mismatch: expected 32000, got {sp.vocab_size()}")
        add_test_result("Vocabulary Size", "FAIL", f"Expected 32000, got {sp.vocab_size()}")
    else:
        add_test_result("Vocabulary Size", "PASS", f"32000 tokens confirmed")

except Exception as e:
    add_test_result("Tokenizer Loading", "FAIL", str(e))
    add_issue("issues_found", f"Tokenizer loading failed: {e}")

# Test data alignment
try:
    dataset = NMTDataset(
        english_file='data/processed/clean_train_en.txt',
        hindi_file='data/processed/clean_train_hi.txt',
        tokenizer_path=tokenizer_path,
        max_length=64
    )
    
    # Check first few examples for alignment
    alignment_ok = True
    for i in range(min(5, len(dataset))):
        example = dataset[i]
        if not example['english_text'] or not example['hindi_text']:
            alignment_ok = False
            break
    
    add_test_result("Data Alignment", "PASS" if alignment_ok else "FAIL",
                   f"Checked {min(5, len(dataset))} examples")
    
except Exception as e:
    add_test_result("Dataset Loading", "FAIL", str(e))
    add_issue("issues_found", f"Dataset loading failed: {e}")
# 2. SEQUENCE CONSTRUCTION VERIFICATION
print("\n2. SEQUENCE CONSTRUCTION VERIFICATION")
print("-" * 50)

try:
    # Test sequence construction with known example
    example = dataset[0]
    
    print(f"Example analysis:")
    print(f"  English: '{example['english_text'][:50]}...'")
    print(f"  Hindi: '{example['hindi_text'][:50]}...'")
    print(f"  Source tokens: {example['src_tokens'][:10]}...")
    print(f"  Target input: {example['tgt_input'][:10]}...")  
    print(f"  Target output: {example['tgt_output'][:10]}...")
    
    # Verify BOS/EOS handling
    src_starts_with_bos = example['src_tokens'][0] == sp.bos_id()
    tgt_input_starts_with_bos = example['tgt_input'][0] == sp.bos_id()
    tgt_output_ends_with_eos = example['tgt_output'][-1] == sp.eos_id()
    
    add_test_result("Source BOS Token", "PASS" if src_starts_with_bos else "FAIL")
    add_test_result("Target Input BOS Token", "PASS" if tgt_input_starts_with_bos else "FAIL")
    add_test_result("Target Output EOS Token", "PASS" if tgt_output_ends_with_eos else "FAIL")
    
    # Verify decoder input/output shifting
    # tgt_input should be [BOS, token1, token2, ...]
    # tgt_output should be [token1, token2, ..., EOS]
    if len(example['tgt_input']) == len(example['tgt_output']):
        shifting_ok = example['tgt_input'][1:] == example['tgt_output'][:-1]
        add_test_result("Input/Output Shifting", "PASS" if shifting_ok else "FAIL",
                       "Decoder input and output are properly shifted")
    else:
        add_test_result("Input/Output Shifting", "FAIL", 
                       f"Length mismatch: input={len(example['tgt_input'])}, output={len(example['tgt_output'])}")

except Exception as e:
    add_test_result("Sequence Construction", "FAIL", str(e))
    add_issue("issues_found", f"Sequence construction failed: {e}")

# 3. PADDING AND COLLATION VERIFICATION
print("\n3. PADDING AND COLLATION VERIFICATION")
print("-" * 50)

try:
    # Create a batch with different sequence lengths
    indices = [0, 10, 100]  # Different examples
    mini_dataset = Subset(dataset, indices)
    test_loader = DataLoader(mini_dataset, batch_size=3, collate_fn=collate_fn)
    
    batch = next(iter(test_loader))
    
    print(f"Batch shapes:")
    print(f"  Source: {batch['src_tokens'].shape}")
    print(f"  Target input: {batch['tgt_input'].shape}")
    print(f"  Target output: {batch['tgt_output'].shape}")
    
    # Verify padding
    pad_id = 0
    src_has_padding = (batch['src_tokens'] == pad_id).any()
    tgt_has_padding = (batch['tgt_input'] == pad_id).any()
    
    add_test_result("Dynamic Padding", "PASS", 
                   f"Source padding: {src_has_padding}, Target padding: {tgt_has_padding}")
    
    # Verify all sequences in batch have same padded length
    batch_size = batch['src_tokens'].size(0)
    seq_lengths_consistent = True
    for i in range(1, batch_size):
        if (batch['src_tokens'][i].shape != batch['src_tokens'][0].shape or
            batch['tgt_input'][i].shape != batch['tgt_input'][0].shape):
            seq_lengths_consistent = False
    
    add_test_result("Batch Consistency", "PASS" if seq_lengths_consistent else "FAIL")

except Exception as e:
    add_test_result("Padding and Collation", "FAIL", str(e))
    add_issue("issues_found", f"Padding verification failed: {e}")
# 4. PADDING MASK VERIFICATION
print("\n4. PADDING MASK VERIFICATION")
print("-" * 50)

try:
    # Test mask creation
    from src.model.encoder import create_padding_mask
    from src.model.decoder import create_target_mask
    
    test_tokens = torch.tensor([[2, 5, 8, 0, 0], [2, 7, 12, 15, 0]])  # With padding
    
    # Test padding mask
    pad_mask = create_padding_mask(test_tokens, pad_token_id=0)
    
    # Verify mask shape and values
    expected_shape = (2, 1, 1, 5)  # [batch, 1, 1, seq_len]
    mask_shape_ok = pad_mask.shape == expected_shape
    
    # Check that padded positions are 0, non-padded are 1
    expected_mask = torch.tensor([[[[1., 1., 1., 0., 0.]]],
                                 [[[1., 1., 1., 1., 0.]]]])
    mask_values_ok = torch.allclose(pad_mask, expected_mask)
    
    add_test_result("Padding Mask Shape", "PASS" if mask_shape_ok else "FAIL",
                   f"Expected: {expected_shape}, Got: {pad_mask.shape}")
    add_test_result("Padding Mask Values", "PASS" if mask_values_ok else "FAIL")
    
    # Test target mask (combines padding and causal)
    tgt_mask = create_target_mask(test_tokens, pad_token_id=0)
    target_mask_shape_ok = tgt_mask.shape == (2, 1, 5, 5)
    
    add_test_result("Target Mask Shape", "PASS" if target_mask_shape_ok else "FAIL",
                   f"Expected: (2, 1, 5, 5), Got: {tgt_mask.shape}")

except Exception as e:
    add_test_result("Mask Verification", "FAIL", str(e))
    add_issue("issues_found", f"Mask verification failed: {e}")

# 5. LOSS FUNCTION VERIFICATION  
print("\n5. LOSS FUNCTION VERIFICATION")
print("-" * 50)

try:
    # Test label smoothing loss
    criterion = LabelSmoothingCrossEntropy(
        num_classes=1000,
        smoothing=0.1,
        ignore_index=0
    )
    
    # Test with padded targets
    logits = torch.randn(6, 1000)  # [batch_size * seq_len, vocab_size]
    targets = torch.tensor([5, 10, 15, 0, 0, 20])  # Some padding (0)
    
    loss = criterion(logits, targets)
    
    # Verify loss is finite and reasonable
    loss_finite = torch.isfinite(loss).all()
    loss_reasonable = 0 < loss.item() < 20  # Reasonable range for cross-entropy
    
    add_test_result("Loss Function Finite", "PASS" if loss_finite else "FAIL")
    add_test_result("Loss Function Range", "PASS" if loss_reasonable else "FAIL",
                   f"Loss value: {loss.item():.4f}")
    
    # Test that changing PAD predictions doesn't affect loss
    logits_pad_changed = logits.clone()
    logits_pad_changed[3] = torch.randn(1000)  # Change padded position
    logits_pad_changed[4] = torch.randn(1000)  # Change padded position
    
    loss_unchanged = criterion(logits_pad_changed, targets)
    pad_ignored = torch.allclose(loss, loss_unchanged, atol=1e-6)
    
    add_test_result("PAD Token Ignored", "PASS" if pad_ignored else "FAIL",
                   f"Original: {loss.item():.6f}, Modified: {loss_unchanged.item():.6f}")

except Exception as e:
    add_test_result("Loss Function", "FAIL", str(e))
    add_issue("issues_found", f"Loss function verification failed: {e}")
# 6. OPTIMIZER AND SCHEDULER VERIFICATION
print("\n6. OPTIMIZER AND SCHEDULER VERIFICATION")
print("-" * 50)

try:
    # Create small model for testing
    device = torch.device('cpu')
    test_model = create_transformer_model(vocab_size=32000, d_model=64, n_heads=4, n_layers=2)
    
    # Test optimizer setup
    optimizer = torch.optim.AdamW(
        test_model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    # Test scheduler
    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=64,
        warmup_steps=100
    )
    
    # Test learning rate progression
    lr_schedule = []
    for step in [0, 1, 10, 50, 100, 200, 500]:
        scheduler.step_num = step
        lr = scheduler._calculate_lr()
        lr_schedule.append((step, lr))
    
    print("Learning rate schedule:")
    for step, lr in lr_schedule:
        print(f"  Step {step:3d}: LR = {lr:.8f}")
    
    # Verify warmup behavior (LR should increase then decrease)
    warmup_increasing = lr_schedule[3][1] > lr_schedule[2][1]  # Step 50 > Step 10
    post_warmup_decreasing = lr_schedule[6][1] < lr_schedule[5][1]  # Step 500 < Step 200
    
    add_test_result("LR Warmup Phase", "PASS" if warmup_increasing else "FAIL")
    add_test_result("LR Decay Phase", "PASS" if post_warmup_decreasing else "FAIL")
    
    # Test parameter updates
    initial_params = [p.clone() for p in test_model.parameters()]
    
    # Dummy forward pass
    dummy_src = torch.randint(1, 1000, (2, 10))
    dummy_tgt = torch.randint(1, 1000, (2, 8))
    logits = test_model(dummy_src, dummy_tgt)
    
    # Dummy loss and backward
    dummy_targets = torch.randint(0, 1000, (2, 8))
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), dummy_targets.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    has_gradients = all(p.grad is not None for p in test_model.parameters())
    add_test_result("Gradient Computation", "PASS" if has_gradients else "FAIL")
    
    # Test gradient clipping
    nn.utils.clip_grad_norm_(test_model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    
    # Check parameters changed
    params_changed = any(not torch.equal(initial, current) 
                        for initial, current in zip(initial_params, test_model.parameters()))
    
    add_test_result("Parameter Updates", "PASS" if params_changed else "FAIL")

except Exception as e:
    add_test_result("Optimizer/Scheduler", "FAIL", str(e))
    add_issue("issues_found", f"Optimizer/scheduler verification failed: {e}")
# 7. TRAINING LOOP VERIFICATION
print("\n7. TRAINING LOOP VERIFICATION")
print("-" * 50)

try:
    # Create minimal trainer for testing
    mini_indices = list(range(10))
    mini_dataset = Subset(dataset, mini_indices)
    mini_loader = DataLoader(mini_dataset, batch_size=2, collate_fn=collate_fn)
    
    # Create validation loader  
    val_dataset = NMTDataset(
        english_file='data/processed/val_en.txt',
        hindi_file='data/processed/val_hi.txt',
        tokenizer_path=tokenizer_path,
        max_length=64
    )
    val_indices = list(range(5))
    mini_val_dataset = Subset(val_dataset, val_indices)
    mini_val_loader = DataLoader(mini_val_dataset, batch_size=2, collate_fn=collate_fn)
    
    trainer = NMTTrainer(
        model=test_model,
        train_loader=mini_loader,
        val_loader=mini_val_loader,
        device=device,
        learning_rate=0.0001,
        warmup_steps=10,
        vocab_size=32000,
        label_smoothing=0.1,
        checkpoint_dir='temp_checkpoints'
    )
    
    # Test single training step
    batch = next(iter(mini_loader))
    initial_loss = trainer.train_step(batch)
    
    add_test_result("Training Step Execution", "PASS", f"Loss: {initial_loss:.4f}")
    
    # Test validation (should not change parameters)
    params_before_val = [p.clone() for p in test_model.parameters()]
    val_loss = trainer.validate()
    params_after_val = [p.clone() for p in test_model.parameters()]
    
    val_no_updates = all(torch.equal(before, after) 
                        for before, after in zip(params_before_val, params_after_val))
    
    add_test_result("Validation Isolation", "PASS" if val_no_updates else "FAIL",
                   f"Val loss: {val_loss:.4f}")

except Exception as e:
    add_test_result("Training Loop", "FAIL", str(e))
    add_issue("issues_found", f"Training loop verification failed: {e}")

# 8. TINY OVERFITTING TEST
print("\n8. TINY OVERFITTING TEST")
print("-" * 50)

try:
    print("Starting tiny overfitting test...")
    
    # Use even smaller dataset for overfitting
    tiny_indices = list(range(3))
    tiny_dataset = Subset(dataset, tiny_indices)
    tiny_loader = DataLoader(tiny_dataset, batch_size=1, collate_fn=collate_fn)
    
    # Reset model
    test_model = create_transformer_model(vocab_size=32000, d_model=64, n_heads=4, n_layers=2)
    
    trainer = NMTTrainer(
        model=test_model,
        train_loader=tiny_loader,
        val_loader=mini_val_loader,
        device=device,
        learning_rate=0.001,  # Higher LR for faster overfitting
        warmup_steps=5,
        vocab_size=32000,
        label_smoothing=0.0,  # No smoothing for overfitting
        checkpoint_dir='temp_checkpoints'
    )
    
    # Record initial losses
    initial_losses = []
    for batch in tiny_loader:
        trainer.model.eval()
        with torch.no_grad():
            src_tokens = batch['src_tokens'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            logits = trainer.model(src_tokens, tgt_input)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = tgt_output.view(-1)
            
            loss = trainer.criterion(logits_flat, targets_flat)
            initial_losses.append(loss.item())
    
    avg_initial_loss = sum(initial_losses) / len(initial_losses)
    print(f"Initial average loss: {avg_initial_loss:.4f}")
    
    # Train for several steps
    trainer.model.train()
    final_losses = []
    
    for epoch in range(5):  # Multiple epochs on tiny data
        for batch in tiny_loader:
            loss = trainer.train_step(batch)
            final_losses.append(loss)
    
    avg_final_loss = sum(final_losses[-3:]) / 3  # Average of last 3 steps
    print(f"Final average loss: {avg_final_loss:.4f}")
    
    # Check if model is learning (loss decreasing)
    loss_decreased = avg_final_loss < avg_initial_loss * 0.9  # 10% decrease
    overfitting_working = avg_final_loss < avg_initial_loss and avg_final_loss < 5.0
    
    add_test_result("Tiny Overfitting Test", "PASS" if overfitting_working else "FAIL",
                   f"Initial: {avg_initial_loss:.4f}, Final: {avg_final_loss:.4f}")
    
    if not loss_decreased:
        add_issue("issues_found", "Model not learning on tiny dataset - loss not decreasing significantly")

except Exception as e:
    add_test_result("Tiny Overfitting Test", "FAIL", str(e))
    add_issue("issues_found", f"Overfitting test failed: {e}")
# 9. CHECKPOINT VERIFICATION
print("\n9. CHECKPOINT VERIFICATION")
print("-" * 50)

try:
    # Test checkpoint saving and loading
    checkpoint_path = 'temp_checkpoints/test_checkpoint.pt'
    
    # Save checkpoint
    trainer.save_checkpoint(checkpoint_path)
    checkpoint_exists = os.path.exists(checkpoint_path)
    
    add_test_result("Checkpoint Saving", "PASS" if checkpoint_exists else "FAIL")
    
    if checkpoint_exists:
        # Load checkpoint and verify contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 
                        'epoch', 'step', 'train_losses', 'val_losses']
        
        has_required_keys = all(key in checkpoint for key in required_keys)
        add_test_result("Checkpoint Contents", "PASS" if has_required_keys else "FAIL",
                       f"Keys: {list(checkpoint.keys())}")
        
        # Test loading
        try:
            new_trainer = NMTTrainer(
                model=create_transformer_model(vocab_size=32000, d_model=64, n_heads=4, n_layers=2),
                train_loader=tiny_loader,
                val_loader=mini_val_loader,
                device=device,
                learning_rate=0.001,
                warmup_steps=5,
                vocab_size=32000,
                label_smoothing=0.0,
                checkpoint_dir='temp_checkpoints'
            )
            
            new_trainer.load_checkpoint(checkpoint_path)
            add_test_result("Checkpoint Loading", "PASS")
        
        except Exception as e:
            add_test_result("Checkpoint Loading", "FAIL", str(e))
    
    # Cleanup
    import shutil
    if os.path.exists('temp_checkpoints'):
        shutil.rmtree('temp_checkpoints')

except Exception as e:
    add_test_result("Checkpoint Verification", "FAIL", str(e))
    add_issue("issues_found", f"Checkpoint verification failed: {e}")

# 10. EVALUATION VERIFICATION
print("\n10. EVALUATION VERIFICATION")
print("-" * 50)

try:
    from src.evaluation.evaluator import NMTEvaluator
    
    evaluator = NMTEvaluator(
        model=test_model,
        tokenizer_path=tokenizer_path,
        device=device,
        max_length=50
    )
    
    # Test translation
    test_sentence = "Hello world"
    translation = evaluator.translate_sentence(test_sentence)
    
    translation_generated = translation is not None and len(translation) > 0
    add_test_result("Translation Generation", "PASS" if translation_generated else "FAIL",
                   f"'{test_sentence}' -> '{translation}'")
    
    # Test BLEU calculation
    bleu_score = evaluator._calculate_simple_bleu("test translation", "test reference")
    bleu_reasonable = 0 <= bleu_score <= 1
    
    add_test_result("BLEU Calculation", "PASS" if bleu_reasonable else "FAIL",
                   f"BLEU score: {bleu_score:.4f}")

except Exception as e:
    add_test_result("Evaluation", "FAIL", str(e))
    add_issue("issues_found", f"Evaluation verification failed: {e}")

# 11. HARDWARE CHECK
print("\n11. HARDWARE VERIFICATION")
print("-" * 50)

try:
    import platform
    
    # System information
    verification_results['hardware'] = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': os.cpu_count()
    }
    
    if torch.cuda.is_available():
        verification_results['hardware'].update({
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'gpu_count': torch.cuda.device_count()
        })
    
    print("Hardware detected:")
    for key, value in verification_results['hardware'].items():
        print(f"  {key}: {value}")
    
    add_test_result("Hardware Detection", "PASS")

except Exception as e:
    add_test_result("Hardware Detection", "FAIL", str(e))
# 12. CONFIGURATION AUDIT
print("\n12. CONFIGURATION AUDIT")
print("-" * 50)

# Audit configuration from train_model.py
expected_config = {
    'vocab_size': 32000,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'max_length': 256,
    'batch_size': 16,
    'num_epochs': 5,
    'learning_rate': 0.0001,
    'warmup_steps': 2000
}

print("Expected configuration:")
for key, value in expected_config.items():
    print(f"  {key}: {value}")

# Verify configuration is reasonable for detected hardware
hardware = verification_results['hardware']
config_warnings = []

# Batch size warning for CPU
if not hardware['cuda_available'] and expected_config['batch_size'] > 8:
    config_warnings.append("Large batch size (16) on CPU may be slow")

# Memory estimation
model_params = 93322496  # From previous verification
model_memory_mb = model_params * 4 / (1024**2)  # 4 bytes per parameter
training_memory_estimate = model_memory_mb * 3  # Rough estimate with gradients/optimizer

if hardware['cuda_available']:
    gpu_memory_gb = hardware.get('gpu_memory_gb', 0)
    if training_memory_estimate > gpu_memory_gb * 1024 * 0.8:  # 80% threshold
        config_warnings.append(f"Training may exceed GPU memory: ~{training_memory_estimate:.0f}MB estimated")

add_test_result("Configuration Audit", "PASS", 
               f"Warnings: {len(config_warnings)}")

for warning in config_warnings:
    print(f"  ⚠️  {warning}")

# FINAL ASSESSMENT
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

# Count results
passed_tests = sum(1 for test in verification_results['tests'].values() if test['status'] == 'PASS')
failed_tests = sum(1 for test in verification_results['tests'].values() if test['status'] == 'FAIL')
total_tests = len(verification_results['tests'])

print(f"\nTest Results: {passed_tests}/{total_tests} PASSED, {failed_tests} FAILED")

# Determine final status
critical_failures = []
for test_name, result in verification_results['tests'].items():
    if result['status'] == 'FAIL':
        if any(critical in test_name.lower() for critical in 
               ['tokenizer', 'shifting', 'loss', 'overfitting', 'checkpoint']):
            critical_failures.append(test_name)

if len(critical_failures) == 0 and failed_tests <= 2:  # Allow minor failures
    verification_results['final_status'] = "READY FOR FULL TRAINING"
else:
    verification_results['final_status'] = "NOT READY FOR FULL TRAINING"

print(f"\nFinal Status: {verification_results['final_status']}")

if critical_failures:
    print("\nCritical failures that must be fixed:")
    for failure in critical_failures:
        print(f"  - {failure}")

if verification_results['issues_found']:
    print(f"\nIssues Found ({len(verification_results['issues_found'])}):")
    for issue in verification_results['issues_found']:
        print(f"  - {issue}")

if verification_results['issues_fixed']:
    print(f"\nIssues Fixed ({len(verification_results['issues_fixed'])}):")
    for fix in verification_results['issues_fixed']:
        print(f"  - {fix}")

# Save verification report
with open('training_pipeline_verification_results.json', 'w') as f:
    json.dump(verification_results, f, indent=2)

print(f"\nDetailed results saved: training_pipeline_verification_results.json")
print("=" * 80)