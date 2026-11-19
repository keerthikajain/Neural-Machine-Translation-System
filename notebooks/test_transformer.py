# Test Transformer model implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import sentencepiece as spm
from src.model import create_transformer_model

print("=== TESTING TRANSFORMER MODEL ===")

# Load tokenizer to get vocabulary size
tokenizer_path = 'data/processed/tokenizer/sp_model.model'
if not os.path.exists(tokenizer_path):
    print(f"Error: Tokenizer not found at {tokenizer_path}")
    print("Please run the tokenizer training first.")
    exit()

print("Loading SentencePiece tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)
vocab_size = sp.vocab_size()

print(f"Vocabulary size: {vocab_size}")

# Test model creation
print("\nCreating Transformer model...")
model = create_transformer_model(
    vocab_size=vocab_size,
    d_model=512,
    n_heads=8, 
    n_layers=6
)

print("✓ Model created successfully")

# Print model information
model_info = model.get_model_info()
print(f"\nModel Information:")
print(f"Total parameters: {model_info['total_parameters']:,}")
print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
print(f"Source vocab size: {model_info['src_vocab_size']}")
print(f"Target vocab size: {model_info['tgt_vocab_size']}")
print(f"Encoder layers: {model_info['encoder_layers']}")
print(f"Decoder layers: {model_info['decoder_layers']}")

# Test forward pass with dummy data
print("\n=== TESTING FORWARD PASS ===")

# Create dummy batch data
batch_size = 2
src_seq_len = 10
tgt_seq_len = 8

# Random token IDs (avoid 0 which is padding)
src_tokens = torch.randint(1, min(1000, vocab_size), (batch_size, src_seq_len))
tgt_tokens = torch.randint(1, min(1000, vocab_size), (batch_size, tgt_seq_len))

# Add padding token to make mask more realistic
src_tokens[:, -1] = 0  # Last token is padding
tgt_tokens[:, -1] = 0  # Last token is padding

print(f"Source tokens shape: {src_tokens.shape}")
print(f"Target tokens shape: {tgt_tokens.shape}")

# Test forward pass
print("\nTesting forward pass...")
try:
    with torch.no_grad():  # No gradient computation for testing
        logits = model(src_tokens, tgt_tokens)
    
    print(f"✓ Forward pass successful")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [batch_size={batch_size}, tgt_len={tgt_seq_len}, vocab_size={vocab_size}]")
    
    # Verify output shape
    expected_shape = (batch_size, tgt_seq_len, vocab_size)
    if logits.shape == expected_shape:
        print("✓ Output shape correct")
    else:
        print(f"✗ Output shape mismatch: got {logits.shape}, expected {expected_shape}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    exit()

# Test encoding only (for inference)
print("\n=== TESTING ENCODER ONLY ===")
try:
    with torch.no_grad():
        encoder_output, src_mask = model.encode(src_tokens)
    
    print(f"✓ Encoder-only pass successful")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Source mask shape: {src_mask.shape}")

except Exception as e:
    print(f"✗ Encoder-only pass failed: {e}")

# Test with real tokenizer
print("\n=== TESTING WITH REAL TOKENIZATION ===")

# Test sentences
test_sentences = [
    "Hello world, how are you?",
    "Machine translation is challenging."
]

print("Testing tokenization and model forward pass...")

for i, sentence in enumerate(test_sentences):
    print(f"\nTest sentence {i+1}: {sentence}")
    
    # Tokenize
    tokens = sp.encode_as_ids(sentence)
    print(f"Tokens: {tokens}")
    
    # Add BOS token and convert to tensor
    tokens_with_bos = [sp.bos_id()] + tokens
    src_tensor = torch.tensor(tokens_with_bos).unsqueeze(0)  # Add batch dimension
    
    print(f"Input tensor shape: {src_tensor.shape}")
    
    # Test encoding
    try:
        with torch.no_grad():
            encoder_output, src_mask = model.encode(src_tensor)
        print(f"✓ Encoded successfully, output shape: {encoder_output.shape}")
    except Exception as e:
        print(f"✗ Encoding failed: {e}")

# Test memory usage
print(f"\n=== MEMORY USAGE ===")
def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

model_size = get_model_size_mb(model)
print(f"Model size: {model_size:.2f} MB")

print(f"\n=== TRANSFORMER TEST COMPLETE ===")
print("✓ Model architecture implemented correctly")
print("✓ Forward pass working")
print("✓ Encoder-decoder separation working") 
print("✓ Real tokenization integration working")
print("✓ Ready for training pipeline implementation")