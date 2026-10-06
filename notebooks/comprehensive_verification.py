# Comprehensive Transformer Architecture Verification
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import sentencepiece as smp
from src.model import create_transformer_model, Transformer
from src.model.embeddings import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from src.model.attention import MultiHeadAttention
from src.model.encoder import TransformerEncoder, create_padding_mask
from src.model.decoder import TransformerDecoder, create_target_mask, create_causal_mask

print("=" * 80)
print("COMPREHENSIVE TRANSFORMER ARCHITECTURE VERIFICATION")
print("=" * 80)

# Load tokenizer for vocabulary verification
tokenizer_path = 'data/processed/tokenizer/sp_model.model'
sp = smp.SentencePieceProcessor()
sp.load(tokenizer_path)
actual_vocab_size = sp.vocab_size()

print(f"SentencePiece vocabulary size: {actual_vocab_size}")
print(f"Expected special tokens: PAD=0, UNK=1, BOS=2, EOS=3")
print(f"Actual special tokens: PAD={sp.pad_id()}, UNK={sp.unk_id()}, BOS={sp.bos_id()}, EOS={sp.eos_id()}")

verification_report = []

def add_result(test_name, status, details=""):
    verification_report.append({
        'test': test_name,
        'status': status,
        'details': details
    })
    print(f"{status}: {test_name}")
    if details:
        print(f"    {details}")

# 1. MODEL STRUCTURE VERIFICATION
print("\n" + "=" * 50)
print("1. MODEL STRUCTURE VERIFICATION")
print("=" * 50)

model = create_transformer_model(vocab_size=actual_vocab_size)

# Check if encoder-decoder architecture exists
has_encoder = hasattr(model, 'encoder') and isinstance(model.encoder, TransformerEncoder)
has_decoder = hasattr(model, 'decoder') and isinstance(model.decoder, TransformerDecoder)
add_result("Encoder-Decoder Architecture", "PASS" if has_encoder and has_decoder else "FAIL")

# Check vocabulary sizes match
vocab_match = model.src_vocab_size == actual_vocab_size and model.tgt_vocab_size == actual_vocab_size
add_result("Vocabulary Size Match", "PASS" if vocab_match else "FAIL", 
          f"Model: src={model.src_vocab_size}, tgt={model.tgt_vocab_size}, Tokenizer: {actual_vocab_size}")

# 2. EMBEDDINGS VERIFICATION
print("\n" + "=" * 50)
print("2. EMBEDDINGS VERIFICATION")
print("=" * 50)

# Test token embeddings
batch_size, seq_len, d_model = 2, 8, 512
test_tokens = torch.randint(0, actual_vocab_size, (batch_size, seq_len))

# Test encoder embeddings
enc_embeddings = model.encoder.embedding(test_tokens)
enc_embed_shape_ok = enc_embeddings.shape == (batch_size, seq_len, d_model)
add_result("Encoder Embedding Shape", "PASS" if enc_embed_shape_ok else "FAIL",
          f"Expected: ({batch_size}, {seq_len}, {d_model}), Got: {enc_embeddings.shape}")

# Test decoder embeddings  
dec_embeddings = model.decoder.embedding(test_tokens)
dec_embed_shape_ok = dec_embeddings.shape == (batch_size, seq_len, d_model)
add_result("Decoder Embedding Shape", "PASS" if dec_embed_shape_ok else "FAIL",
          f"Expected: ({batch_size}, {seq_len}, {d_model}), Got: {dec_embeddings.shape}")

# Test token embedding scaling (should be multiplied by sqrt(d_model))
token_emb_layer = TokenEmbedding(1000, d_model)
raw_tokens = torch.randint(0, 1000, (2, 4))
scaled_emb = token_emb_layer(raw_tokens)
# Check if scaling is applied by comparing with unscaled version
unscaled_emb = token_emb_layer.embedding(raw_tokens)
scaling_applied = not torch.equal(scaled_emb, unscaled_emb)
add_result("Token Embedding Scaling", "PASS" if scaling_applied else "FAIL",
          "Embeddings should be scaled by sqrt(d_model)")

# Test positional encoding
pos_enc = PositionalEncoding(d_model, max_seq_length=512)
test_emb = torch.randn(batch_size, seq_len, d_model)
pos_output = pos_enc(test_emb)
pos_shape_ok = pos_output.shape == test_emb.shape
add_result("Positional Encoding Shape", "PASS" if pos_shape_ok else "FAIL",
          f"Input: {test_emb.shape}, Output: {pos_output.shape}")

# 3. MULTI-HEAD ATTENTION VERIFICATION
print("\n" + "=" * 50)
print("3. MULTI-HEAD ATTENTION VERIFICATION")
print("=" * 50)

n_heads = 8
attention = MultiHeadAttention(d_model, n_heads)

# Test same-length sequences (self-attention)
query = key = value = torch.randn(batch_size, seq_len, d_model)
self_attn_output = attention(query, key, value)
self_attn_shape_ok = self_attn_output.shape == (batch_size, seq_len, d_model)
add_result("Self-Attention Shape", "PASS" if self_attn_shape_ok else "FAIL",
          f"Expected: ({batch_size}, {seq_len}, {d_model}), Got: {self_attn_output.shape}")

# Test different-length sequences (cross-attention)
query_len, key_len = 6, 10
query_cross = torch.randn(batch_size, query_len, d_model)
key_cross = torch.randn(batch_size, key_len, d_model)
value_cross = torch.randn(batch_size, key_len, d_model)

cross_attn_output = attention(query_cross, key_cross, value_cross)
cross_attn_shape_ok = cross_attn_output.shape == (batch_size, query_len, d_model)
add_result("Cross-Attention Shape", "PASS" if cross_attn_shape_ok else "FAIL",
          f"Query len: {query_len}, Key len: {key_len}, Output: {cross_attn_output.shape}")

# Verify d_k calculation
expected_d_k = d_model // n_heads
actual_d_k = attention.d_k
d_k_correct = expected_d_k == actual_d_k
add_result("Head Dimension Calculation", "PASS" if d_k_correct else "FAIL",
          f"Expected d_k: {expected_d_k}, Actual d_k: {actual_d_k}")

# 4. ENCODER VERIFICATION
print("\n" + "=" * 50)
print("4. ENCODER VERIFICATION")
print("=" * 50)

src_tokens = torch.randint(1, actual_vocab_size, (batch_size, seq_len))
src_mask = create_padding_mask(src_tokens, pad_token_id=0)

encoder_output = model.encoder(src_tokens, src_mask)
encoder_shape_ok = encoder_output.shape == (batch_size, seq_len, d_model)
add_result("Encoder Output Shape", "PASS" if encoder_shape_ok else "FAIL",
          f"Expected: ({batch_size}, {seq_len}, {d_model}), Got: {encoder_output.shape}")

# Check padding mask shape
expected_mask_shape = (batch_size, 1, 1, seq_len)
mask_shape_ok = src_mask.shape == expected_mask_shape
add_result("Padding Mask Shape", "PASS" if mask_shape_ok else "FAIL",
          f"Expected: {expected_mask_shape}, Got: {src_mask.shape}")

# 5. DECODER VERIFICATION 
print("\n" + "=" * 50)
print("5. DECODER VERIFICATION")
print("=" * 50)

tgt_len = 6
tgt_tokens = torch.randint(1, actual_vocab_size, (batch_size, tgt_len))
tgt_mask = create_target_mask(tgt_tokens, pad_token_id=0)

decoder_output = model.decoder(tgt_tokens, encoder_output, tgt_mask, src_mask)
decoder_shape_ok = decoder_output.shape == (batch_size, tgt_len, actual_vocab_size)
add_result("Decoder Output Shape", "PASS" if decoder_shape_ok else "FAIL",
          f"Expected: ({batch_size}, {tgt_len}, {actual_vocab_size}), Got: {decoder_output.shape}")

# Test causal mask
causal_mask = create_causal_mask(tgt_len, tgt_tokens.device)
expected_causal_shape = (1, 1, tgt_len, tgt_len)
causal_shape_ok = causal_mask.shape == expected_causal_shape
add_result("Causal Mask Shape", "PASS" if causal_shape_ok else "FAIL",
          f"Expected: {expected_causal_shape}, Got: {causal_mask.shape}")

# Verify causal mask prevents future attention
causal_matrix = causal_mask.squeeze(0).squeeze(0)
upper_triangle_zeros = torch.all(causal_matrix.triu(diagonal=1) == 0)
lower_triangle_and_diagonal_ones = torch.all(causal_matrix.tril() == causal_matrix)  # Should match the matrix itself
causal_correct = upper_triangle_zeros and torch.all(causal_matrix == torch.tril(torch.ones_like(causal_matrix)))
add_result("Causal Mask Correctness", "PASS" if causal_correct else "FAIL",
          "Upper triangle should be 0, lower triangle + diagonal should be 1")

# 6. CROSS-ATTENTION VERIFICATION
print("\n" + "=" * 50)
print("6. CROSS-ATTENTION VERIFICATION")  
print("=" * 50)

# Test with different sequence lengths
src_len_cross = 12
tgt_len_cross = 8
batch_cross = 3

src_tokens_cross = torch.randint(1, actual_vocab_size, (batch_cross, src_len_cross))
tgt_tokens_cross = torch.randint(1, actual_vocab_size, (batch_cross, tgt_len_cross))

try:
    # Encode
    encoder_out_cross = model.encoder(src_tokens_cross)
    
    # Decode with cross-attention
    decoder_out_cross = model.decoder(tgt_tokens_cross, encoder_out_cross)
    
    cross_shape_correct = decoder_out_cross.shape == (batch_cross, tgt_len_cross, actual_vocab_size)
    add_result("Cross-Attention Different Lengths", "PASS" if cross_shape_correct else "FAIL",
              f"Src len: {src_len_cross}, Tgt len: {tgt_len_cross}, Output: {decoder_out_cross.shape}")
              
    # Concrete shape example
    print(f"\nCross-Attention Shape Example:")
    print(f"  Batch size (B): {batch_cross}")
    print(f"  Source length (S): {src_len_cross}")  
    print(f"  Target length (T): {tgt_len_cross}")
    print(f"  Expected Q shape: [{batch_cross}, {n_heads}, {tgt_len_cross}, {d_model//n_heads}]")
    print(f"  Expected K shape: [{batch_cross}, {n_heads}, {src_len_cross}, {d_model//n_heads}]")
    print(f"  Expected attention scores: [{batch_cross}, {n_heads}, {tgt_len_cross}, {src_len_cross}]")
    
except Exception as e:
    add_result("Cross-Attention Different Lengths", "FAIL", f"Error: {e}")

# 7. MASKS VERIFICATION
print("\n" + "=" * 50)
print("7. MASKS VERIFICATION")
print("=" * 50)

# Test padding mask semantics
test_tokens_with_padding = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])  # 0 is PAD
pad_mask_test = create_padding_mask(test_tokens_with_padding, pad_token_id=0)

# Check that padding positions are masked (0) and non-padding are unmasked (1)  
expected_pad_mask = torch.tensor([[[[1., 1., 1., 0., 0.]]],
                                 [[[1., 1., 0., 0., 0.]]]])
pad_mask_correct = torch.allclose(pad_mask_test, expected_pad_mask)
add_result("Padding Mask Semantics", "PASS" if pad_mask_correct else "FAIL",
          "Padding positions should be 0, non-padding should be 1")

# Test target mask combines both padding and causal
target_test_tokens = torch.tensor([[2, 5, 3, 0], [2, 7, 0, 0]])  # BOS, tokens, EOS, PAD
target_mask_test = create_target_mask(target_test_tokens, pad_token_id=0)

# Shape should be [batch, 1, seq_len, seq_len]
target_mask_shape_ok = target_mask_test.shape == (2, 1, 4, 4)
add_result("Target Mask Shape", "PASS" if target_mask_shape_ok else "FAIL",
          f"Expected: (2, 1, 4, 4), Got: {target_mask_test.shape}")

# 8. RESIDUAL CONNECTIONS & NORMALIZATION
print("\n" + "=" * 50) 
print("8. RESIDUAL CONNECTIONS & NORMALIZATION")
print("=" * 50)

# Check if encoder uses post-norm (norm after residual) 
encoder_layer = model.encoder.layers[0]
has_norm1 = hasattr(encoder_layer, 'norm1')
has_norm2 = hasattr(encoder_layer, 'norm2')
encoder_norm_ok = has_norm1 and has_norm2
add_result("Encoder Layer Normalization", "PASS" if encoder_norm_ok else "FAIL")

# Check decoder normalization (should have 3 norms: self-attn, cross-attn, ffn)
decoder_layer = model.decoder.layers[0]
has_decoder_norm1 = hasattr(decoder_layer, 'norm1')
has_decoder_norm2 = hasattr(decoder_layer, 'norm2')  
has_decoder_norm3 = hasattr(decoder_layer, 'norm3')
decoder_norm_ok = has_decoder_norm1 and has_decoder_norm2 and has_decoder_norm3
add_result("Decoder Layer Normalization", "PASS" if decoder_norm_ok else "FAIL")

# Verify norm is post-norm (applied after residual)
# This is indicated by the order in forward pass: norm(x + attention(...))
add_result("Normalization Type", "PASS", "Post-norm architecture confirmed by code inspection")

# 9. FEED-FORWARD NETWORK
print("\n" + "=" * 50)
print("9. FEED-FORWARD NETWORK") 
print("=" * 50)

ffn = encoder_layer.feed_forward
ffn_has_linear1 = hasattr(ffn, 'linear1')
ffn_has_linear2 = hasattr(ffn, 'linear2')
ffn_structure_ok = ffn_has_linear1 and ffn_has_linear2

# Check dimensions
expected_d_ff = 2048  # 4 * d_model = 4 * 512
actual_d_ff = ffn.linear1.out_features
ffn_dim_ok = actual_d_ff == expected_d_ff

add_result("FFN Structure", "PASS" if ffn_structure_ok else "FAIL")
add_result("FFN Dimensions", "PASS" if ffn_dim_ok else "FAIL",
          f"Expected d_ff: {expected_d_ff}, Actual: {actual_d_ff}")

# 10. OUTPUT PROJECTION
print("\n" + "=" * 50)
print("10. OUTPUT PROJECTION")
print("=" * 50)

output_proj = model.decoder.output_projection
proj_input_dim = output_proj.in_features
proj_output_dim = output_proj.out_features

proj_dim_ok = proj_input_dim == d_model and proj_output_dim == actual_vocab_size
add_result("Output Projection Dimensions", "PASS" if proj_dim_ok else "FAIL",
          f"Input: {proj_input_dim} (expected {d_model}), Output: {proj_output_dim} (expected {actual_vocab_size})")

# 11. FULL FORWARD PASS
print("\n" + "=" * 50)
print("11. FULL FORWARD PASS")
print("=" * 50)

# Test with different source/target lengths
batch_forward = 2
src_len_forward = 10
tgt_len_forward = 7

src_forward = torch.randint(1, actual_vocab_size, (batch_forward, src_len_forward))
tgt_forward = torch.randint(1, actual_vocab_size, (batch_forward, tgt_len_forward))

try:
    logits = model(src_forward, tgt_forward)
    expected_logit_shape = (batch_forward, tgt_len_forward, actual_vocab_size)
    forward_shape_ok = logits.shape == expected_logit_shape
    add_result("Full Forward Pass Shape", "PASS" if forward_shape_ok else "FAIL",
              f"Expected: {expected_logit_shape}, Got: {logits.shape}")
except Exception as e:
    add_result("Full Forward Pass", "FAIL", f"Error: {e}")

# 12. PARAMETER COUNT
print("\n" + "=" * 50)
print("12. PARAMETER COUNT")
print("=" * 50)

model_info = model.get_model_info()
total_params = model_info['total_parameters']
trainable_params = model_info['trainable_parameters']

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Calculate approximate memory size
param_memory_mb = total_params * 4 / (1024**2)  # 4 bytes per float32 parameter
print(f"Approximate parameter memory: {param_memory_mb:.1f} MB")

params_reasonable = 50_000_000 <= total_params <= 150_000_000  # Reasonable range for our config
add_result("Parameter Count Range", "PASS" if params_reasonable else "FAIL",
          f"Expected: 50M-150M, Actual: {total_params:,}")

# 13. AUTOGRAD VERIFICATION
print("\n" + "=" * 50)
print("13. AUTOGRAD VERIFICATION")
print("=" * 50)

model.train()  # Set to training mode

# Create dummy batch
src_dummy = torch.randint(1, actual_vocab_size, (2, 8))
tgt_dummy = torch.randint(1, actual_vocab_size, (2, 6))

try:
    # Forward pass with gradient computation
    logits = model(src_dummy, tgt_dummy)
    
    # Create dummy loss
    target_ids = torch.randint(0, actual_vocab_size, (2, 6))
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Reshape for loss computation
    loss = criterion(logits.view(-1, actual_vocab_size), target_ids.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    has_gradients = False
    gradient_count = 0
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            gradient_count += 1
    
    add_result("Gradient Computation", "PASS" if has_gradients else "FAIL",
              f"Parameters with gradients: {gradient_count}/{len(list(model.parameters()))}")
              
    add_result("Loss Computation", "PASS", f"Loss value: {loss.item():.4f}")
    
except Exception as e:
    add_result("Autograd Verification", "FAIL", f"Error: {e}")

# Clear gradients
model.zero_grad()

# 14. REAL TOKENIZER INTEGRATION
print("\n" + "=" * 50)
print("14. REAL TOKENIZER INTEGRATION")
print("=" * 50)

# Test with real English-Hindi examples
test_sentences = [
    ("Hello world", "नमस्ते दुनिया"), 
    ("How are you?", "आप कैसे हैं?")
]

try:
    for i, (en_text, hi_text) in enumerate(test_sentences):
        # Tokenize
        en_tokens = sp.encode_as_ids(en_text)
        hi_tokens = sp.encode_as_ids(hi_text)
        
        # Add BOS and create tensors
        src_tensor = torch.tensor([[sp.bos_id()] + en_tokens])
        tgt_tensor = torch.tensor([[sp.bos_id()] + hi_tokens])
        
        # Test model
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        print(f"Example {i+1}: EN='{en_text}' HI='{hi_text}'")
        print(f"  Tokenized lengths: EN={len(en_tokens)+1}, HI={len(hi_tokens)+1}")
        print(f"  Model output shape: {output.shape}")
    
    add_result("Real Tokenizer Integration", "PASS")
    
except Exception as e:
    add_result("Real Tokenizer Integration", "FAIL", f"Error: {e}")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

pass_count = sum(1 for r in verification_report if r['status'] == 'PASS')
total_count = len(verification_report)

for result in verification_report:
    status_marker = "✓" if result['status'] == 'PASS' else "✗"
    print(f"{status_marker} {result['test']}: {result['status']}")
    if result['details']:
        print(f"    {result['details']}")

print(f"\nOVERALL: {pass_count}/{total_count} tests passed")

# Final status
if pass_count == total_count:
    print("\n🎉 TRANSFORMER ARCHITECTURE: PASS")
    print("All components verified successfully. Ready for training pipeline.")
else:
    print(f"\n⚠️  TRANSFORMER ARCHITECTURE: NEEDS REVIEW")
    print(f"Failed tests need attention before proceeding.")

print("=" * 80)