# Transformer Decoder implementation
import torch
import torch.nn as nn
from .attention import MultiHeadAttention, FeedForwardNetwork

class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    Contains: Masked Self-Attention + Encoder-Decoder Attention + Feed-Forward + Residual + LayerNorm
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        
        # Masked self-attention (for target sequence)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Encoder-decoder attention (attending to encoder output)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass of decoder layer
        
        Args:
            x: Target embeddings [batch_size, tgt_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target causal mask [batch_size, 1, tgt_len, tgt_len]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]
        Returns:
            output: Decoder layer output [batch_size, tgt_len, d_model]
        """
        # 1. Masked self-attention on target sequence
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Encoder-decoder attention
        # Query from decoder, Key and Value from encoder
        enc_attn_output = self.encoder_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(enc_attn_output))
        
        # 3. Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder: Stack of N decoder layers + output projection
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_seq_length=512, dropout=0.1):
        """
        Args:
            vocab_size: Size of target vocabulary
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Dimension of feed-forward layer
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(TransformerDecoder, self).__init__()
        
        # Import embedding here to avoid circular imports
        from .embeddings import TransformerEmbedding
        
        self.d_model = d_model
        
        # Token and positional embeddings
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt_tokens, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass through decoder
        
        Args:
            tgt_tokens: Target token IDs [batch_size, tgt_len]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target causal mask [batch_size, 1, tgt_len, tgt_len]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]
        Returns:
            logits: Output logits [batch_size, tgt_len, vocab_size]
        """
        # 1. Get target embeddings
        x = self.embedding(tgt_tokens)
        
        # 2. Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        # 3. Project to vocabulary size
        logits = self.output_projection(x)
        
        return logits


def create_causal_mask(seq_len, device):
    """
    Create causal (look-ahead) mask for decoder self-attention
    Prevents positions from attending to subsequent positions
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    Returns:
        mask: Causal mask [1, 1, seq_len, seq_len]
    """
    # Create lower triangular matrix (1s below and on diagonal, 0s above)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.float32))
    
    # Return with batch and head dimensions
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt_tokens, pad_token_id=0):
    """
    Create combined padding and causal mask for target sequence
    
    Args:
        tgt_tokens: Target token tensor [batch_size, tgt_len]
        pad_token_id: ID of padding token
    Returns:
        mask: Combined mask [batch_size, 1, tgt_len, tgt_len]
    """
    batch_size, tgt_len = tgt_tokens.size()
    device = tgt_tokens.device
    
    # Create padding mask
    pad_mask = (tgt_tokens != pad_token_id).unsqueeze(1).float()  # [batch_size, 1, tgt_len]
    
    # Create causal mask  
    causal_mask = create_causal_mask(tgt_len, device).squeeze(0).squeeze(0)  # [tgt_len, tgt_len]
    
    # Expand padding mask to match causal mask
    pad_mask_expanded = pad_mask.unsqueeze(-1)  # [batch_size, 1, tgt_len, 1]
    causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
    
    # Combine masks
    combined_mask = pad_mask_expanded * causal_mask_expanded  # [batch_size, 1, tgt_len, tgt_len]
    
    return combined_mask