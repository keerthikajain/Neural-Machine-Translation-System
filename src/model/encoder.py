# Transformer Encoder implementation
import torch
import torch.nn as nn
from .attention import MultiHeadAttention, FeedForwardNetwork

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    Contains: Multi-Head Attention + Feed-Forward + Residual Connections + Layer Norm
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        """
        Forward pass of encoder layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            src_mask: Source padding mask [batch_size, 1, 1, seq_len]
        Returns:
            output: Encoder layer output [batch_size, seq_len, d_model]
        """
        # 1. Multi-head self-attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 2. Feed-forward network with residual connection and layer norm  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder: Stack of N encoder layers
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_length=512, dropout=0.1):
        """
        Args:
            vocab_size: Size of source vocabulary
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads  
            n_layers: Number of encoder layers
            d_ff: Dimension of feed-forward layer
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        # Import embedding here to avoid circular imports
        from .embeddings import TransformerEmbedding
        
        self.d_model = d_model
        
        # Token and positional embeddings
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, src_tokens, src_mask=None):
        """
        Forward pass through encoder
        
        Args:
            src_tokens: Source token IDs [batch_size, src_len]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]
        Returns:
            encoder_output: Encoded representations [batch_size, src_len, d_model]
        """
        # 1. Get embeddings
        x = self.embedding(src_tokens)
        
        # 2. Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x


def create_padding_mask(tokens, pad_token_id=0):
    """
    Create padding mask to ignore padding tokens in attention
    
    Args:
        tokens: Token tensor [batch_size, seq_len]
        pad_token_id: ID of padding token (0 for our SentencePiece)
    Returns:
        mask: Padding mask [batch_size, 1, 1, seq_len]
    """
    # Create mask where padding tokens are False, others are True
    mask = (tokens != pad_token_id).unsqueeze(1).unsqueeze(1).float()
    return mask