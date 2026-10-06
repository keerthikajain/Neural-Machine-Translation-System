# Multi-Head Attention implementation for Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    Core component of Transformer architecture
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: Dimension of model embeddings (512)
            n_heads: Number of attention heads (8 typical)
            dropout: Dropout rate for attention weights
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)  
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            Q: Queries [batch_size, n_heads, seq_len, d_k]
            K: Keys [batch_size, n_heads, seq_len, d_k] 
            V: Values [batch_size, n_heads, seq_len, d_k]
            mask: Attention mask [batch_size, 1, seq_len, seq_len]
        Returns:
            output: Attention output [batch_size, n_heads, seq_len, d_k]
            attention_weights: Attention scores [batch_size, n_heads, seq_len, seq_len]
        """
        # Calculate attention scores
        # scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            query: Query tensor [batch_size, query_len, d_model]
            key: Key tensor [batch_size, key_len, d_model]
            value: Value tensor [batch_size, value_len, d_model]  
            mask: Attention mask [batch_size, 1, query_len, key_len]
        Returns:
            output: Multi-head attention output [batch_size, query_len, d_model]
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # 1. Linear projections for Q, K, V
        Q = self.w_q(query)  # [batch_size, query_len, d_model]
        K = self.w_k(key)    # [batch_size, key_len, d_model] 
        V = self.w_v(value)  # [batch_size, value_len, d_model]
        
        # 2. Reshape to multi-head format
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        # [batch_size, n_heads, query_len, d_k] -> [batch_size, query_len, n_heads * d_k]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.n_heads * self.d_k
        )
        
        # 5. Final linear projection
        output = self.w_o(attention_output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    Two linear layers with ReLU activation
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of model embeddings (512)
            d_ff: Dimension of feed-forward layer (2048 typical)
            dropout: Dropout rate
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass: Linear -> ReLU -> Dropout -> Linear
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            output: FFN output [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))