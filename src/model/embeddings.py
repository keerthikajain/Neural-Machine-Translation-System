# Token and Positional Embeddings for Transformer
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors
    """
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Size of vocabulary (32000 for our SentencePiece)
            d_model: Dimension of model embeddings (512 typical for student model)
        """
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, tokens):
        """
        Convert token IDs to embeddings
        
        Args:
            tokens: Token IDs tensor [batch_size, seq_len]
        Returns:
            embeddings: Token embeddings [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as in original Transformer paper
        return self.embedding(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions
    Adds position information to token embeddings
    """
    def __init__(self, d_model, max_seq_length=512):
        """
        Args:
            d_model: Dimension of model embeddings
            max_seq_length: Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Create div_term for sine/cosine calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerEmbedding(nn.Module):
    """
    Combined token and positional embeddings with dropout
    """
    def __init__(self, vocab_size, d_model, max_seq_length=512, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary 
            d_model: Dimension of model embeddings
            max_seq_length: Maximum sequence length
            dropout: Dropout rate for regularization
        """
        super(TransformerEmbedding, self).__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens):
        """
        Convert tokens to embeddings with positional encoding
        
        Args:
            tokens: Token IDs [batch_size, seq_len]
        Returns:
            embeddings: Combined embeddings [batch_size, seq_len, d_model]
        """
        # Get token embeddings
        token_emb = self.token_embedding(tokens)
        
        # Add positional encoding
        pos_emb = self.positional_encoding(token_emb)
        
        # Apply dropout
        return self.dropout(pos_emb)