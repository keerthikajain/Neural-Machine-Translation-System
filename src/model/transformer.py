# Complete Transformer Model for Neural Machine Translation
import torch
import torch.nn as nn
from .encoder import TransformerEncoder, create_padding_mask
from .decoder import TransformerDecoder, create_target_mask

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence translation
    Combines encoder and decoder with shared or separate vocabularies
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_seq_length=512, dropout=0.1,
                 pad_token_id=0):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size  
            d_model: Dimension of model embeddings (512 for student model)
            n_heads: Number of attention heads (8 typical)
            n_layers: Number of encoder/decoder layers (6 typical)
            d_ff: Feed-forward dimension (2048 typical)
            max_seq_length: Maximum sequence length (512)
            dropout: Dropout rate (0.1)
            pad_token_id: Padding token ID (0 for SentencePiece)
        """
        super(Transformer, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_token_id = pad_token_id
        
        # Encoder: Processes source sequence (English)
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads, 
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Decoder: Generates target sequence (Hindi) 
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers, 
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """
        Initialize model parameters using Xavier initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src_tokens, tgt_tokens):
        """
        Forward pass for training
        
        Args:
            src_tokens: Source token IDs [batch_size, src_len]
            tgt_tokens: Target token IDs [batch_size, tgt_len]
        Returns:
            logits: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create masks
        src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        tgt_mask = create_target_mask(tgt_tokens, self.pad_token_id)
        
        # Encode source sequence
        encoder_output = self.encoder(src_tokens, src_mask)
        
        # Decode target sequence
        logits = self.decoder(tgt_tokens, encoder_output, tgt_mask, src_mask)
        
        return logits
    
    def encode(self, src_tokens):
        """
        Encode source sequence only (for inference)
        
        Args:
            src_tokens: Source token IDs [batch_size, src_len]
        Returns:
            encoder_output: Encoded representations [batch_size, src_len, d_model]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]
        """
        src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        encoder_output = self.encoder(src_tokens, src_mask)
        return encoder_output, src_mask
    
    def decode_step(self, tgt_tokens, encoder_output, src_mask):
        """
        Single decoding step (for inference)
        
        Args:
            tgt_tokens: Target tokens so far [batch_size, tgt_len]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]
        Returns:
            logits: Next token logits [batch_size, tgt_len, tgt_vocab_size]
        """
        tgt_mask = create_target_mask(tgt_tokens, self.pad_token_id)
        logits = self.decoder(tgt_tokens, encoder_output, tgt_mask, src_mask)
        return logits
    
    def get_model_info(self):
        """
        Get model information for debugging/logging
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'encoder_layers': len(self.encoder.layers),
            'decoder_layers': len(self.decoder.layers)
        }


def create_transformer_model(vocab_size=32000, d_model=512, n_heads=8, n_layers=6):
    """
    Factory function to create Transformer model with sensible defaults for student project
    
    Args:
        vocab_size: Vocabulary size (32000 for our SentencePiece)
        d_model: Model dimension (512 for manageable size)
        n_heads: Attention heads (8 standard)
        n_layers: Number of layers (6 standard)
    Returns:
        model: Initialized Transformer model
    """
    # For our project, source and target share same vocabulary (multilingual SentencePiece)
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,  # Same vocabulary for English and Hindi
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,  # Standard ratio: d_ff = 4 * d_model
        max_seq_length=512,
        dropout=0.1,
        pad_token_id=0
    )
    
    return model