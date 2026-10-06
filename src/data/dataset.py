# Dataset loading and preprocessing for NMT training
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as smp
from typing import List, Tuple, Optional
import random

class NMTDataset(Dataset):
    """
    Neural Machine Translation Dataset for English-Hindi translation
    Handles tokenization, padding, and batch preparation
    """
    
    def __init__(self, english_file: str, hindi_file: str, tokenizer_path: str, 
                 max_length: int = 512, bos_token_id: int = 2, eos_token_id: int = 3,
                 pad_token_id: int = 0):
        """
        Args:
            english_file: Path to English sentences file
            hindi_file: Path to Hindi sentences file  
            tokenizer_path: Path to SentencePiece model
            max_length: Maximum sequence length (512 for our model)
            bos_token_id: Beginning of sequence token ID (2)
            eos_token_id: End of sequence token ID (3)
            pad_token_id: Padding token ID (0)
        """
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # Load tokenizer
        self.tokenizer = smp.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Load parallel sentences
        self.english_sentences = self._load_sentences(english_file)
        self.hindi_sentences = self._load_sentences(hindi_file)
        
        # Verify alignment
        assert len(self.english_sentences) == len(self.hindi_sentences), \
            f"Misaligned data: {len(self.english_sentences)} EN vs {len(self.hindi_sentences)} HI"
        
        print(f"Loaded {len(self.english_sentences)} sentence pairs")
        
    def _load_sentences(self, filepath: str) -> List[str]:
        """Load sentences from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]
        return sentences
    
    def _tokenize_and_truncate(self, text: str) -> List[int]:
        """
        Tokenize text and truncate to max_length (accounting for BOS/EOS)
        Returns token IDs without BOS/EOS (added later)
        """
        tokens = self.tokenizer.encode_as_ids(text)
        
        # Truncate to leave space for BOS and EOS
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
            
        return tokens
    
    def __len__(self) -> int:
        return len(self.english_sentences)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training example
        
        Returns:
            dict: {
                'src_tokens': Source tokens with BOS [batch_size, src_len]
                'tgt_input': Target input tokens with BOS [batch_size, tgt_len] 
                'tgt_output': Target output tokens with EOS [batch_size, tgt_len]
                'src_length': Actual source length (before padding)
                'tgt_length': Actual target length (before padding)
            }
        """
        english_text = self.english_sentences[idx]
        hindi_text = self.hindi_sentences[idx]
        
        # Tokenize (without BOS/EOS)
        src_tokens = self._tokenize_and_truncate(english_text)
        tgt_tokens = self._tokenize_and_truncate(hindi_text)
        
        # Add BOS to source  
        src_tokens = [self.bos_token_id] + src_tokens
        
        # For target: input has BOS, output has EOS
        tgt_input = [self.bos_token_id] + tgt_tokens
        tgt_output = tgt_tokens + [self.eos_token_id]
        
        # Store actual lengths (before padding)
        src_length = len(src_tokens)
        tgt_length = len(tgt_input)  # Same as tgt_output length
        
        return {
            'src_tokens': src_tokens,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            'src_length': src_length,
            'tgt_length': tgt_length,
            'english_text': english_text,  # For debugging
            'hindi_text': hindi_text       # For debugging
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function to pad sequences in a batch to same length
    
    Args:
        batch: List of examples from __getitem__
        
    Returns:
        dict: Batched and padded tensors
    """
    pad_token_id = 0  # PAD token ID
    
    # Extract sequences and lengths
    src_tokens = [item['src_tokens'] for item in batch]
    tgt_input = [item['tgt_input'] for item in batch]
    tgt_output = [item['tgt_output'] for item in batch]
    src_lengths = [item['src_length'] for item in batch]
    tgt_lengths = [item['tgt_length'] for item in batch]
    
    # Find max lengths in this batch
    max_src_len = max(len(seq) for seq in src_tokens)
    max_tgt_len = max(len(seq) for seq in tgt_input)
    
    # Pad sequences
    batch_size = len(batch)
    
    # Pad source sequences
    src_padded = torch.full((batch_size, max_src_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(src_tokens):
        src_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    # Pad target input sequences
    tgt_input_padded = torch.full((batch_size, max_tgt_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(tgt_input):
        tgt_input_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    # Pad target output sequences  
    tgt_output_padded = torch.full((batch_size, max_tgt_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(tgt_output):
        tgt_output_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    return {
        'src_tokens': src_padded,           # [batch_size, max_src_len]
        'tgt_input': tgt_input_padded,      # [batch_size, max_tgt_len]  
        'tgt_output': tgt_output_padded,    # [batch_size, max_tgt_len]
        'src_lengths': torch.tensor(src_lengths, dtype=torch.long),
        'tgt_lengths': torch.tensor(tgt_lengths, dtype=torch.long),
        'english_texts': [item['english_text'] for item in batch],
        'hindi_texts': [item['hindi_text'] for item in batch]
    }


def create_data_loaders(train_english_file: str, train_hindi_file: str,
                       val_english_file: str, val_hindi_file: str,
                       tokenizer_path: str, batch_size: int = 32,
                       max_length: int = 512, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        train_english_file: Path to training English sentences
        train_hindi_file: Path to training Hindi sentences
        val_english_file: Path to validation English sentences  
        val_hindi_file: Path to validation Hindi sentences
        tokenizer_path: Path to SentencePiece model
        batch_size: Training batch size (32 for student project)
        max_length: Maximum sequence length (512)
        num_workers: Number of data loading workers (0 for Windows compatibility)
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = NMTDataset(
        english_file=train_english_file,
        hindi_file=train_hindi_file,
        tokenizer_path=tokenizer_path,
        max_length=max_length
    )
    
    val_dataset = NMTDataset(
        english_file=val_english_file,
        hindi_file=val_hindi_file, 
        tokenizer_path=tokenizer_path,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def create_validation_files(dataset_name: str = "cfilt/iitb-english-hindi", 
                          output_dir: str = "data/processed") -> None:
    """
    Create validation files from the original dataset for training
    
    Args:
        dataset_name: Hugging Face dataset name
        output_dir: Directory to save validation files
    """
    import os
    from datasets import load_dataset
    
    print("Loading validation dataset...")
    dataset = load_dataset(dataset_name)
    
    val_english_path = os.path.join(output_dir, "val_en.txt")
    val_hindi_path = os.path.join(output_dir, "val_hi.txt")
    
    # Extract validation examples
    val_examples = dataset['validation']
    
    with open(val_english_path, 'w', encoding='utf-8') as en_file, \
         open(val_hindi_path, 'w', encoding='utf-8') as hi_file:
        
        for example in val_examples:
            en_text = example['translation']['en'].strip()
            hi_text = example['translation']['hi'].strip()
            
            # Apply same normalization as training data
            import unicodedata
            import re
            
            en_text = unicodedata.normalize('NFC', en_text)
            hi_text = unicodedata.normalize('NFC', hi_text)
            en_text = re.sub(r'\s+', ' ', en_text).strip()
            hi_text = re.sub(r'\s+', ' ', hi_text).strip()
            
            # Skip empty pairs
            if en_text and hi_text:
                en_file.write(en_text + '\n')
                hi_file.write(hi_text + '\n')
    
    print(f"Validation files created: {val_english_path}, {val_hindi_path}")


if __name__ == "__main__":
    # Test dataset loading
    train_en = "data/processed/clean_train_en.txt"
    train_hi = "data/processed/clean_train_hi.txt"
    tokenizer_path = "data/processed/tokenizer/sp_model.model"
    
    # Create validation files if they don't exist
    if not os.path.exists("data/processed/val_en.txt"):
        create_validation_files()
    
    # Test dataset
    dataset = NMTDataset(train_en, train_hi, tokenizer_path, max_length=128)
    
    print("Dataset test:")
    example = dataset[0]
    print(f"Source length: {example['src_length']}")
    print(f"Target length: {example['tgt_length']}")
    print(f"English: {example['english_text']}")
    print(f"Hindi: {example['hindi_text']}")
    
    # Test data loader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\nBatch test:")
    print(f"Source shape: {batch['src_tokens'].shape}")
    print(f"Target input shape: {batch['tgt_input'].shape}")
    print(f"Target output shape: {batch['tgt_output'].shape}")