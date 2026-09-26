# Train SentencePiece tokenizer on cleaned English-Hindi data
import sentencepiece as spm
import os

print("=== TRAINING SENTENCEPIECE TOKENIZER ===")

# Create tokenizer output directory
tokenizer_dir = 'data/processed/tokenizer'
os.makedirs(tokenizer_dir, exist_ok=True)

# Prepare combined training data for SentencePiece
print("Preparing training data for SentencePiece...")
combined_file = os.path.join(tokenizer_dir, 'combined_train.txt')

# Combine English and Hindi data into single file
with open('data/processed/clean_train_en.txt', 'r', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'r', encoding='utf-8') as f_hi, \
     open(combined_file, 'w', encoding='utf-8') as f_out:
    
    # Add English sentences
    lines_added = 0
    for line in f_en:
        f_out.write(line)
        lines_added += 1
    
    print(f"Added {lines_added:,} English sentences")
    
    # Add Hindi sentences
    f_hi.seek(0)  # Reset file pointer
    for line in f_hi:
        f_out.write(line)
        lines_added += 1
    
    print(f"Total sentences for tokenizer training: {lines_added:,}")

# Vocabulary size choice
vocab_size = 32000
print(f"\nVocabulary size: {vocab_size}")
print("Reasoning: 32K is standard for NMT systems - large enough for good")
print("subword coverage of both English and Hindi, small enough for efficient")
print("training on student hardware. Common choice for multilingual models.")

# SentencePiece training parameters
model_prefix = os.path.join(tokenizer_dir, 'sp_model')

print(f"\nTraining SentencePiece model...")
print("Model type: BPE (Byte Pair Encoding)")
print("Special tokens: <pad>, <unk>, <s>, </s>")

spm.SentencePieceTrainer.train(
    input=combined_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type='bpe',
    character_coverage=0.995,  # Good for multilingual (EN+HI)
    pad_id=0,                  # PAD token
    unk_id=1,                  # UNK token  
    bos_id=2,                  # BOS token (<s>)
    eos_id=3,                  # EOS token (</s>)
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
    user_defined_symbols=[],
    # normalization_rule_name='nfc',  # Use default instead
    remove_extra_whitespaces=True,
    max_sentence_length=1000
)

print(f"✓ SentencePiece model trained successfully!")
print(f"Files created:")
print(f"- {model_prefix}.model")
print(f"- {model_prefix}.vocab")

# Clean up temporary file
os.remove(combined_file)
print(f"✓ Temporary combined file removed")

print(f"\nTokenizer training complete!")