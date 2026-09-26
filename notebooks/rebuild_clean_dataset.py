# Rebuild cleaned training dataset from original IITB data
from datasets import load_dataset
import re
import unicodedata
import os

def normalize_text(text):
    """
    Normalize text with NFC Unicode normalization and whitespace cleaning
    """
    if not text:
        return ""
    
    # Apply NFC Unicode normalization
    normalized = unicodedata.normalize('NFC', text)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    # Collapse repeated whitespace, tabs, and internal line breaks into single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def is_valid_pair(en_text, hi_text):
    """
    Check if translation pair is valid after normalization
    """
    en_normalized = normalize_text(en_text)
    hi_normalized = normalize_text(hi_text)
    
    return len(en_normalized) > 0 and len(hi_normalized) > 0

print("Loading original IITB English-Hindi dataset...")
dataset = load_dataset("cfilt/iitb-english-hindi")

print(f"Original training size: {len(dataset['train']):,}")

print("\nProcessing and cleaning training data...")

seen_pairs = set()
clean_pairs = []
empty_pairs_removed = 0
duplicates_removed = 0

for i, example in enumerate(dataset['train']):
    if i % 100000 == 0:
        print(f"Processed {i:,} examples...")
    
    en_raw = example['translation']['en']
    hi_raw = example['translation']['hi']
    
    # Check if pair is valid (not empty after normalization)
    if not is_valid_pair(en_raw, hi_raw):
        empty_pairs_removed += 1
        continue
    
    # Normalize for deduplication key
    en_normalized = normalize_text(en_raw)
    hi_normalized = normalize_text(hi_raw)
    pair_key = (en_normalized, hi_normalized)
    
    # Check for duplicates using normalized pair
    if pair_key not in seen_pairs:
        seen_pairs.add(pair_key)
        # Store normalized versions for saving
        clean_pairs.append({
            'en': en_normalized,
            'hi': hi_normalized
        })
    else:
        duplicates_removed += 1

original_size = len(dataset['train'])
final_size = len(clean_pairs)

print(f"\nCleaning results:")
print(f"Original size: {original_size:,}")
print(f"Empty pairs removed: {empty_pairs_removed:,}")
print(f"Duplicates removed: {duplicates_removed:,}")
print(f"Final clean size: {final_size:,}")
print(f"Total reduction: {((original_size - final_size) / original_size * 100):.1f}%")

# Create output directory
os.makedirs('data/processed', exist_ok=True)

print(f"\nSaving cleaned data to data/processed/...")

# Save normalized cleaned pairs
with open('data/processed/clean_train_en.txt', 'w', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'w', encoding='utf-8') as f_hi:
    for pair in clean_pairs:
        f_en.write(pair['en'] + '\n')
        f_hi.write(pair['hi'] + '\n')

print("Saved to:")
print("- data/processed/clean_train_en.txt")
print("- data/processed/clean_train_hi.txt")
print("\nRebuild complete!")