# Final train/validation/test leakage check with cleaned training data
from datasets import load_dataset
import re
import unicodedata

def normalize_text(text):
    """Same normalization used in cleaning"""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFC', text)
    normalized = normalized.strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

print("=== FINAL TRAIN/VALIDATION/TEST LEAKAGE CHECK ===")

# Load cleaned training data
print("Loading cleaned training data...")
train_pairs = set()
train_english = set()

with open('data/processed/clean_train_en.txt', 'r', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'r', encoding='utf-8') as f_hi:
    for en_line, hi_line in zip(f_en, f_hi):
        en_normalized = normalize_text(en_line.strip())
        hi_normalized = normalize_text(hi_line.strip())
        
        train_pairs.add((en_normalized, hi_normalized))
        train_english.add(en_normalized)

print(f"Cleaned training pairs: {len(train_pairs):,}")
print(f"Unique English sources: {len(train_english):,}")

# Load original validation and test sets
print("\nLoading original validation and test sets...")
dataset = load_dataset("cfilt/iitb-english-hindi")

# Process validation set
print("\n=== VALIDATION SET OVERLAP CHECK ===")
val_pairs = []
val_english = []

for example in dataset['validation']:
    en_normalized = normalize_text(example['translation']['en'])
    hi_normalized = normalize_text(example['translation']['hi'])
    
    val_pairs.append((en_normalized, hi_normalized))
    val_english.append(en_normalized)

print(f"Validation examples: {len(val_pairs)}")

# Check validation overlaps
val_pair_overlaps = [pair for pair in val_pairs if pair in train_pairs]
val_english_overlaps = [en for en in val_english if en in train_english]

print(f"Exact pair overlaps (train-val): {len(val_pair_overlaps)}")
print(f"English source overlaps (train-val): {len(val_english_overlaps)}")

if val_pair_overlaps:
    print("First few validation pair overlaps:")
    for i, pair in enumerate(val_pair_overlaps[:3]):
        print(f"{i+1}. EN: {pair[0]}")
        print(f"   HI: {pair[1]}")

if val_english_overlaps:
    print("First few validation English overlaps:")
    for i, en in enumerate(val_english_overlaps[:3]):
        print(f"{i+1}. {en}")

# Process test set
print("\n=== TEST SET OVERLAP CHECK ===")
test_pairs = []
test_english = []

for example in dataset['test']:
    en_normalized = normalize_text(example['translation']['en'])
    hi_normalized = normalize_text(example['translation']['hi'])
    
    test_pairs.append((en_normalized, hi_normalized))
    test_english.append(en_normalized)

print(f"Test examples: {len(test_pairs)}")

# Check test overlaps
test_pair_overlaps = [pair for pair in test_pairs if pair in train_pairs]
test_english_overlaps = [en for en in test_english if en in train_english]

print(f"Exact pair overlaps (train-test): {len(test_pair_overlaps)}")
print(f"English source overlaps (train-test): {len(test_english_overlaps)}")

if test_pair_overlaps:
    print("First few test pair overlaps:")
    for i, pair in enumerate(test_pair_overlaps[:3]):
        print(f"{i+1}. EN: {pair[0]}")
        print(f"   HI: {pair[1]}")

if test_english_overlaps:
    print("First few test English overlaps:")
    for i, en in enumerate(test_english_overlaps[:3]):
        print(f"{i+1}. {en}")

# Check validation vs test overlap
print("\n=== VALIDATION VS TEST OVERLAP CHECK ===")
val_test_pair_overlaps = [pair for pair in val_pairs if pair in test_pairs]
val_test_english_overlaps = [en for en in val_english if en in test_english]

print(f"Exact pair overlaps (val-test): {len(val_test_pair_overlaps)}")
print(f"English source overlaps (val-test): {len(val_test_english_overlaps)}")

if val_test_pair_overlaps:
    print("First few val-test pair overlaps:")
    for i, pair in enumerate(val_test_pair_overlaps[:3]):
        print(f"{i+1}. EN: {pair[0]}")
        print(f"   HI: {pair[1]}")

# Final summary
print(f"\n=== LEAKAGE CHECK SUMMARY ===")
print(f"Dataset sizes:")
print(f"  Training: {len(train_pairs):,} pairs")
print(f"  Validation: {len(val_pairs)} pairs") 
print(f"  Test: {len(test_pairs)} pairs")
print()
print(f"Overlap results:")
print(f"  Train-Val exact pairs: {len(val_pair_overlaps)}")
print(f"  Train-Val English sources: {len(val_english_overlaps)}")
print(f"  Train-Test exact pairs: {len(test_pair_overlaps)}")
print(f"  Train-Test English sources: {len(test_english_overlaps)}")
print(f"  Val-Test exact pairs: {len(val_test_pair_overlaps)}")
print(f"  Val-Test English sources: {len(val_test_english_overlaps)}")
print()

# Data integrity assessment
no_train_val_leak = len(val_pair_overlaps) == 0 and len(val_english_overlaps) == 0
no_train_test_leak = len(test_pair_overlaps) == 0 and len(test_english_overlaps) == 0
no_val_test_leak = len(val_test_pair_overlaps) == 0 and len(val_test_english_overlaps) == 0

print(f"Data integrity:")
print(f"  {'✓' if no_train_val_leak else '✗'} No train-validation leakage")
print(f"  {'✓' if no_train_test_leak else '✗'} No train-test leakage")
print(f"  {'✓' if no_val_test_leak else '✗'} No validation-test leakage")
print()

if no_train_val_leak and no_train_test_leak:
    print("✅ DATASET INTEGRITY CONFIRMED - Ready for training!")
else:
    print("⚠️  DATA LEAKAGE DETECTED - Needs attention before training")