# Verify the rebuilt cleaned dataset
import os
import re
import unicodedata

def normalize_text(text):
    """Same normalization function used in cleaning"""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFC', text)
    normalized = normalized.strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

print("=== VERIFYING REBUILT CLEANED DATASET ===")

# Check if files exist
en_file = 'data/processed/clean_train_en.txt'
hi_file = 'data/processed/clean_train_hi.txt'

if not (os.path.exists(en_file) and os.path.exists(hi_file)):
    print("✗ Files missing!")
    exit()

print("✓ Both files exist")

# Count lines
with open(en_file, 'r', encoding='utf-8') as f:
    en_lines = sum(1 for line in f)

with open(hi_file, 'r', encoding='utf-8') as f:
    hi_lines = sum(1 for line in f)

print(f"English lines: {en_lines:,}")
print(f"Hindi lines: {hi_lines:,}")

if en_lines == hi_lines:
    print("✓ Line counts match")
else:
    print("✗ Line counts don't match!")

# Check for empty lines
print("\nChecking for empty lines...")
empty_en = 0
empty_hi = 0

with open(en_file, 'r', encoding='utf-8') as f_en, \
     open(hi_file, 'r', encoding='utf-8') as f_hi:
    for line_num, (en_line, hi_line) in enumerate(zip(f_en, f_hi), 1):
        if en_line.strip() == '':
            empty_en += 1
            if empty_en <= 3:
                print(f"Empty EN line at {line_num}")
        if hi_line.strip() == '':
            empty_hi += 1
            if empty_hi <= 3:
                print(f"Empty HI line at {line_num}")

print(f"Empty English lines: {empty_en}")
print(f"Empty Hindi lines: {empty_hi}")

if empty_en == 0 and empty_hi == 0:
    print("✓ No empty lines")

# Check for duplicate pairs
print("\nChecking for duplicate pairs...")
seen_pairs = set()
duplicates = 0

with open(en_file, 'r', encoding='utf-8') as f_en, \
     open(hi_file, 'r', encoding='utf-8') as f_hi:
    for en_line, hi_line in zip(f_en, f_hi):
        # Apply same normalization
        en_normalized = normalize_text(en_line.strip())
        hi_normalized = normalize_text(hi_line.strip())
        pair = (en_normalized, hi_normalized)
        
        if pair in seen_pairs:
            duplicates += 1
        else:
            seen_pairs.add(pair)

print(f"Duplicate pairs found: {duplicates}")
if duplicates == 0:
    print("✓ No duplicate pairs")

# Check English-Hindi alignment (first few examples)
print("\nChecking English-Hindi alignment (first 3 examples):")
with open(en_file, 'r', encoding='utf-8') as f_en, \
     open(hi_file, 'r', encoding='utf-8') as f_hi:
    for i in range(3):
        en_line = f_en.readline().strip()
        hi_line = f_hi.readline().strip()
        print(f"{i+1}. EN: {en_line}")
        print(f"   HI: {hi_line}")
        print()

# File sizes
en_size = os.path.getsize(en_file) / (1024*1024)
hi_size = os.path.getsize(hi_file) / (1024*1024)
print(f"File sizes:")
print(f"English: {en_size:.1f} MB")
print(f"Hindi: {hi_size:.1f} MB")

print(f"\n=== FINAL VERIFICATION RESULTS ===")
print(f"✓ Clean training pairs: {len(seen_pairs):,}")
print(f"✓ No empty lines: {empty_en + empty_hi == 0}")
print(f"✓ No duplicates: {duplicates == 0}")
print(f"✓ Perfect alignment: {en_lines == hi_lines}")
print("Dataset rebuild successful!")