# Remove duplicates from training data
from datasets import load_dataset
import os

print("Loading dataset...")
dataset = load_dataset("cfilt/iitb-english-hindi")

print("Removing duplicates from training data...")

# Track unique pairs
seen_pairs = set()
clean_examples = []
duplicates_removed = 0

for i, example in enumerate(dataset['train']):
    if i % 100000 == 0:
        print(f"Processed {i:,} examples...")
    
    # Create pair tuple
    pair = (example['translation']['en'], example['translation']['hi'])
    
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        clean_examples.append(example)
    else:
        duplicates_removed += 1

original_size = len(dataset['train'])
clean_size = len(clean_examples)

print(f"\nCleaning complete:")
print(f"Original training size: {original_size:,}")
print(f"After deduplication: {clean_size:,}")
print(f"Duplicates removed: {duplicates_removed:,}")
print(f"Reduction: {(duplicates_removed/original_size)*100:.1f}%")

# Save to processed folder
os.makedirs('data/processed', exist_ok=True)

print(f"\nSaving clean examples to data/processed/...")

# Save as simple text files (one per line)
with open('data/processed/clean_train_en.txt', 'w', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'w', encoding='utf-8') as f_hi:
    for example in clean_examples:
        f_en.write(example['translation']['en'] + '\n')
        f_hi.write(example['translation']['hi'] + '\n')

print(f"Saved to:")
print(f"- data/processed/clean_train_en.txt")
print(f"- data/processed/clean_train_hi.txt")
print("Done!")