# Check sentence lengths in the dataset
from datasets import load_dataset

dataset = load_dataset("cfilt/iitb-english-hindi")

print("Checking sentence lengths...")

english_lengths = []
hindi_lengths = []

# Check first 5,000 examples for speed
sample_size = min(5000, len(dataset['train']))

for i in range(sample_size):
    example = dataset['train'][i]['translation']
    
    # Count words (split by spaces)
    en_words = len(example['en'].split())
    hi_words = len(example['hi'].split())
    
    english_lengths.append(en_words)
    hindi_lengths.append(hi_words)

# Calculate statistics
en_avg = sum(english_lengths) / len(english_lengths)
hi_avg = sum(hindi_lengths) / len(hindi_lengths)

en_min = min(english_lengths)
en_max = max(english_lengths)
hi_min = min(hindi_lengths)
hi_max = max(hindi_lengths)

print(f"\nResults from {sample_size:,} examples:")
print(f"English - Min: {en_min}, Max: {en_max}, Average: {en_avg:.1f}")
print(f"Hindi   - Min: {hi_min}, Max: {hi_max}, Average: {hi_avg:.1f}")

# Show some examples of different lengths
print(f"\nExample sentences:")
print("SHORT:")
short_idx = english_lengths.index(min(english_lengths[:100]))
example = dataset['train'][short_idx]['translation']
print(f"EN ({len(example['en'].split())} words): {example['en']}")
print(f"HI ({len(example['hi'].split())} words): {example['hi']}")

print("\nMEDIUM:")
medium_idx = 100  # Just pick one from middle
example = dataset['train'][medium_idx]['translation']
print(f"EN ({len(example['en'].split())} words): {example['en']}")
print(f"HI ({len(example['hi'].split())} words): {example['hi']}")